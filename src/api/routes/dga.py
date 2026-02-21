"""
DGA (Dissolved Gas Analysis) Endpoints
API routes for DGA analysis and records
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from src.database.connection import DatabaseConnection
from src.database.models import DGARecord, Transformer
from src.diagnosis import (
    Doernenburg,
    DuvalTriangle1,
    IecRatios,
    KeyGasMethod,
    MultiMethodDiagnosis,
    RogersRatios,
)

from ..schemas.dga import (
    BatchDGAAnalysisResponse,
    BatchRequestDGAAnalysis,
    DGAAnalysisRequest,
    DGAAnalysisResult,
    DGACreate,
    DGAHistoryResponse,
    DGAInput,
    DGAResponse,
)


def get_db():
    """Dependency for database session."""
    db_connection = DatabaseConnection.get_instance()
    db = db_connection.get_session()
    try:
        yield db
    finally:
        db.close()


router = APIRouter()


def _calculate_tdcg(
    h2: float,
    ch4: float,
    c2h2: float,
    c2h4: float,
    c2h6: float,
    co: float,
    co2: float,
    o2: Optional[float] = None,
    n2: Optional[float] = None,
) -> float:
    """Calculate Total Dissolved Combustible Gas."""
    tdcg = h2 + ch4 + c2h2 + c2h4 + c2h6 + co + co2
    if o2 is not None:
        tdcg += o2
    if n2 is not None:
        tdcg += n2
    return tdcg


def _run_diagnosis(dga_input: DGAInput, method: str = "multi") -> DGAAnalysisResult:
    """Run DGA diagnosis using specified method."""

    gas_values = {
        "h2": dga_input.h2,
        "ch4": dga_input.ch4,
        "c2h2": dga_input.c2h2,
        "c2h4": dga_input.c2h4,
        "c2h6": dga_input.c2h6,
        "co": dga_input.co,
        "co2": dga_input.co2,
    }

    method_results = {}

    if method == "multi":
        diagnosis = MultiMethodDiagnosis()
        result = diagnosis.diagnose(**gas_values)
        method_results["multi"] = {
            "fault_type": result.fault_type.value,
            "confidence": result.confidence,
            "explanation": result.explanation,
        }
    elif method == "duval":
        diagnosis = DuvalTriangle1()
        result = diagnosis.diagnose(**gas_values)
        method_results["duval"] = {
            "fault_type": result.fault_type.value,
            "confidence": result.confidence,
        }
    elif method == "rogers":
        diagnosis = RogersRatios()
        result = diagnosis.diagnose(**gas_values)
        method_results["rogers"] = {
            "fault_type": result.fault_type.value,
            "confidence": result.confidence,
        }
    elif method == "iec":
        diagnosis = IecRatios()
        result = diagnosis.diagnose(**gas_values)
        method_results["iec"] = {
            "fault_type": result.fault_type.value,
            "confidence": result.confidence,
        }
    elif method == "doernenburg":
        diagnosis = Doernenburg()
        result = diagnosis.diagnose(**gas_values)
        method_results["doernenburg"] = {
            "fault_type": result.fault_type.value,
            "confidence": result.confidence,
        }
    elif method == "key_gas":
        diagnosis = KeyGasMethod()
        result = diagnosis.diagnose(**gas_values)
        method_results["key_gas"] = {
            "fault_type": result.fault_type.value,
            "confidence": result.confidence,
        }
    else:
        raise HTTPException(
            status_code=400, detail=f"Unknown diagnosis method: {method}"
        )

    # Get the primary result
    primary_result = method_results.get(method) or method_results.get(
        list(method_results.keys())[0]
    )

    return DGAAnalysisResult(
        fault_type=primary_result["fault_type"],
        fault_confidence=primary_result["confidence"],
        tdcg=_calculate_tdcg(**gas_values),
        explanation=primary_result.get(
            "explanation", f"Fault type: {primary_result['fault_type']}"
        ),
        method_results=method_results,
    )


@router.post("/analyze", response_model=DGAAnalysisResult)
async def analyze_dga(
    analysis_request: DGAAnalysisRequest,
):
    """
    Analyze a DGA sample without saving to database.

    - **analysis_request**: DGA sample data and analysis parameters
    """
    # Verify transformer exists
    db_connection = DatabaseConnection.get_instance()
    with db_connection.session_scope() as db:
        transformer = (
            db.query(Transformer)
            .filter(Transformer.id == analysis_request.transformer_id)
            .first()
        )
        if not transformer:
            raise HTTPException(status_code=404, detail="Transformer not found")

    # Create DGA input
    dga_input = DGAInput(
        transformer_id=analysis_request.transformer_id,
        sample_date=analysis_request.sample_date,
        h2=analysis_request.h2,
        ch4=analysis_request.ch4,
        c2h2=analysis_request.c2h2,
        c2h4=analysis_request.c2h4,
        c2h6=analysis_request.c2h6,
        co=analysis_request.co,
        co2=analysis_request.co2,
        o2=analysis_request.o2,
        n2=analysis_request.n2,
    )

    # Run diagnosis
    return _run_diagnosis(dga_input, analysis_request.method)


@router.post("/upload", response_model=List[DGAResponse])
async def upload_dga_file(
    transformer_id: int,
    sample_type: str = Query("Routine"),
    db: Session = Depends(get_db),
    file: UploadFile = File(...),
):
    """
    Upload DGA data from CSV or Excel file.

    - **transformer_id**: ID of the transformer
    - **sample_type**: Type of sample (Routine, Emergency, Follow-up)
    - **file**: CSV or Excel file with DGA data
    """
    # Verify transformer exists
    transformer = db.query(Transformer).filter(Transformer.id == transformer_id).first()
    if not transformer:
        raise HTTPException(status_code=404, detail="Transformer not found")

    # Read file content
    content = await file.read()

    records = []
    errors = []

    try:
        # Try to parse as CSV
        import csv
        import io

        lines = content.decode("utf-8").strip().split("\n")
        reader = csv.DictReader(lines)

        for row_num, row in enumerate(reader, start=2):
            try:
                # Parse the row
                sample_date = datetime.fromisoformat(row.get("sample_date", "").strip())

                dga_input = DGAInput(
                    transformer_id=transformer_id,
                    sample_date=sample_date,
                    h2=float(row.get("h2", 0)),
                    ch4=float(row.get("ch4", 0)),
                    c2h2=float(row.get("c2h2", 0)),
                    c2h4=float(row.get("c2h4", 0)),
                    c2h6=float(row.get("c2h6", 0)),
                    co=float(row.get("co", 0)),
                    co2=float(row.get("co2", 0)),
                    o2=float(row.get("o2", 0)) if row.get("o2") else None,
                    n2=float(row.get("n2", 0)) if row.get("n2") else None,
                    sample_type=sample_type,
                )

                # Calculate TDCG
                tdcg = _calculate_tdcg(
                    dga_input.h2,
                    dga_input.ch4,
                    dga_input.c2h2,
                    dga_input.c2h4,
                    dga_input.c2h6,
                    dga_input.co,
                    dga_input.co2,
                    dga_input.o2,
                    dga_input.n2,
                )

                # Run diagnosis
                analysis_result = _run_diagnosis(dga_input)

                # Create record
                db_record = DGARecord(
                    transformer_id=transformer_id,
                    sample_date=sample_date,
                    h2=dga_input.h2,
                    ch4=dga_input.ch4,
                    c2h2=dga_input.c2h2,
                    c2h4=dga_input.c2h4,
                    c2h6=dga_input.c2h6,
                    co=dga_input.co,
                    co2=dga_input.co2,
                    o2=dga_input.o2,
                    n2=dga_input.n2,
                    tdcg=tdcg,
                    fault_type=analysis_result.fault_type,
                    fault_confidence=analysis_result.fault_confidence,
                    diagnosis_method="multi",
                    sample_type=sample_type,
                )

                db.add(db_record)
                records.append(dga_input)

            except Exception as e:
                errors.append(f"Row {row_num}: {str(e)}")

        db.commit()

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

    # Return saved records
    saved_records = (
        db.query(DGARecord)
        .filter(DGARecord.transformer_id == transformer_id)
        .order_by(DGARecord.sample_date.desc())
        .limit(len(records))
        .all()
    )

    return saved_records


@router.get("/history/{transformer_id}", response_model=DGAHistoryResponse)
async def get_dga_history(
    transformer_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """
    Get DGA history for a transformer.

    - **transformer_id**: ID of the transformer
    - **skip**: Number of records to skip
    - **limit**: Maximum number of records to return
    """
    # Verify transformer exists
    transformer = db.query(Transformer).filter(Transformer.id == transformer_id).first()
    if not transformer:
        raise HTTPException(status_code=404, detail="Transformer not found")

    records = (
        db.query(DGARecord)
        .filter(DGARecord.transformer_id == transformer_id)
        .order_by(DGARecord.sample_date.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    total_count = (
        db.query(DGARecord).filter(DGARecord.transformer_id == transformer_id).count()
    )

    return DGAHistoryResponse(
        transformer_id=transformer_id,
        records=records,
        total_count=total_count,
    )


@router.get("/records/{record_id}", response_model=DGAResponse)
async def get_dga_record(
    record_id: int,
    db: Session = Depends(get_db),
):
    """
    Get a specific DGA record.

    - **record_id**: ID of the DGA record
    """
    record = db.query(DGARecord).filter(DGARecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="DGA record not found")
    return record


@router.post("/batch-analyze", response_model=BatchDGAAnalysisResponse)
async def batch_analyze(
    batch_request: BatchRequestDGAAnalysis,
):
    """
    Batch analyze multiple DGA samples.

    - **batch_request**: List of DGA samples to analyze
    """
    results = []
    errors = []

    for i, dga_input in enumerate(batch_request.samples):
        try:
            # Run diagnosis
            analysis_result = _run_diagnosis(dga_input)

            # Create response record
            tdcg = _calculate_tdcg(
                dga_input.h2,
                dga_input.ch4,
                dga_input.c2h2,
                dga_input.c2h4,
                dga_input.c2h6,
                dga_input.co,
                dga_input.co2,
                dga_input.o2,
                dga_input.n2,
            )

            results.append(
                DGAResponse(
                    id=0,  # Placeholder for analysis-only results
                    transformer_id=dga_input.transformer_id,
                    sample_date=dga_input.sample_date,
                    h2=dga_input.h2,
                    ch4=dga_input.ch4,
                    c2h2=dga_input.c2h2,
                    c2h4=dga_input.c2h4,
                    c2h6=dga_input.c2h6,
                    co=dga_input.co,
                    co2=dga_input.co2,
                    o2=dga_input.o2,
                    n2=dga_input.n2,
                    tdcg=tdcg,
                    fault_type=analysis_result.fault_type,
                    fault_confidence=analysis_result.fault_confidence,
                    diagnosis_method="multi",
                    lab_name=dga_input.lab_name,
                    sample_type=dga_input.sample_type,
                    created_at=datetime.utcnow(),
                )
            )

        except Exception as e:
            errors.append(f"Sample {i}: {str(e)}")

    return BatchDGAAnalysisResponse(
        results=results,
        total_analyzed=len(results),
        errors=errors,
    )


@router.post("", response_model=DGAResponse, status_code=201)
async def create_dga_record(
    dga_record: DGACreate,
    db: Session = Depends(get_db),
):
    """
    Create a new DGA record.

    - **dga_record**: DGA record data to create
    """
    # Verify transformer exists
    transformer = (
        db.query(Transformer)
        .filter(Transformer.id == dga_record.transformer_id)
        .first()
    )
    if not transformer:
        raise HTTPException(status_code=404, detail="Transformer not found")

    # Calculate TDCG
    tdcg = _calculate_tdcg(
        dga_record.h2,
        dga_record.ch4,
        dga_record.c2h2,
        dga_record.c2h4,
        dga_record.c2h6,
        dga_record.co,
        dga_record.co2,
        dga_record.o2,
        dga_record.n2,
    )

    # Run diagnosis
    analysis_result = _run_diagnosis(dga_record)

    # Create record
    db_record = DGARecord(
        **dga_record.model_dump(),
        tdcg=tdcg,
        fault_type=analysis_result.fault_type,
        fault_confidence=analysis_result.fault_confidence,
        diagnosis_method="multi",
    )

    db.add(db_record)
    db.commit()
    db.refresh(db_record)

    return db_record
