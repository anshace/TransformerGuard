"""
Report Endpoints
API routes for PDF report generation and download
"""

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.database.connection import DatabaseConnection
from src.database.models import Alert, DGARecord, HealthIndexRecord, Transformer
from src.reporting import PDFReportGenerator, ReportConfig

from ..schemas.health import HealthIndexResponse


def get_db():
    """Dependency for database session."""
    db_connection = DatabaseConnection.get_instance()
    db = db_connection.get_session()
    try:
        yield db
    finally:
        db.close()


router = APIRouter()


# In-memory storage for generated reports (in production, use database or file storage)
_generated_reports = {}


class ReportGenerateRequest(BaseModel):
    """Schema for report generation request."""

    format: str = Query("PDF", description="Report format: PDF or HTML")
    include_charts: bool = Query(True, description="Include charts in report")
    include_recommendations: bool = Query(True, description="Include recommendations")
    include_historical_data: bool = Query(True, description="Include historical data")


class ReportGenerateResponse(BaseModel):
    """Schema for report generation response."""

    report_id: str
    transformer_id: int
    transformer_name: str
    status: str
    message: str
    created_at: datetime


class ReportListItem(BaseModel):
    """Schema for report list item."""

    report_id: str
    transformer_id: int
    transformer_name: str
    format: str
    created_at: datetime
    file_path: Optional[str] = None


class ReportListResponse(BaseModel):
    """Schema for report list response."""

    reports: List[ReportListItem]
    total_count: int


def _get_transformer_data(transformer: Transformer, db: Session) -> dict:
    """Get transformer data for report."""
    return {
        "id": transformer.id,
        "name": transformer.name,
        "serial_number": transformer.serial_number,
        "manufacturer": transformer.manufacturer,
        "manufacture_date": transformer.manufacture_date.isoformat()
        if transformer.manufacture_date
        else None,
        "installation_date": transformer.installation_date.isoformat()
        if transformer.installation_date
        else None,
        "rated_mva": transformer.rated_mva,
        "rated_voltage_kv": transformer.rated_voltage_kv,
        "cooling_type": transformer.cooling_type,
        "location": transformer.location,
        "substation": transformer.substation,
    }


def _get_dga_results(transformer_id: int, db: Session) -> dict:
    """Get DGA results for report."""
    # Get recent DGA records
    dga_records = (
        db.query(DGARecord)
        .filter(DGARecord.transformer_id == transformer_id)
        .order_by(DGARecord.sample_date.desc())
        .limit(10)
        .all()
    )

    latest = dga_records[0] if dga_records else None

    return {
        "latest": {
            "sample_date": latest.sample_date.isoformat() if latest else None,
            "tdcg": latest.tdcg if latest else None,
            "fault_type": latest.fault_type if latest else None,
            "fault_confidence": latest.fault_confidence if latest else None,
            "diagnosis_method": latest.diagnosis_method if latest else None,
            "h2": latest.h2 if latest else None,
            "ch4": latest.ch4 if latest else None,
            "c2h2": latest.c2h2 if latest else None,
            "c2h4": latest.c2h4 if latest else None,
            "c2h6": latest.c2h6 if latest else None,
            "co": latest.co if latest else None,
            "co2": latest.co2 if latest else None,
        }
        if latest
        else None,
        "history": [
            {
                "sample_date": r.sample_date.isoformat(),
                "tdcg": r.tdcg,
                "fault_type": r.fault_type,
            }
            for r in dga_records
        ],
    }


def _get_health_results(transformer_id: int, db: Session) -> dict:
    """Get health index results for report."""
    # Get latest health index
    latest_health = (
        db.query(HealthIndexRecord)
        .filter(HealthIndexRecord.transformer_id == transformer_id)
        .order_by(HealthIndexRecord.calculation_date.desc())
        .first()
    )

    # Get health history
    health_records = (
        db.query(HealthIndexRecord)
        .filter(HealthIndexRecord.transformer_id == transformer_id)
        .order_by(HealthIndexRecord.calculation_date.desc())
        .limit(12)
        .all()
    )

    return {
        "latest": {
            "health_index": latest_health.health_index if latest_health else None,
            "category": latest_health.category if latest_health else None,
            "dga_score": latest_health.dga_score if latest_health else None,
            "oil_quality_score": latest_health.oil_quality_score
            if latest_health
            else None,
            "electrical_score": latest_health.electrical_score
            if latest_health
            else None,
            "age_score": latest_health.age_score if latest_health else None,
            "loading_score": latest_health.loading_score if latest_health else None,
            "calculation_date": latest_health.calculation_date.isoformat()
            if latest_health
            else None,
        }
        if latest_health
        else None,
        "history": [
            {
                "calculation_date": r.calculation_date.isoformat(),
                "health_index": r.health_index,
                "category": r.category,
            }
            for r in health_records
        ],
    }


def _get_alerts(transformer_id: int, db: Session) -> List[dict]:
    """Get active alerts for report."""
    alerts = (
        db.query(Alert)
        .filter(Alert.transformer_id == transformer_id, Alert.acknowledged == False)
        .order_by(Alert.created_at.desc())
        .limit(10)
        .all()
    )

    return [
        {
            "priority": a.priority,
            "category": a.category,
            "title": a.title,
            "message": a.message,
            "created_at": a.created_at.isoformat(),
        }
        for a in alerts
    ]


@router.post("/generate/{transformer_id}", response_model=ReportGenerateResponse)
async def generate_report(
    transformer_id: int,
    format: str = Query("PDF", description="Report format: PDF or HTML"),
    include_charts: bool = Query(True),
    include_recommendations: bool = Query(True),
    include_historical_data: bool = Query(True),
    db: Session = Depends(get_db),
):
    """
    Generate a PDF/HTML report for a transformer.

    - **transformer_id**: ID of the transformer
    - **format**: Report format (PDF or HTML)
    - **include_charts**: Include charts in report
    - **include_recommendations**: Include recommendations
    - **include_historical_data**: Include historical data
    """
    # Verify transformer exists
    transformer = db.query(Transformer).filter(Transformer.id == transformer_id).first()
    if not transformer:
        raise HTTPException(status_code=404, detail="Transformer not found")

    # Generate report ID
    report_id = str(uuid.uuid4())

    # Prepare report data
    transformer_data = _get_transformer_data(transformer, db)
    dga_results = _get_dga_results(transformer_id, db)
    health_results = _get_health_results(transformer_id, db)
    alerts = _get_alerts(transformer_id, db)

    # Create config
    config = ReportConfig(
        format=format.upper() if format.upper() in ["PDF", "HTML"] else "PDF",
        include_charts=include_charts,
        include_recommendations=include_recommendations,
        include_historical_data=include_historical_data,
        report_title=f"Transformer Condition Assessment Report - {transformer.name}",
    )

    # Generate report
    try:
        report_generator = PDFReportGenerator(config)

        # Use temp directory for reports
        output_dir = Path("/tmp/transformerguard_reports")
        output_dir.mkdir(parents=True, exist_ok=True)

        config.output_path = str(
            output_dir / f"report_{transformer_id}_{report_id[:8]}.{format.lower()}"
        )

        report_path = report_generator.generate_report(
            transformer_data=transformer_data,
            dga_results=dga_results,
            health_results=health_results,
            alerts=alerts,
        )

        # Store report info
        _generated_reports[report_id] = {
            "report_id": report_id,
            "transformer_id": transformer_id,
            "transformer_name": transformer.name,
            "format": format.upper(),
            "file_path": report_path,
            "created_at": datetime.utcnow(),
        }

        return ReportGenerateResponse(
            report_id=report_id,
            transformer_id=transformer_id,
            transformer_name=transformer.name,
            status="completed",
            message=f"Report generated successfully at {report_path}",
            created_at=datetime.utcnow(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate report: {str(e)}"
        )


@router.get("/download/{report_id}")
async def download_report(
    report_id: str,
):
    """
    Download a generated report.

    - **report_id**: ID of the report to download
    """
    if report_id not in _generated_reports:
        raise HTTPException(status_code=404, detail="Report not found")

    report_info = _generated_reports[report_id]
    file_path = report_info.get("file_path")

    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Report file not found")

    # Determine media type
    media_type = "application/pdf" if report_info["format"] == "PDF" else "text/html"

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=f"transformer_report_{report_info['transformer_id']}.{report_info['format'].lower()}",
    )


@router.get("/list", response_model=ReportListResponse)
async def list_reports(
    transformer_id: Optional[int] = Query(None, description="Filter by transformer ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
):
    """
    List generated reports.

    - **transformer_id**: Optional filter by transformer ID
    - **skip**: Number of records to skip
    - **limit**: Maximum number of records to return
    """
    reports = list(_generated_reports.values())

    # Filter by transformer if specified
    if transformer_id is not None:
        reports = [r for r in reports if r["transformer_id"] == transformer_id]

    # Sort by creation date (newest first)
    reports.sort(key=lambda x: x["created_at"], reverse=True)

    # Apply pagination
    total_count = len(reports)
    paginated_reports = reports[skip : skip + limit]

    return ReportListResponse(
        reports=[
            ReportListItem(
                report_id=r["report_id"],
                transformer_id=r["transformer_id"],
                transformer_name=r["transformer_name"],
                format=r["format"],
                created_at=r["created_at"],
                file_path=r.get("file_path"),
            )
            for r in paginated_reports
        ],
        total_count=total_count,
    )


@router.get("/{report_id}")
async def get_report_info(
    report_id: str,
):
    """
    Get information about a generated report.

    - **report_id**: ID of the report
    """
    if report_id not in _generated_reports:
        raise HTTPException(status_code=404, detail="Report not found")

    report_info = _generated_reports[report_id]

    return {
        "report_id": report_info["report_id"],
        "transformer_id": report_info["transformer_id"],
        "transformer_name": report_info["transformer_name"],
        "format": report_info["format"],
        "created_at": report_info["created_at"].isoformat(),
        "file_path": report_info.get("file_path"),
        "exists": os.path.exists(report_info.get("file_path", ""))
        if report_info.get("file_path")
        else False,
    }
