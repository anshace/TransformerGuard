"""
Transformer CRUD Endpoints
API routes for transformer management
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.database.connection import DatabaseConnection
from src.database.models import Alert, DGARecord, HealthIndexRecord, Transformer

from ..schemas.transformer import (
    TransformerCreate,
    TransformerResponse,
    TransformerSummary,
    TransformerUpdate,
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


@router.get("", response_model=List[TransformerResponse])
async def list_transformers(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """
    List all transformers with pagination and optional search.

    - **skip**: Number of records to skip (for pagination)
    - **limit**: Maximum number of records to return
    - **search**: Optional search term for name or serial number
    """
    query = db.query(Transformer)

    if search:
        search_term = f"%{search}%"
        query = query.filter(
            (Transformer.name.ilike(search_term))
            | (Transformer.serial_number.ilike(search_term))
        )

    transformers = query.order_by(Transformer.name).offset(skip).limit(limit).all()
    return transformers


@router.get("/{transformer_id}", response_model=TransformerResponse)
async def get_transformer(
    transformer_id: int,
    db: Session = Depends(get_db),
):
    """
    Get a transformer by ID.

    - **transformer_id**: The ID of the transformer to retrieve
    """
    transformer = db.query(Transformer).filter(Transformer.id == transformer_id).first()
    if not transformer:
        raise HTTPException(status_code=404, detail="Transformer not found")
    return transformer


@router.post("", response_model=TransformerResponse, status_code=201)
async def create_transformer(
    transformer: TransformerCreate,
    db: Session = Depends(get_db),
):
    """
    Create a new transformer.

    - **transformer**: Transformer data to create
    """
    # Check for duplicate serial number
    if transformer.serial_number:
        existing = (
            db.query(Transformer)
            .filter(Transformer.serial_number == transformer.serial_number)
            .first()
        )
        if existing:
            raise HTTPException(
                status_code=400,
                detail="Transformer with this serial number already exists",
            )

    db_transformer = Transformer(**transformer.model_dump())
    db.add(db_transformer)
    db.commit()
    db.refresh(db_transformer)
    return db_transformer


@router.put("/{transformer_id}", response_model=TransformerResponse)
async def update_transformer(
    transformer_id: int,
    transformer: TransformerUpdate,
    db: Session = Depends(get_db),
):
    """
    Update an existing transformer.

    - **transformer_id**: The ID of the transformer to update
    - **transformer**: Updated transformer data
    """
    db_transformer = (
        db.query(Transformer).filter(Transformer.id == transformer_id).first()
    )
    if not db_transformer:
        raise HTTPException(status_code=404, detail="Transformer not found")

    # Check for duplicate serial number
    if (
        transformer.serial_number
        and transformer.serial_number != db_transformer.serial_number
    ):
        existing = (
            db.query(Transformer)
            .filter(Transformer.serial_number == transformer.serial_number)
            .first()
        )
        if existing:
            raise HTTPException(
                status_code=400,
                detail="Transformer with this serial number already exists",
            )

    # Update only provided fields
    update_data = transformer.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_transformer, field, value)

    db.commit()
    db.refresh(db_transformer)
    return db_transformer


@router.delete("/{transformer_id}", status_code=204)
async def delete_transformer(
    transformer_id: int,
    db: Session = Depends(get_db),
):
    """
    Delete a transformer.

    - **transformer_id**: The ID of the transformer to delete
    """
    transformer = db.query(Transformer).filter(Transformer.id == transformer_id).first()
    if not transformer:
        raise HTTPException(status_code=404, detail="Transformer not found")

    db.delete(transformer)
    db.commit()
    return None


@router.get("/{transformer_id}/summary", response_model=TransformerSummary)
async def get_transformer_summary(
    transformer_id: int,
    db: Session = Depends(get_db),
):
    """
    Get transformer summary with latest health information.

    - **transformer_id**: The ID of the transformer
    """
    transformer = db.query(Transformer).filter(Transformer.id == transformer_id).first()
    if not transformer:
        raise HTTPException(status_code=404, detail="Transformer not found")

    # Get latest health index
    latest_health = (
        db.query(HealthIndexRecord)
        .filter(HealthIndexRecord.transformer_id == transformer_id)
        .order_by(HealthIndexRecord.calculation_date.desc())
        .first()
    )

    # Get latest DGA record
    latest_dga = (
        db.query(DGARecord)
        .filter(DGARecord.transformer_id == transformer_id)
        .order_by(DGARecord.sample_date.desc())
        .first()
    )

    # Get active alert count
    alert_count = (
        db.query(func.count(Alert.id))
        .filter(Alert.transformer_id == transformer_id, Alert.acknowledged == False)
        .scalar()
    )

    return TransformerSummary(
        id=transformer.id,
        name=transformer.name,
        serial_number=transformer.serial_number,
        manufacturer=transformer.manufacturer,
        location=transformer.location,
        substation=transformer.substation,
        rated_mva=transformer.rated_mva,
        rated_voltage_kv=transformer.rated_voltage_kv,
        health_index=latest_health.health_index if latest_health else None,
        health_category=latest_health.category if latest_health else None,
        latest_dga_date=latest_dga.sample_date if latest_dga else None,
        fault_type=latest_dga.fault_type if latest_dga else None,
        alert_count=alert_count or 0,
    )
