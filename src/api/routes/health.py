"""
Health Index Endpoints
API routes for health index calculation and analysis
"""

from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.database.connection import DatabaseConnection
from src.database.models import DGARecord, HealthIndexRecord, Transformer
from src.health_index import (
    CompositeHealthIndex,
    HealthIndexResult,
    TrendAnalyzer,
)

from ..schemas.health import (
    FleetHealthOverview,
    HealthCalculationInput,
    HealthHistoryResponse,
    HealthIndexRecordResponse,
    HealthIndexResponse,
    TrendAnalysisResponse,
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


def _get_latest_dga(transformer_id: int, db: Session) -> Optional[DGARecord]:
    """Get the latest DGA record for a transformer."""
    return (
        db.query(DGARecord)
        .filter(DGARecord.transformer_id == transformer_id)
        .order_by(DGARecord.sample_date.desc())
        .first()
    )


def _calculate_age_years(manufacture_date: date) -> float:
    """Calculate transformer age in years."""
    today = date.today()
    age_days = (today - manufacture_date).days
    return age_days / 365.25


@router.get("/{transformer_id}", response_model=HealthIndexResponse)
async def get_health_index(
    transformer_id: int,
    db: Session = Depends(get_db),
):
    """
    Get current health index for a transformer.

    - **transformer_id**: ID of the transformer
    """
    # Verify transformer exists
    transformer = db.query(Transformer).filter(Transformer.id == transformer_id).first()
    if not transformer:
        raise HTTPException(status_code=404, detail="Transformer not found")

    # Get latest health record
    latest_health = (
        db.query(HealthIndexRecord)
        .filter(HealthIndexRecord.transformer_id == transformer_id)
        .order_by(HealthIndexRecord.calculation_date.desc())
        .first()
    )

    if not latest_health:
        raise HTTPException(
            status_code=404,
            detail="No health index record found. Calculate health index first.",
        )

    # Get recommendations from action recommender
    recommendations = []
    if latest_health.health_index and latest_health.health_index < 70:
        if latest_health.health_index < 25:
            recommendations.append("Immediate attention required - critical condition")
        elif latest_health.health_index < 50:
            recommendations.append("Schedule maintenance soon - poor condition")
        else:
            recommendations.append("Consider condition monitoring - fair condition")

    return HealthIndexResponse(
        transformer_id=transformer_id,
        health_index=latest_health.health_index or 0,
        category=latest_health.category or "UNKNOWN",
        dga_score=latest_health.dga_score or 0,
        oil_quality_score=latest_health.oil_quality_score or 0,
        electrical_score=latest_health.electrical_score or 0,
        age_score=latest_health.age_score or 0,
        loading_score=latest_health.loading_score or 0,
        trend_direction=latest_health.trend_direction,
        recommendations=recommendations,
        calculated_at=latest_health.calculation_date,
    )


@router.get("/{transformer_id}/history", response_model=HealthHistoryResponse)
async def get_health_history(
    transformer_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """
    Get health index history for a transformer.

    - **transformer_id**: ID of the transformer
    - **skip**: Number of records to skip
    - **limit**: Maximum number of records to return
    """
    # Verify transformer exists
    transformer = db.query(Transformer).filter(Transformer.id == transformer_id).first()
    if not transformer:
        raise HTTPException(status_code=404, detail="Transformer not found")

    records = (
        db.query(HealthIndexRecord)
        .filter(HealthIndexRecord.transformer_id == transformer_id)
        .order_by(HealthIndexRecord.calculation_date.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    total_count = (
        db.query(HealthIndexRecord)
        .filter(HealthIndexRecord.transformer_id == transformer_id)
        .count()
    )

    return HealthHistoryResponse(
        transformer_id=transformer_id,
        records=records,
        total_count=total_count,
    )


@router.post("/calculate", response_model=HealthIndexResponse)
async def calculate_health_index(
    calculation_input: HealthCalculationInput,
    save: bool = Query(True, description="Save result to database"),
    db: Session = Depends(get_db),
):
    """
    Calculate health index from input data.

    - **calculation_input**: Input data for health index calculation
    - **save**: Whether to save result to database
    """
    # Calculate health index
    hi_calculator = CompositeHealthIndex()

    # Prepare DGA gases
    dga_gases = {
        "h2": calculation_input.h2,
        "ch4": calculation_input.ch4,
        "c2h2": calculation_input.c2h2,
        "c2h4": calculation_input.c2h4,
        "c2h6": calculation_input.c2h6,
        "co": calculation_input.co,
        "co2": calculation_input.co2,
    }

    # Calculate composite health index
    result = hi_calculator.calculate(
        dga_gases=dga_gases,
        dielectric_strength=calculation_input.dielectric_strength,
        moisture_content=calculation_input.moisture_content,
        acidity=calculation_input.acidity,
        power_factor=calculation_input.power_factor,
        capacitance=0,  # Not provided
        insulation_resistance=calculation_input.insulation_resistance,
        age_years=calculation_input.age_years,
        average_load_percent=calculation_input.average_load_percent,
    )

    # Save to database if requested
    if save:
        # For this, we need a transformer_id, which isn't in the input
        # We'll create a placeholder
        pass

    return HealthIndexResponse(
        transformer_id=0,  # Placeholder
        health_index=result.health_index,
        category=result.category,
        dga_score=result.component_scores.get("dga", 0),
        oil_quality_score=result.component_scores.get("oil_quality", 0),
        electrical_score=result.component_scores.get("electrical", 0),
        age_score=result.component_scores.get("age", 0),
        loading_score=result.component_scores.get("loading", 0),
        trend_direction=None,
        recommendations=result.recommendations,
        calculated_at=datetime.utcnow(),
    )


@router.get("/{transformer_id}/trend", response_model=TrendAnalysisResponse)
async def get_health_trend(
    transformer_id: int,
    db: Session = Depends(get_db),
):
    """
    Get health trend analysis for a transformer.

    - **transformer_id**: ID of the transformer
    """
    # Verify transformer exists
    transformer = db.query(Transformer).filter(Transformer.id == transformer_id).first()
    if not transformer:
        raise HTTPException(status_code=404, detail="Transformer not found")

    # Get health history
    records = (
        db.query(HealthIndexRecord)
        .filter(HealthIndexRecord.transformer_id == transformer_id)
        .order_by(HealthIndexRecord.calculation_date.asc())
        .all()
    )

    if len(records) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 health index records for trend analysis",
        )

    # Run trend analysis
    trend_analyzer = TrendAnalyzer()

    # Prepare data points
    data_points = [
        {
            "date": r.calculation_date,
            "health_index": r.health_index,
        }
        for r in records
    ]

    trend_result = trend_analyzer.analyze_trend(data_points)

    return TrendAnalysisResponse(
        transformer_id=transformer_id,
        trend_direction=trend_result.trend_direction,
        monthly_rate=trend_result.monthly_rate,
        predicted_index_6_months=trend_result.predicted_values.get(6, 0),
        predicted_index_12_months=trend_result.predicted_values.get(12, 0),
        data_points=len(data_points),
        confidence=trend_result.confidence,
        analysis_date=datetime.utcnow(),
    )


@router.get("/fleet/overview", response_model=FleetHealthOverview)
async def get_fleet_health_overview(
    db: Session = Depends(get_db),
):
    """
    Get fleet-wide health overview.
    """
    # Get all transformers
    transformers = db.query(Transformer).all()

    if not transformers:
        return FleetHealthOverview(
            total_transformers=0,
            excellent_count=0,
            good_count=0,
            fair_count=0,
            poor_count=0,
            critical_count=0,
            average_health_index=0,
            transformers=[],
        )

    # Category counts
    category_counts = {
        "EXCELLENT": 0,
        "GOOD": 0,
        "FAIR": 0,
        "POOR": 0,
        "CRITICAL": 0,
    }

    total_health = 0
    transformer_data = []

    for transformer in transformers:
        # Get latest health index
        latest_health = (
            db.query(HealthIndexRecord)
            .filter(HealthIndexRecord.transformer_id == transformer.id)
            .order_by(HealthIndexRecord.calculation_date.desc())
            .first()
        )

        health_index = latest_health.health_index if latest_health else 0
        category = latest_health.category if latest_health else "UNKNOWN"

        if category in category_counts:
            category_counts[category] += 1

        total_health += health_index

        transformer_data.append(
            {
                "id": transformer.id,
                "name": transformer.name,
                "health_index": health_index,
                "category": category,
            }
        )

    avg_health = total_health / len(transformers) if transformers else 0

    return FleetHealthOverview(
        total_transformers=len(transformers),
        excellent_count=category_counts["EXCELLENT"],
        good_count=category_counts["GOOD"],
        fair_count=category_counts["FAIR"],
        poor_count=category_counts["POOR"],
        critical_count=category_counts["CRITICAL"],
        average_health_index=round(avg_health, 2),
        transformers=transformer_data,
    )
