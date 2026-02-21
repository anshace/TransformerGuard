"""
Prediction Endpoints
API routes for RUL estimation, failure probability, and forecasting
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.database.connection import DatabaseConnection
from src.database.models import DGARecord, HealthIndexRecord, Transformer
from src.health_index import TrendAnalyzer
from src.prediction import (
    FailureProbability,
    FailureProbabilityResult,
    ForecastResult,
    GasTrendForecaster,
    RULEstimator,
    RULResult,
)

from ..schemas.dga import DGAInput


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


def _get_health_history(transformer_id: int, db: Session) -> List[HealthIndexRecord]:
    """Get health index history for a transformer."""
    return (
        db.query(HealthIndexRecord)
        .filter(HealthIndexRecord.transformer_id == transformer_id)
        .order_by(HealthIndexRecord.calculation_date.asc())
        .all()
    )


def _calculate_age_years(manufacture_date: Optional[date]) -> float:
    """Calculate transformer age in years."""
    if not manufacture_date:
        return 0
    today = date.today()
    age_days = (today - manufacture_date).days
    return max(0, age_days / 365.25)


@router.get("/{transformer_id}/rul")
async def get_rul_estimate(
    transformer_id: int,
    db: Session = Depends(get_db),
):
    """
    Get Remaining Useful Life (RUL) estimate for a transformer.

    - **transformer_id**: ID of the transformer
    """
    # Verify transformer exists
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

    if not latest_health or latest_health.health_index is None:
        raise HTTPException(
            status_code=400, detail="No health index data available for RUL estimation"
        )

    # Get health index history for degradation calculation
    health_history = _get_health_history(transformer_id, db)

    # Calculate age
    age_years = _calculate_age_years(transformer.manufacture_date)

    # Estimate RUL
    rul_estimator = RULEstimator()

    history_values = None
    history_dates = None
    if len(health_history) >= 2:
        history_values = [h.health_index for h in health_history if h.health_index]
        history_dates = [h.calculation_date for h in health_history if h.health_index]

    result = rul_estimator.estimate_rul(
        current_health_index=latest_health.health_index,
        age_years=age_years,
        health_index_history=history_values,
        health_index_dates=history_dates,
    )

    return {
        "transformer_id": transformer_id,
        "rul_years": round(result.rul_years, 2),
        "rul_days": round(result.rul_days, 0),
        "confidence": result.confidence,
        "method": result.method,
        "end_of_life_date": result.end_of_life_date.isoformat()
        if result.end_of_life_date
        else None,
        "assumptions": result.assumptions,
        "current_health_index": latest_health.health_index,
        "age_years": round(age_years, 2),
    }


@router.get("/{transformer_id}/failure-probability")
async def get_failure_probability(
    transformer_id: int,
    time_horizon_years: float = Query(
        1.0, ge=0.1, le=10, description="Time horizon in years"
    ),
    db: Session = Depends(get_db),
):
    """
    Get failure probability estimate for a transformer.

    - **transformer_id**: ID of the transformer
    - **time_horizon_years**: Time horizon for probability calculation
    """
    # Verify transformer exists
    transformer = db.query(Transformer).filter(Transformer.id == transformer_id).first()
    if not transformer:
        raise HTTPException(status_code=404, detail="Transformer not found")

    # Get latest DGA record
    latest_dga = _get_latest_dga(transformer_id, db)

    # Get latest health index
    latest_health = (
        db.query(HealthIndexRecord)
        .filter(HealthIndexRecord.transformer_id == transformer_id)
        .order_by(HealthIndexRecord.calculation_date.desc())
        .first()
    )

    if not latest_health or latest_health.health_index is None:
        raise HTTPException(
            status_code=400,
            detail="No health index data available for failure probability estimation",
        )

    # Get fault types
    fault_types = []
    if latest_dga and latest_dga.fault_type:
        fault_types = [latest_dga.fault_type]

    # Calculate age
    age_years = _calculate_age_years(transformer.manufacture_date)

    # Calculate failure probability
    fp_calculator = FailureProbability()

    result = fp_calculator.calculate_failure_probability(
        health_index=latest_health.health_index,
        age_years=age_years,
        time_horizon_years=time_horizon_years,
        fault_types_present=fault_types,
        dga_severity=latest_dga.fault_confidence if latest_dga else None,
    )

    return {
        "transformer_id": transformer_id,
        "probability_1_year": round(result.probability_1_year, 4),
        "probability_5_years": round(result.probability_5_years, 4),
        "probability_10_years": round(result.probability_10_years, 4),
        "risk_level": result.risk_level,
        "risk_factors": result.risk_factors,
        "confidence": result.confidence,
        "current_health_index": latest_health.health_index,
        "time_horizon_years": time_horizon_years,
    }


@router.get("/{transformer_id}/forecast")
async def get_gas_trend_forecast(
    transformer_id: int,
    gas_type: str = Query(
        "tdcg", description="Gas to forecast (tdcg, h2, ch4, c2h2, c2h4, co, co2)"
    ),
    forecast_months: int = Query(
        6, ge=1, le=24, description="Number of months to forecast"
    ),
    db: Session = Depends(get_db),
):
    """
    Get gas trend forecast for a transformer.

    - **transformer_id**: ID of the transformer
    - **gas_type**: Type of gas to forecast
    - **forecast_months**: Number of months to forecast
    """
    # Verify transformer exists
    transformer = db.query(Transformer).filter(Transformer.id == transformer_id).first()
    if not transformer:
        raise HTTPException(status_code=404, detail="Transformer not found")

    # Get DGA history
    dga_records = (
        db.query(DGARecord)
        .filter(DGARecord.transformer_id == transformer_id)
        .order_by(DGARecord.sample_date.asc())
        .all()
    )

    if len(dga_records) < 3:
        raise HTTPException(
            status_code=400, detail="Need at least 3 DGA records for trend forecasting"
        )

    # Prepare gas data
    gas_mapping = {
        "tdcg": "tdcg",
        "h2": "h2",
        "ch4": "ch4",
        "c2h2": "c2h2",
        "c2h4": "c2h4",
        "c2h6": "c2h6",
        "co": "co",
        "co2": "co2",
    }

    if gas_type not in gas_mapping:
        raise HTTPException(status_code=400, detail=f"Invalid gas type: {gas_type}")

    gas_field = gas_mapping[gas_type]

    # Extract gas values and dates
    gas_data = []
    for record in dga_records:
        gas_value = getattr(record, gas_field, None)
        if gas_value is not None:
            gas_data.append(
                {
                    "date": record.sample_date,
                    "value": gas_value,
                }
            )

    if len(gas_data) < 3:
        raise HTTPException(
            status_code=400, detail=f"Not enough {gas_type} data for forecasting"
        )

    # Run forecast
    forecaster = GasTrendForecaster()
    result = forecaster.forecast(
        gas_data=gas_data,
        forecast_periods=forecast_months,
    )

    return {
        "transformer_id": transformer_id,
        "gas_type": gas_type,
        "trend": result.trend,
        "trend_slope": result.trend_slope,
        "forecast_values": result.forecast_values,
        "forecast_dates": [d.isoformat() for d in result.forecast_dates],
        "confidence": result.confidence,
        "data_points": len(gas_data),
        "forecast_periods": forecast_months,
    }


class PredictionAnalysisRequest:
    """Request model for full prediction analysis."""

    transformer_id: int
    time_horizon_years: float = 1.0
    forecast_months: int = 6


@router.post("/analyze")
async def full_prediction_analysis(
    transformer_id: int,
    time_horizon_years: float = Query(1.0, ge=0.1, le=10),
    forecast_months: int = Query(6, ge=1, le=24),
    db: Session = Depends(get_db),
):
    """
    Run full prediction analysis for a transformer.

    - **transformer_id**: ID of the transformer
    - **time_horizon_years**: Time horizon for failure probability
    - **forecast_months**: Number of months for gas forecasting
    """
    # Verify transformer exists
    transformer = db.query(Transformer).filter(Transformer.id == transformer_id).first()
    if not transformer:
        raise HTTPException(status_code=404, detail="Transformer not found")

    results = {}

    # 1. RUL Estimation
    try:
        latest_health = (
            db.query(HealthIndexRecord)
            .filter(HealthIndexRecord.transformer_id == transformer_id)
            .order_by(HealthIndexRecord.calculation_date.desc())
            .first()
        )

        if latest_health and latest_health.health_index:
            age_years = _calculate_age_years(transformer.manufacture_date)
            rul_estimator = RULEstimator()
            rul_result = rul_estimator.estimate_rul(
                current_health_index=latest_health.health_index,
                age_years=age_years,
            )
            results["rul"] = {
                "rul_years": round(rul_result.rul_years, 2),
                "rul_days": round(rul_result.rul_days, 0),
                "confidence": rul_result.confidence,
                "end_of_life_date": rul_result.end_of_life_date.isoformat()
                if rul_result.end_of_life_date
                else None,
            }
    except Exception as e:
        results["rul"] = {"error": str(e)}

    # 2. Failure Probability
    try:
        latest_dga = _get_latest_dga(transformer_id, db)
        if latest_health and latest_health.health_index:
            age_years = _calculate_age_years(transformer.manufacture_date)
            fp_calculator = FailureProbability()
            fp_result = fp_calculator.calculate_failure_probability(
                health_index=latest_health.health_index,
                age_years=age_years,
                time_horizon_years=time_horizon_years,
            )
            results["failure_probability"] = {
                "probability_1_year": round(fp_result.probability_1_year, 4),
                "probability_5_years": round(fp_result.probability_5_years, 4),
                "risk_level": fp_result.risk_level,
            }
    except Exception as e:
        results["failure_probability"] = {"error": str(e)}

    # 3. Gas Trend Forecast
    try:
        dga_records = (
            db.query(DGARecord)
            .filter(DGARecord.transformer_id == transformer_id)
            .order_by(DGARecord.sample_date.asc())
            .all()
        )

        if len(dga_records) >= 3:
            gas_data = [
                {"date": r.sample_date, "value": r.tdcg} for r in dga_records if r.tdcg
            ]
            if len(gas_data) >= 3:
                forecaster = GasTrendForecaster()
                forecast_result = forecaster.forecast(
                    gas_data=gas_data,
                    forecast_periods=forecast_months,
                )
                results["gas_forecast"] = {
                    "trend": forecast_result.trend,
                    "trend_slope": forecast_result.trend_slope,
                    "confidence": forecast_result.confidence,
                }
    except Exception as e:
        results["gas_forecast"] = {"error": str(e)}

    return {
        "transformer_id": transformer_id,
        "transformer_name": transformer.name,
        "analysis_timestamp": datetime.utcnow().isoformat(),
        "results": results,
    }
