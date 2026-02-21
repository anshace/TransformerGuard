"""
Health Index Pydantic Models
Request and response schemas for health index endpoints
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class HealthCalculationInput(BaseModel):
    """Schema for health index calculation input data."""

    # DGA data
    h2: float
    ch4: float
    c2h2: float
    c2h4: float
    c2h6: float
    co: float
    co2: float
    # Oil quality
    dielectric_strength: Optional[float] = None
    moisture_content: Optional[float] = None
    acidity: Optional[float] = None
    # Electrical
    power_factor: Optional[float] = None
    insulation_resistance: Optional[float] = None
    # Transformer info
    age_years: Optional[float] = None
    average_load_percent: Optional[float] = None


class HealthIndexResponse(BaseModel):
    """Schema for health index response."""

    transformer_id: int
    health_index: float
    category: str
    dga_score: float
    oil_quality_score: float
    electrical_score: float
    age_score: float
    loading_score: float
    trend_direction: Optional[str] = None
    recommendations: List[str]
    calculated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class HealthIndexRecordResponse(BaseModel):
    """Schema for health index record from database."""

    id: int
    transformer_id: int
    calculation_date: datetime
    health_index: Optional[float] = None
    dga_score: Optional[float] = None
    oil_quality_score: Optional[float] = None
    electrical_score: Optional[float] = None
    age_score: Optional[float] = None
    loading_score: Optional[float] = None
    category: Optional[str] = None
    trend_direction: Optional[str] = None
    monthly_rate: Optional[float] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class HealthHistoryResponse(BaseModel):
    """Schema for health index history response."""

    transformer_id: int
    records: List[HealthIndexRecordResponse]
    total_count: int


class TrendAnalysisResponse(BaseModel):
    """Schema for health trend analysis response."""

    transformer_id: int
    trend_direction: str
    monthly_rate: float
    predicted_index_6_months: float
    predicted_index_12_months: float
    data_points: int
    confidence: float
    analysis_date: datetime


class FleetHealthOverview(BaseModel):
    """Schema for fleet-wide health overview."""

    total_transformers: int
    excellent_count: int
    good_count: int
    fair_count: int
    poor_count: int
    critical_count: int
    average_health_index: float
    transformers: List[Dict[str, Any]]
