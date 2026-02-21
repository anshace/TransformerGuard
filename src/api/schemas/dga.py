"""
DGA (Dissolved Gas Analysis) Pydantic Models
Request and response schemas for DGA endpoints
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class DGAInput(BaseModel):
    """Schema for DGA sample input data."""

    transformer_id: int
    sample_date: datetime
    h2: float
    ch4: float
    c2h2: float
    c2h4: float
    c2h6: float
    co: float
    co2: float
    o2: Optional[float] = None
    n2: Optional[float] = None
    lab_name: Optional[str] = None
    sample_type: Optional[str] = "Routine"


class DGAAnalysisResult(BaseModel):
    """Schema for DGA analysis result."""

    fault_type: str
    fault_confidence: float
    tdcg: float
    explanation: str
    method_results: Dict[str, Any]


class DGAResponse(BaseModel):
    """Schema for DGA record response."""

    id: int
    transformer_id: int
    sample_date: datetime
    h2: Optional[float] = None
    ch4: Optional[float] = None
    c2h2: Optional[float] = None
    c2h4: Optional[float] = None
    c2h6: Optional[float] = None
    co: Optional[float] = None
    co2: Optional[float] = None
    o2: Optional[float] = None
    n2: Optional[float] = None
    tdcg: Optional[float] = None
    fault_type: Optional[str] = None
    fault_confidence: Optional[float] = None
    diagnosis_method: Optional[str] = None
    lab_name: Optional[str] = None
    sample_type: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DGACreate(DGAInput):
    """Schema for creating a DGA record."""

    notes: Optional[str] = None


class DGAAnalysisRequest(BaseModel):
    """Schema for requesting DGA analysis without saving."""

    transformer_id: int
    sample_date: datetime
    h2: float
    ch4: float
    c2h2: float
    c2h4: float
    c2h6: float
    co: float
    co2: float
    o2: Optional[float] = None
    n2: Optional[float] = None
    method: Optional[str] = "multi"  # multi, duval, rogers, iec, doernenburg, key_gas


class BatchRequestDGAAnalysis(BaseModel):
    """Schema for batch DGA analysis."""

    samples: List[DGAInput]


class BatchDGAAnalysisResponse(BaseModel):
    """Schema for batch DGA analysis response."""

    results: List[DGAResponse]
    total_analyzed: int
    errors: List[str]


class DGAHistoryResponse(BaseModel):
    """Schema for DGA history response."""

    transformer_id: int
    records: List[DGAResponse]
    total_count: int
