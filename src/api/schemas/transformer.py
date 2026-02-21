"""
Transformer Pydantic Models
Request and response schemas for transformer endpoints
"""

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class TransformerBase(BaseModel):
    """Base transformer schema with common fields."""

    name: str
    serial_number: Optional[str] = None
    manufacturer: Optional[str] = None
    manufacture_date: Optional[date] = None
    installation_date: Optional[date] = None
    rated_mva: Optional[float] = None
    rated_voltage_kv: Optional[float] = None
    cooling_type: Optional[str] = None
    location: Optional[str] = None
    substation: Optional[str] = None


class TransformerCreate(TransformerBase):
    """Schema for creating a new transformer."""

    pass


class TransformerUpdate(BaseModel):
    """Schema for updating an existing transformer."""

    name: Optional[str] = None
    serial_number: Optional[str] = None
    manufacturer: Optional[str] = None
    manufacture_date: Optional[date] = None
    installation_date: Optional[date] = None
    rated_mva: Optional[float] = None
    rated_voltage_kv: Optional[float] = None
    cooling_type: Optional[str] = None
    location: Optional[str] = None
    substation: Optional[str] = None


class TransformerResponse(TransformerBase):
    """Schema for transformer response."""

    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TransformerSummary(BaseModel):
    """Schema for transformer summary with health information."""

    id: int
    name: str
    serial_number: Optional[str] = None
    manufacturer: Optional[str] = None
    location: Optional[str] = None
    substation: Optional[str] = None
    rated_mva: Optional[float] = None
    rated_voltage_kv: Optional[float] = None
    health_index: Optional[float] = None
    health_category: Optional[str] = None
    latest_dga_date: Optional[datetime] = None
    fault_type: Optional[str] = None
    alert_count: int = 0

    model_config = ConfigDict(from_attributes=True)
