"""
Alert Endpoints
API routes for alert management
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.database.connection import DatabaseConnection
from src.database.models import Alert, Transformer


def get_db():
    """Dependency for database session."""
    db_connection = DatabaseConnection.get_instance()
    db = db_connection.get_session()
    try:
        yield db
    finally:
        db.close()


router = APIRouter()


class AlertResponse(BaseModel):
    """Schema for alert response."""

    id: int
    transformer_id: int
    priority: str
    category: str
    title: str
    message: Optional[str]
    acknowledged: bool
    acknowledged_by: Optional[str]
    acknowledged_at: Optional[datetime]
    created_at: datetime
    resolved_at: Optional[datetime]

    class Config:
        from_attributes = True


class AlertListResponse(BaseModel):
    """Schema for alert list response."""

    alerts: List[AlertResponse]
    total_count: int
    page: int
    page_size: int


class AlertSummary(BaseModel):
    """Schema for alert summary statistics."""

    total_alerts: int
    active_alerts: int
    acknowledged_alerts: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    info_count: int
    by_category: dict


class AcknowledgeRequest(BaseModel):
    """Schema for acknowledging an alert."""

    acknowledged_by: str


@router.get("", response_model=AlertListResponse)
async def list_alerts(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=1000),
    priority: Optional[str] = Query(
        None, description="Filter by priority (CRITICAL, HIGH, MEDIUM, LOW, INFO)"
    ),
    category: Optional[str] = Query(
        None, description="Filter by category (DGA, THERMAL, HEALTH, LOADING, ANOMALY)"
    ),
    acknowledged: Optional[bool] = Query(
        None, description="Filter by acknowledgment status"
    ),
    db: Session = Depends(get_db),
):
    """
    List all alerts with optional filters.

    - **skip**: Number of records to skip
    - **limit**: Maximum number of records to return
    - **priority**: Filter by priority level
    - **category**: Filter by category
    - **acknowledged**: Filter by acknowledgment status
    """
    query = db.query(Alert)

    if priority:
        query = query.filter(Alert.priority == priority.upper())

    if category:
        query = query.filter(Alert.category == category.upper())

    if acknowledged is not None:
        query = query.filter(Alert.acknowledged == acknowledged)

    total_count = query.count()

    alerts = (
        query.order_by(
            Alert.acknowledged.asc(), Alert.priority.desc(), Alert.created_at.desc()
        )
        .offset(skip)
        .limit(limit)
        .all()
    )

    return AlertListResponse(
        alerts=alerts,
        total_count=total_count,
        page=skip // limit + 1,
        page_size=limit,
    )


@router.get("/active", response_model=AlertListResponse)
async def get_active_alerts(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """
    Get all active (unacknowledged) alerts.

    - **skip**: Number of records to skip
    - **limit**: Maximum number of records to return
    """
    query = db.query(Alert).filter(Alert.acknowledged == False)

    total_count = query.count()

    alerts = (
        query.order_by(Alert.priority.desc(), Alert.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    return AlertListResponse(
        alerts=alerts,
        total_count=total_count,
        page=skip // limit + 1,
        page_size=limit,
    )


@router.get("/{transformer_id}", response_model=List[AlertResponse])
async def get_transformer_alerts(
    transformer_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=1000),
    acknowledged: Optional[bool] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Get alerts for a specific transformer.

    - **transformer_id**: ID of the transformer
    - **skip**: Number of records to skip
    - **limit**: Maximum number of records to return
    - **acknowledged**: Filter by acknowledgment status
    """
    # Verify transformer exists
    transformer = db.query(Transformer).filter(Transformer.id == transformer_id).first()
    if not transformer:
        raise HTTPException(status_code=404, detail="Transformer not found")

    query = db.query(Alert).filter(Alert.transformer_id == transformer_id)

    if acknowledged is not None:
        query = query.filter(Alert.acknowledged == acknowledged)

    alerts = (
        query.order_by(
            Alert.acknowledged.asc(), Alert.priority.desc(), Alert.created_at.desc()
        )
        .offset(skip)
        .limit(limit)
        .all()
    )

    return alerts


@router.put("/{alert_id}/acknowledge", response_model=AlertResponse)
async def acknowledge_alert(
    alert_id: int,
    acknowledge_request: AcknowledgeRequest,
    db: Session = Depends(get_db),
):
    """
    Acknowledge an alert.

    - **alert_id**: ID of the alert to acknowledge
    - **acknowledged_by**: Name of the person acknowledging the alert
    """
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    if alert.acknowledged:
        raise HTTPException(status_code=400, detail="Alert is already acknowledged")

    alert.acknowledged = True
    alert.acknowledged_by = acknowledge_request.acknowledged_by
    alert.acknowledged_at = datetime.utcnow()

    db.commit()
    db.refresh(alert)

    return alert


@router.get("/summary", response_model=AlertSummary)
async def get_alert_summary(
    db: Session = Depends(get_db),
):
    """
    Get alert summary statistics.
    """
    # Total alerts
    total_alerts = db.query(func.count(Alert.id)).scalar()

    # Active alerts
    active_alerts = (
        db.query(func.count(Alert.id)).filter(Alert.acknowledged == False).scalar()
    )

    # Acknowledged alerts
    acknowledged_alerts = (
        db.query(func.count(Alert.id)).filter(Alert.acknowledged == True).scalar()
    )

    # By priority
    priority_counts = {}
    for priority in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
        count = (
            db.query(func.count(Alert.id))
            .filter(Alert.priority == priority, Alert.acknowledged == False)
            .scalar()
        )
        priority_counts[priority.lower()] = count

    # By category
    category_counts = {}
    for category in ["DGA", "THERMAL", "HEALTH", "LOADING", "ANOMALY"]:
        count = (
            db.query(func.count(Alert.id))
            .filter(Alert.category == category, Alert.acknowledged == False)
            .scalar()
        )
        category_counts[category.lower()] = count

    return AlertSummary(
        total_alerts=total_alerts or 0,
        active_alerts=active_alerts or 0,
        acknowledged_alerts=acknowledged_alerts or 0,
        critical_count=priority_counts.get("critical", 0),
        high_count=priority_counts.get("high", 0),
        medium_count=priority_counts.get("medium", 0),
        low_count=priority_counts.get("low", 0),
        info_count=priority_counts.get("info", 0),
        by_category=category_counts,
    )


@router.delete("/{alert_id}", status_code=204)
async def delete_alert(
    alert_id: int,
    db: Session = Depends(get_db),
):
    """
    Delete an alert.

    - **alert_id**: ID of the alert to delete
    """
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    db.delete(alert)
    db.commit()

    return None
