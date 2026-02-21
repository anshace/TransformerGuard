"""
SQLAlchemy ORM Models for TransformerGuard
Database models for transformers, DGA records, health indices, and alerts
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    relationship,
    sessionmaker,
)


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class Transformer(Base):
    """
    Transformer model representing power transformer assets.
    """

    __tablename__ = "transformers"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    serial_number = Column(String(50), unique=True, index=True)
    manufacturer = Column(String(100))
    manufacture_date = Column(Date)
    installation_date = Column(Date)
    rated_mva = Column(Float)
    rated_voltage_kv = Column(Float)
    cooling_type = Column(String(20))  # ONAN, ONAF, OFAF, ODAF
    location = Column(String(200))
    substation = Column(String(100))

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    dga_records: Mapped["DGARecord"] = relationship(
        "DGARecord",
        back_populates="transformer",
        cascade="all, delete-orphan",
        order_by="DGARecord.sample_date.desc()",
    )
    health_records: Mapped["HealthIndexRecord"] = relationship(
        "HealthIndexRecord",
        back_populates="transformer",
        cascade="all, delete-orphan",
        order_by="HealthIndexRecord.calculation_date.desc()",
    )
    alerts: Mapped["Alert"] = relationship(
        "Alert",
        back_populates="transformer",
        cascade="all, delete-orphan",
        order_by="Alert.created_at.desc()",
    )
    load_records: Mapped["LoadRecord"] = relationship(
        "LoadRecord",
        back_populates="transformer",
        cascade="all, delete-orphan",
        order_by="LoadRecord.timestamp.desc()",
    )
    thermal_records: Mapped["ThermalRecord"] = relationship(
        "ThermalRecord",
        back_populates="transformer",
        cascade="all, delete-orphan",
        order_by="ThermalRecord.timestamp.desc()",
    )

    def __repr__(self) -> str:
        return f"<Transformer(id={self.id}, name='{self.name}', serial='{self.serial_number}')>"


class DGARecord(Base):
    """
    Dissolved Gas Analysis (DGA) record for transformer oil testing.
    Tracks gas concentrations and analysis results.
    """

    __tablename__ = "dga_records"

    id = Column(Integer, primary_key=True)
    transformer_id = Column(
        Integer,
        ForeignKey("transformers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sample_date = Column(DateTime, nullable=False, index=True)

    # Gas concentrations (ppm)
    h2 = Column(Float)  # Hydrogen
    ch4 = Column(Float)  # Methane
    c2h2 = Column(Float)  # Acetylene
    c2h4 = Column(Float)  # Ethylene
    c2h6 = Column(Float)  # Ethane
    co = Column(Float)  # Carbon Monoxide
    co2 = Column(Float)  # Carbon Dioxide
    o2 = Column(Float)  # Oxygen
    n2 = Column(Float)  # Nitrogen

    # Analysis results
    tdcg = Column(Float)  # Total Dissolved Combustible Gas
    fault_type = Column(String(50), index=True)
    fault_confidence = Column(Float)
    diagnosis_method = Column(String(50))

    # Metadata
    lab_name = Column(String(100))
    sample_type = Column(String(20))  # Routine, Emergency, Follow-up
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    transformer: Mapped["Transformer"] = relationship(
        "Transformer", back_populates="dga_records"
    )

    def __repr__(self) -> str:
        return f"<DGARecord(id={self.id}, transformer_id={self.transformer_id}, sample_date='{self.sample_date}')>"


class HealthIndexRecord(Base):
    """
    Health Index calculation record for transformer condition assessment.
    """

    __tablename__ = "health_index_records"

    id = Column(Integer, primary_key=True)
    transformer_id = Column(
        Integer,
        ForeignKey("transformers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    calculation_date = Column(DateTime, nullable=False, index=True)

    # Component scores (0-100)
    health_index = Column(Float)  # Overall 0-100
    dga_score = Column(Float)
    oil_quality_score = Column(Float)
    electrical_score = Column(Float)
    age_score = Column(Float)
    loading_score = Column(Float)

    # Category
    category = Column(String(20), index=True)  # EXCELLENT, GOOD, FAIR, POOR, CRITICAL

    # Trend analysis
    trend_direction = Column(String(20))  # IMPROVING, STABLE, DECLINING
    monthly_rate = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    transformer: Mapped["Transformer"] = relationship(
        "Transformer", back_populates="health_records"
    )

    def __repr__(self) -> str:
        return f"<HealthIndexRecord(id={self.id}, transformer_id={self.transformer_id}, health_index={self.health_index})>"


class Alert(Base):
    """
    Alert model for transformer condition monitoring notifications.
    """

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True)
    transformer_id = Column(
        Integer,
        ForeignKey("transformers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    priority = Column(String(20), index=True)  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    category = Column(String(20), index=True)  # DGA, THERMAL, HEALTH, LOADING, ANOMALY
    title = Column(String(200))
    message = Column(Text)

    # Acknowledgment tracking
    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String(100))
    acknowledged_at = Column(DateTime)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    resolved_at = Column(DateTime)

    # Relationship
    transformer: Mapped["Transformer"] = relationship(
        "Transformer", back_populates="alerts"
    )

    def __repr__(self) -> str:
        return f"<Alert(id={self.id}, transformer_id={self.transformer_id}, priority='{self.priority}', title='{self.title}')>"


class LoadRecord(Base):
    """
    Load recording for transformer loading conditions.
    """

    __tablename__ = "load_records"

    id = Column(Integer, primary_key=True)
    transformer_id = Column(
        Integer,
        ForeignKey("transformers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    timestamp = Column(DateTime, nullable=False, index=True)

    load_mva = Column(Float)
    load_percent = Column(Float)  # Percentage of rated
    current_a = Column(Float)

    # Relationship
    transformer: Mapped["Transformer"] = relationship(
        "Transformer", back_populates="load_records"
    )

    def __repr__(self) -> str:
        return f"<LoadRecord(id={self.id}, transformer_id={self.transformer_id}, timestamp='{self.timestamp}', load_percent={self.load_percent})>"


class ThermalRecord(Base):
    """
    Thermal recording for transformer temperature monitoring.
    """

    __tablename__ = "thermal_records"

    id = Column(Integer, primary_key=True)
    transformer_id = Column(
        Integer,
        ForeignKey("transformers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    timestamp = Column(DateTime, nullable=False, index=True)

    # Temperature readings (Celsius)
    ambient_temp = Column(Float)
    top_oil_temp = Column(Float)
    hotspot_temp = Column(Float)
    winding_temp = Column(Float)

    # Aging metrics
    aging_acceleration = Column(Float)
    loss_of_life_hours = Column(Float)

    # Relationship
    transformer: Mapped["Transformer"] = relationship(
        "Transformer", back_populates="thermal_records"
    )

    def __repr__(self) -> str:
        return f"<ThermalRecord(id={self.id}, transformer_id={self.transformer_id}, timestamp='{self.timestamp}', hotspot={self.hotspot_temp})>"


# Define indexes for frequently queried columns
__table_args__ = (
    Index("ix_dga_transformer_date", "transformer_id", "sample_date"),
    Index("ix_health_transformer_date", "transformer_id", "calculation_date"),
    Index("ix_load_transformer_timestamp", "transformer_id", "timestamp"),
    Index("ix_thermal_transformer_timestamp", "transformer_id", "timestamp"),
    Index("ix_alerts_transformer_created", "transformer_id", "created_at"),
)
