"""
Database Module
SQLAlchemy ORM models and connection management for TransformerGuard
"""

from .connection import DatabaseConnection, get_session, init_db, session_scope
from .models import (
    Alert,
    Base,
    DGARecord,
    HealthIndexRecord,
    LoadRecord,
    ThermalRecord,
    Transformer,
)

__all__ = [
    # Base class
    "Base",
    # Models
    "Transformer",
    "DGARecord",
    "HealthIndexRecord",
    "Alert",
    "LoadRecord",
    "ThermalRecord",
    # Connection management
    "DatabaseConnection",
    "get_session",
    "init_db",
    "session_scope",
]
