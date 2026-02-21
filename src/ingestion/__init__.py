"""
Data Ingestion Module

This module provides data ingestion capabilities for TransformerGuard,
including parsers for CSV and Excel files, weather data retrieval,
data validation, and synthetic data generation.

Components:
- DGACSVParser: Parse DGA data from CSV files
- DGAExcelParser: Parse DGA data from Excel files
- WeatherClient: Retrieve historical weather data from Open-Meteo API
- DGADataValidator: Validate DGA records for data quality
- SyntheticTransformerGenerator: Generate synthetic DGA and sensor data

Author: TransformerGuard Team
"""

from .csv_parser import COLUMN_MAPPINGS, DGARecord, DGACSVParser
from .data_validator import (
    DGADataValidator,
    ValidationIssue,
    ValidationResult,
    GAS_LIMITS,
)
from .excel_parser import DGAExcelParser, TransformerInfo, TRANSFORMER_INFO_MAPPINGS
from .synthetic_generator import (
    GAS_PROFILES,
    SyntheticTransformerGenerator,
    TransformerCondition,
    TransformerConfig,
)
from .weather_client import WeatherClient, WeatherData

__all__ = [
    # CSV Parser
    "DGACSVParser",
    "DGARecord",
    "COLUMN_MAPPINGS",
    # Excel Parser
    "DGAExcelParser",
    "TransformerInfo",
    "TRANSFORMER_INFO_MAPPINGS",
    # Weather Client
    "WeatherClient",
    "WeatherData",
    # Data Validator
    "DGADataValidator",
    "ValidationResult",
    "ValidationIssue",
    "GAS_LIMITS",
    # Synthetic Generator
    "SyntheticTransformerGenerator",
    "TransformerCondition",
    "TransformerConfig",
    "GAS_PROFILES",
]
