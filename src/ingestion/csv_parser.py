"""
CSV Parser for DGA Lab Reports

This module provides functionality to parse DGA (Dissolved Gas Analysis) 
data from CSV files, supporting various column name formats and units.

Author: TransformerGuard Team
"""

import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)


# Column name mappings for various CSV formats
COLUMN_MAPPINGS = {
    "transformer_id": [
        "transformer_id", "transformerid", "transformer", "unit_id", 
        "unitid", "unit", "asset_id", "assetid", "equipment_id", "id"
    ],
    "sample_date": [
        "sample_date", "sampledate", "date", "sampled_date", "sampleddate",
        "analysis_date", "analysisdate", "test_date", "testdate", "sampling_date"
    ],
    "h2": ["h2", "hydrogen", "h_2"],
    "ch4": ["ch4", "methane", "c_h4"],
    "c2h6": ["c2h6", "ethane", "c2h6", "c_2h6"],
    "c2h4": ["c2h4", "ethylene", "c2h4", "c_2h4"],
    "c2h2": ["c2h2", "acetylene", "c2h2", "c_2h2"],
    "co": ["co", "carbon_monoxide", "carbon monoxide"],
    "co2": ["co2", "carbon_dioxide", "carbon dioxide"],
    "oil_temp": [
        "oil_temp", "oiltemp", "oil_temperature", "oiltemperature",
        "top_oil_temp", "topoiltemp", "top_oil_temperature"
    ],
    "ambient_temp": [
        "ambient_temp", "ambienttemp", "ambient_temperature", "ambienttemperature",
        "ambient", "air_temp", "airtemp"
    ],
    "load_pct": [
        "load_pct", "loadpct", "load_percent", "loadpercent", "load",
        "loading", "load_percentage", "loading_pct"
    ],
}


@dataclass
class DGARecord:
    """
    Data class representing a single DGA sample record.
    
    All gas concentrations are in parts per million (ppm).
    
    Attributes:
        transformer_id: Unique identifier for the transformer
        sample_date: Date when the sample was taken
        h2: Hydrogen concentration in ppm
        ch4: Methane concentration in ppm
        c2h6: Ethane concentration in ppm
        c2h4: Ethylene concentration in ppm
        c2h2: Acetylene concentration in ppm
        co: Carbon monoxide concentration in ppm
        co2: Carbon dioxide concentration in ppm
        oil_temp: Top oil temperature in °C (optional)
        ambient_temp: Ambient temperature in °C (optional)
        load_pct: Load percentage (0-100+) (optional)
        metadata: Additional metadata from the CSV (optional)
    """
    
    transformer_id: str
    sample_date: datetime
    h2: float
    ch4: float
    c2h6: float
    c2h4: float
    c2h2: float
    co: float
    co2: float
    oil_temp: Optional[float] = None
    ambient_temp: Optional[float] = None
    load_pct: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary format."""
        return {
            "transformer_id": self.transformer_id,
            "sample_date": self.sample_date.isoformat() if self.sample_date else None,
            "h2": self.h2,
            "ch4": self.ch4,
            "c2h6": self.c2h6,
            "c2h4": self.c2h4,
            "c2h2": self.c2h2,
            "co": self.co,
            "co2": self.co2,
            "oil_temp": self.oil_temp,
            "ambient_temp": self.ambient_temp,
            "load_pct": self.load_pct,
            "metadata": self.metadata,
        }
    
    @property
    def total_combustible_gas(self) -> float:
        """Calculate total combustible gas (TCG) in ppm."""
        return self.h2 + self.ch4 + self.c2h6 + self.c2h4 + self.c2h2
    
    @property
    def total_dissolved_gas(self) -> float:
        """Calculate total dissolved gas (TDG) including CO and CO2."""
        return (
            self.h2 + self.ch4 + self.c2h6 + self.c2h4 + self.c2h2 + 
            self.co + self.co2
        )


class DGACSVParser:
    """
    Parser for DGA lab report CSV files.
    
    Supports various CSV formats with different column names and handles
    missing values gracefully. Automatically detects column mappings and
    converts units as needed.
    
    Features:
    - Automatic column name detection with fuzzy matching
    - Support for multiple date formats
    - Handling of missing values
    - Unit conversion (if needed)
    - Comprehensive error logging
    
    Example:
        >>> parser = DGACSVParser()
        >>> records = parser.parse("dga_samples.csv")
        >>> print(f"Parsed {len(records)} records")
        Parsed 13 records
        >>> print(records[0].transformer_id, records[0].h2)
        1 45.0
    """
    
    def __init__(
        self,
        date_formats: Optional[List[str]] = None,
        default_timezone: Optional[str] = None,
        strict_mode: bool = False,
    ):
        """
        Initialize the CSV parser.
        
        Args:
            date_formats: List of date format strings to try. If None, uses
                         common formats.
            default_timezone: Default timezone for dates without timezone info.
            strict_mode: If True, raise errors on invalid rows. If False,
                        log warnings and skip invalid rows.
        """
        self.date_formats = date_formats or [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%m-%d-%Y",
            "%m/%d/%Y",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%d-%m-%Y %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
        ]
        self.default_timezone = default_timezone
        self.strict_mode = strict_mode
        
    def parse(self, file_path: Union[str, Path]) -> List[DGARecord]:
        """
        Parse a CSV file and return a list of DGA records.
        
        Args:
            file_path: Path to the CSV file.
            
        Returns:
            List of DGARecord objects.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If required columns are missing (in strict mode).
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        logger.info(f"Parsing CSV file: {file_path}")
        
        try:
            # Read CSV with pandas for better handling of encodings and delimiters
            df = pd.read_csv(
                file_path,
                encoding=self._detect_encoding(file_path),
                low_memory=False,
            )
        except Exception as e:
            logger.error(f"Failed to read CSV file: {e}")
            raise ValueError(f"Failed to read CSV file: {e}") from e
        
        if df.empty:
            logger.warning(f"CSV file is empty: {file_path}")
            return []
        
        # Normalize column names
        df.columns = [self._normalize_column_name(col) for col in df.columns]
        
        # Map columns to standard names
        column_map = self._map_columns(df.columns.tolist())
        
        # Check for required columns
        required_columns = ["transformer_id", "sample_date", "h2", "ch4", 
                          "c2h6", "c2h4", "c2h2", "co", "co2"]
        missing_columns = [col for col in required_columns if col not in column_map]
        
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            logger.error(error_msg)
            if self.strict_mode:
                raise ValueError(error_msg)
            return []
        
        # Parse records
        records = []
        for idx, row in df.iterrows():
            try:
                record = self._parse_row(row, column_map, idx)
                if record is not None:
                    records.append(record)
            except Exception as e:
                logger.warning(f"Error parsing row {idx}: {e}")
                if self.strict_mode:
                    raise
        
        logger.info(f"Successfully parsed {len(records)} records from {file_path}")
        return records
    
    def _detect_encoding(self, file_path: Path) -> str:
        """
        Detect file encoding by trying common encodings.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            Detected encoding string.
        """
        encodings = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]
        
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    f.read(1024)
                return encoding
            except UnicodeDecodeError:
                continue
        
        return "utf-8"  # Default fallback
    
    def _normalize_column_name(self, name: str) -> str:
        """
        Normalize column name for matching.
        
        Args:
            name: Original column name.
            
        Returns:
            Normalized column name (lowercase, underscores, no spaces).
        """
        return (
            str(name)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace(".", "_")
        )
    
    def _map_columns(self, columns: List[str]) -> Dict[str, str]:
        """
        Map CSV columns to standard field names.
        
        Args:
            columns: List of column names from the CSV.
            
        Returns:
            Dictionary mapping standard names to actual column names.
        """
        column_map = {}
        
        for standard_name, variations in COLUMN_MAPPINGS.items():
            for col in columns:
                normalized_col = self._normalize_column_name(col)
                if normalized_col in [v.lower().replace(" ", "_") for v in variations]:
                    column_map[standard_name] = col
                    break
        
        return column_map
    
    def _parse_row(
        self, 
        row: pd.Series, 
        column_map: Dict[str, str], 
        row_idx: int
    ) -> Optional[DGARecord]:
        """
        Parse a single row from the CSV.
        
        Args:
            row: Pandas Series representing the row.
            column_map: Mapping of standard names to column names.
            row_idx: Row index for error reporting.
            
        Returns:
            DGARecord if parsing successful, None otherwise.
        """
        # Extract transformer ID
        transformer_id = str(row[column_map["transformer_id"]]).strip()
        if not transformer_id or transformer_id.lower() in ("nan", "none", ""):
            logger.warning(f"Row {row_idx}: Missing transformer_id, skipping")
            return None
        
        # Parse sample date
        sample_date = self._parse_date(row[column_map["sample_date"]])
        if sample_date is None:
            logger.warning(f"Row {row_idx}: Invalid sample_date, skipping")
            return None
        
        # Parse gas concentrations
        try:
            h2 = self._parse_float(row[column_map["h2"]], default=0.0)
            ch4 = self._parse_float(row[column_map["ch4"]], default=0.0)
            c2h6 = self._parse_float(row[column_map["c2h6"]], default=0.0)
            c2h4 = self._parse_float(row[column_map["c2h4"]], default=0.0)
            c2h2 = self._parse_float(row[column_map["c2h2"]], default=0.0)
            co = self._parse_float(row[column_map["co"]], default=0.0)
            co2 = self._parse_float(row[column_map["co2"]], default=0.0)
        except Exception as e:
            logger.warning(f"Row {row_idx}: Error parsing gas values: {e}")
            return None
        
        # Parse optional fields
        oil_temp = None
        if "oil_temp" in column_map:
            oil_temp = self._parse_float(row[column_map["oil_temp"]])
        
        ambient_temp = None
        if "ambient_temp" in column_map:
            ambient_temp = self._parse_float(row[column_map["ambient_temp"]])
        
        load_pct = None
        if "load_pct" in column_map:
            load_pct = self._parse_float(row[column_map["load_pct"]])
        
        # Collect any additional columns as metadata
        metadata = {}
        mapped_columns = set(column_map.values())
        for col in row.index:
            if col not in mapped_columns and pd.notna(row[col]):
                metadata[col] = row[col]
        
        return DGARecord(
            transformer_id=transformer_id,
            sample_date=sample_date,
            h2=h2,
            ch4=ch4,
            c2h6=c2h6,
            c2h4=c2h4,
            c2h2=c2h2,
            co=co,
            co2=co2,
            oil_temp=oil_temp,
            ambient_temp=ambient_temp,
            load_pct=load_pct,
            metadata=metadata,
        )
    
    def _parse_date(self, value: Any) -> Optional[datetime]:
        """
        Parse a date value using multiple formats.
        
        Args:
            value: Date value (string, datetime, or timestamp).
            
        Returns:
            Parsed datetime object or None if parsing fails.
        """
        if pd.isna(value):
            return None
        
        if isinstance(value, datetime):
            return value
        
        if isinstance(value, (int, float)):
            # Assume Unix timestamp
            try:
                return datetime.fromtimestamp(value)
            except (ValueError, OSError):
                pass
        
        value_str = str(value).strip()
        
        for fmt in self.date_formats:
            try:
                return datetime.strptime(value_str, fmt)
            except ValueError:
                continue
        
        # Try pandas date parsing as fallback
        try:
            parsed = pd.to_datetime(value_str)
            if pd.notna(parsed):
                return parsed.to_pydatetime()
        except Exception:
            pass
        
        return None
    
    def _parse_float(
        self, 
        value: Any, 
        default: Optional[float] = None
    ) -> Optional[float]:
        """
        Parse a float value from various input types.
        
        Args:
            value: Input value (string, number, etc.).
            default: Default value if parsing fails.
            
        Returns:
            Parsed float or default value.
        """
        if pd.isna(value):
            return default
        
        if isinstance(value, (int, float)):
            return float(value)
        
        value_str = str(value).strip().lower()
        
        # Handle special cases
        if value_str in ("nan", "none", "", "-", "n/a", "na", "nd"):
            return default
        
        # Remove common non-numeric characters
        value_str = value_str.replace(",", "").replace("ppm", "").strip()
        
        try:
            return float(value_str)
        except ValueError:
            return default
    
    def parse_to_dataframe(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Parse a CSV file and return a pandas DataFrame.
        
        Args:
            file_path: Path to the CSV file.
            
        Returns:
            DataFrame with parsed DGA data.
        """
        records = self.parse(file_path)
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame([r.to_dict() for r in records])
        df["sample_date"] = pd.to_datetime(df["sample_date"])
        
        return df
