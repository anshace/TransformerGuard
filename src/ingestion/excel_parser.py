"""
Excel Parser for Utility DGA Spreadsheets

This module provides functionality to parse DGA (Dissolved Gas Analysis) 
data from Excel files (.xlsx and .xls), supporting multi-sheet workbooks
and transformer metadata extraction.

Author: TransformerGuard Team
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .csv_parser import DGARecord

# Configure module logger
logger = logging.getLogger(__name__)


# Column name mappings for transformer info sheets
TRANSFORMER_INFO_MAPPINGS = {
    "transformer_id": [
        "transformer_id", "transformerid", "transformer", "unit_id", 
        "unitid", "unit", "asset_id", "assetid", "equipment_id", "id",
        "serial_number", "serialnumber", "serial_no"
    ],
    "name": [
        "name", "transformer_name", "transformername", "unit_name", 
        "unitname", "asset_name", "assetname", "description"
    ],
    "manufacturer": [
        "manufacturer", "oem", "make", "brand", "vendor"
    ],
    "manufacture_year": [
        "manufacture_year", "manufactureyear", "year", "mfg_year", 
        "mfgyear", "year_of_manufacture", "yearmanufactured"
    ],
    "installation_year": [
        "installation_year", "installationyear", "install_year", 
        "installyear", "year_installed", "commissioning_year",
        "commissioningyear", "in_service_year"
    ],
    "rated_mva": [
        "rated_mva", "ratedmva", "mva", "capacity_mva", "capacitymva",
        "rating_mva", "ratingmva", "rated_capacity", "power_rating"
    ],
    "rated_kv": [
        "rated_kv", "ratedkv", "kv", "voltage_kv", "voltagekv",
        "rated_voltage", "ratedvoltage", "primary_voltage"
    ],
    "cooling_type": [
        "cooling_type", "coolingtype", "cooling", "cooling_class",
        "coolingclass", "type_of_cooling"
    ],
    "oil_type": [
        "oil_type", "oiltype", "insulation_oil", "insulationoil",
        "oil", "fluid_type", "fluidtype"
    ],
    "location": [
        "location", "site", "substation", "station", "plant"
    ],
    "latitude": [
        "latitude", "lat", "gps_lat", "gpslat", "y_coord"
    ],
    "longitude": [
        "longitude", "lon", "long", "gps_lon", "gpslon", "x_coord"
    ],
}


@dataclass
class TransformerInfo:
    """
    Data class representing transformer nameplate/metadata information.
    
    Attributes:
        transformer_id: Unique identifier for the transformer
        name: Human-readable name of the transformer
        manufacturer: Manufacturer/OEM name
        manufacture_year: Year of manufacture
        installation_year: Year of installation/commissioning
        rated_mva: Rated capacity in MVA
        rated_kv: Rated voltage in kV
        cooling_type: Cooling type (e.g., ONAN, ONAF, OFAF)
        oil_type: Insulation oil type (e.g., Mineral, Synthetic, Natural Ester)
        location: Physical location/substation name
        latitude: GPS latitude coordinate
        longitude: GPS longitude coordinate
        additional_info: Any additional metadata
    """
    
    transformer_id: str
    name: Optional[str] = None
    manufacturer: Optional[str] = None
    manufacture_year: Optional[int] = None
    installation_year: Optional[int] = None
    rated_mva: Optional[float] = None
    rated_kv: Optional[float] = None
    cooling_type: Optional[str] = None
    oil_type: Optional[str] = None
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transformer info to dictionary format."""
        return {
            "transformer_id": self.transformer_id,
            "name": self.name,
            "manufacturer": self.manufacturer,
            "manufacture_year": self.manufacture_year,
            "installation_year": self.installation_year,
            "rated_mva": self.rated_mva,
            "rated_kv": self.rated_kv,
            "cooling_type": self.cooling_type,
            "oil_type": self.oil_type,
            "location": self.location,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "additional_info": self.additional_info,
        }
    
    @property
    def age_years(self) -> Optional[int]:
        """Calculate transformer age in years."""
        if self.installation_year:
            return datetime.now().year - self.installation_year
        if self.manufacture_year:
            return datetime.now().year - self.manufacture_year
        return None


class DGAExcelParser:
    """
    Parser for DGA data from Excel files.
    
    Supports both .xlsx and .xls formats with multi-sheet workbooks.
    Can extract both DGA sample data and transformer metadata.
    
    Features:
    - Support for .xlsx and .xls formats
    - Multi-sheet workbook handling
    - Automatic sheet detection for DGA data vs transformer info
    - Flexible column name matching
    - Comprehensive error handling and logging
    
    Example:
        >>> parser = DGAExcelParser()
        >>> records = parser.parse("transformer_data.xlsx")
        >>> print(f"Parsed {len(records)} DGA records")
        Parsed 50 DGA records
        
        >>> transformer_info = parser.parse_transformer_info("fleet_data.xlsx")
        >>> print(f"Found {len(transformer_info)} transformers")
        Found 10 transformers
    """
    
    def __init__(
        self,
        date_formats: Optional[List[str]] = None,
        strict_mode: bool = False,
        dga_sheet_keywords: Optional[List[str]] = None,
        info_sheet_keywords: Optional[List[str]] = None,
    ):
        """
        Initialize the Excel parser.
        
        Args:
            date_formats: List of date format strings to try.
            strict_mode: If True, raise errors on invalid data.
            dga_sheet_keywords: Keywords to identify DGA data sheets.
            info_sheet_keywords: Keywords to identify transformer info sheets.
        """
        self.date_formats = date_formats or [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%m-%d-%Y",
            "%m/%d/%Y",
        ]
        self.strict_mode = strict_mode
        self.dga_sheet_keywords = dga_sheet_keywords or [
            "dga", "gas", "analysis", "sample", "lab", "oil"
        ]
        self.info_sheet_keywords = info_sheet_keywords or [
            "transformer", "fleet", "asset", "equipment", "nameplate", 
            "info", "inventory", "master"
        ]
    
    def parse(
        self, 
        file_path: Union[str, Path], 
        sheet_name: Optional[str] = None
    ) -> List[DGARecord]:
        """
        Parse an Excel file and return DGA records.
        
        Args:
            file_path: Path to the Excel file.
            sheet_name: Specific sheet name to parse. If None, auto-detects
                       DGA data sheets.
            
        Returns:
            List of DGARecord objects.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If no valid DGA data found.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Excel file not found: {file_path}")
        
        if file_path.suffix.lower() not in (".xlsx", ".xls"):
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Parsing Excel file: {file_path}")
        
        try:
            # Get sheet names
            xl = pd.ExcelFile(file_path)
            sheet_names = xl.sheet_names
            
            if sheet_name:
                if sheet_name not in sheet_names:
                    raise ValueError(f"Sheet '{sheet_name}' not found. Available: {sheet_names}")
                sheets_to_parse = [sheet_name]
            else:
                # Auto-detect DGA sheets
                sheets_to_parse = self._detect_dga_sheets(sheet_names)
            
            if not sheets_to_parse:
                # Try all sheets if no DGA sheets detected
                logger.warning("No DGA sheets detected, trying all sheets")
                sheets_to_parse = sheet_names
            
            all_records = []
            for sheet in sheets_to_parse:
                try:
                    records = self._parse_sheet(file_path, sheet)
                    all_records.extend(records)
                except Exception as e:
                    logger.warning(f"Error parsing sheet '{sheet}': {e}")
                    if self.strict_mode:
                        raise
            
            logger.info(f"Successfully parsed {len(all_records)} records from {file_path}")
            return all_records
            
        except Exception as e:
            logger.error(f"Failed to parse Excel file: {e}")
            raise
    
    def _detect_dga_sheets(self, sheet_names: List[str]) -> List[str]:
        """
        Detect sheets that likely contain DGA data.
        
        Args:
            sheet_names: List of available sheet names.
            
        Returns:
            List of sheet names that likely contain DGA data.
        """
        dga_sheets = []
        for name in sheet_names:
            name_lower = name.lower()
            if any(kw in name_lower for kw in self.dga_sheet_keywords):
                dga_sheets.append(name)
        return dga_sheets
    
    def _parse_sheet(
        self, 
        file_path: Path, 
        sheet_name: str
    ) -> List[DGARecord]:
        """
        Parse a single sheet from the Excel file.
        
        Args:
            file_path: Path to the Excel file.
            sheet_name: Name of the sheet to parse.
            
        Returns:
            List of DGARecord objects from the sheet.
        """
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        if df.empty:
            logger.warning(f"Sheet '{sheet_name}' is empty")
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
            logger.debug(f"Sheet '{sheet_name}' missing columns: {missing_columns}")
            return []
        
        # Parse records
        records = []
        for idx, row in df.iterrows():
            try:
                record = self._parse_row(row, column_map, idx, sheet_name)
                if record is not None:
                    records.append(record)
            except Exception as e:
                logger.warning(f"Error parsing row {idx} in sheet '{sheet_name}': {e}")
                if self.strict_mode:
                    raise
        
        return records
    
    def _normalize_column_name(self, name: str) -> str:
        """Normalize column name for matching."""
        return (
            str(name)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace(".", "_")
            .replace("(", "")
            .replace(")", "")
        )
    
    def _map_columns(self, columns: List[str]) -> Dict[str, str]:
        """Map Excel columns to standard field names."""
        from .csv_parser import COLUMN_MAPPINGS
        
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
        row_idx: int,
        sheet_name: str
    ) -> Optional[DGARecord]:
        """Parse a single row from the Excel sheet."""
        # Extract transformer ID
        transformer_id = str(row[column_map["transformer_id"]]).strip()
        if not transformer_id or transformer_id.lower() in ("nan", "none", ""):
            return None
        
        # Parse sample date
        sample_date = self._parse_date(row[column_map["sample_date"]])
        if sample_date is None:
            return None
        
        # Parse gas concentrations
        h2 = self._parse_float(row[column_map["h2"]], default=0.0)
        ch4 = self._parse_float(row[column_map["ch4"]], default=0.0)
        c2h6 = self._parse_float(row[column_map["c2h6"]], default=0.0)
        c2h4 = self._parse_float(row[column_map["c2h4"]], default=0.0)
        c2h2 = self._parse_float(row[column_map["c2h2"]], default=0.0)
        co = self._parse_float(row[column_map["co"]], default=0.0)
        co2 = self._parse_float(row[column_map["co2"]], default=0.0)
        
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
        
        # Collect additional columns as metadata
        metadata = {"source_sheet": sheet_name}
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
        """Parse a date value using multiple formats."""
        if pd.isna(value):
            return None
        
        if isinstance(value, datetime):
            return value
        
        if isinstance(value, (int, float)):
            # Excel date serial number
            try:
                return pd.to_datetime("1899-12-30") + pd.Timedelta(days=value)
            except Exception:
                pass
        
        value_str = str(value).strip()
        
        for fmt in self.date_formats:
            try:
                return datetime.strptime(value_str, fmt)
            except ValueError:
                continue
        
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
        """Parse a float value from various input types."""
        if pd.isna(value):
            return default
        
        if isinstance(value, (int, float)):
            return float(value)
        
        value_str = str(value).strip().lower()
        
        if value_str in ("nan", "none", "", "-", "n/a", "na", "nd"):
            return default
        
        value_str = value_str.replace(",", "").replace("ppm", "").strip()
        
        try:
            return float(value_str)
        except ValueError:
            return default
    
    def parse_transformer_info(
        self, 
        file_path: Union[str, Path],
        sheet_name: Optional[str] = None
    ) -> List[TransformerInfo]:
        """
        Parse transformer nameplate/metadata from an Excel file.
        
        Args:
            file_path: Path to the Excel file.
            sheet_name: Specific sheet name to parse. If None, auto-detects
                       transformer info sheets.
            
        Returns:
            List of TransformerInfo objects.
            
        Raises:
            FileNotFoundError: If the file does not exist.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Excel file not found: {file_path}")
        
        logger.info(f"Parsing transformer info from: {file_path}")
        
        try:
            xl = pd.ExcelFile(file_path)
            sheet_names = xl.sheet_names
            
            if sheet_name:
                if sheet_name not in sheet_names:
                    raise ValueError(f"Sheet '{sheet_name}' not found. Available: {sheet_names}")
                sheets_to_parse = [sheet_name]
            else:
                sheets_to_parse = self._detect_info_sheets(sheet_names)
            
            if not sheets_to_parse:
                logger.warning("No transformer info sheets detected, trying all sheets")
                sheets_to_parse = sheet_names
            
            all_info = []
            for sheet in sheets_to_parse:
                try:
                    info = self._parse_info_sheet(file_path, sheet)
                    all_info.extend(info)
                except Exception as e:
                    logger.warning(f"Error parsing info sheet '{sheet}': {e}")
                    if self.strict_mode:
                        raise
            
            logger.info(f"Found {len(all_info)} transformer records")
            return all_info
            
        except Exception as e:
            logger.error(f"Failed to parse transformer info: {e}")
            raise
    
    def _detect_info_sheets(self, sheet_names: List[str]) -> List[str]:
        """Detect sheets that likely contain transformer info."""
        info_sheets = []
        for name in sheet_names:
            name_lower = name.lower()
            if any(kw in name_lower for kw in self.info_sheet_keywords):
                info_sheets.append(name)
        return info_sheets
    
    def _parse_info_sheet(
        self, 
        file_path: Path, 
        sheet_name: str
    ) -> List[TransformerInfo]:
        """Parse a single sheet for transformer info."""
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        if df.empty:
            return []
        
        # Normalize column names
        df.columns = [self._normalize_column_name(col) for col in df.columns]
        
        # Map columns
        column_map = self._map_info_columns(df.columns.tolist())
        
        # Check for transformer_id column
        if "transformer_id" not in column_map:
            logger.debug(f"Sheet '{sheet_name}' has no transformer_id column")
            return []
        
        # Parse records
        info_list = []
        for idx, row in df.iterrows():
            try:
                info = self._parse_info_row(row, column_map, idx)
                if info is not None:
                    info_list.append(info)
            except Exception as e:
                logger.warning(f"Error parsing info row {idx}: {e}")
        
        return info_list
    
    def _map_info_columns(self, columns: List[str]) -> Dict[str, str]:
        """Map columns to transformer info field names."""
        column_map = {}
        
        for standard_name, variations in TRANSFORMER_INFO_MAPPINGS.items():
            for col in columns:
                normalized_col = self._normalize_column_name(col)
                if normalized_col in [v.lower().replace(" ", "_") for v in variations]:
                    column_map[standard_name] = col
                    break
        
        return column_map
    
    def _parse_info_row(
        self, 
        row: pd.Series, 
        column_map: Dict[str, str], 
        row_idx: int
    ) -> Optional[TransformerInfo]:
        """Parse a single row for transformer info."""
        transformer_id = str(row[column_map["transformer_id"]]).strip()
        if not transformer_id or transformer_id.lower() in ("nan", "none", ""):
            return None
        
        # Parse optional fields
        name = self._parse_string(row.get(column_map.get("name", "")))
        manufacturer = self._parse_string(row.get(column_map.get("manufacturer", "")))
        manufacture_year = self._parse_int(row.get(column_map.get("manufacture_year", "")))
        installation_year = self._parse_int(row.get(column_map.get("installation_year", "")))
        rated_mva = self._parse_float(row.get(column_map.get("rated_mva", "")))
        rated_kv = self._parse_float(row.get(column_map.get("rated_kv", "")))
        cooling_type = self._parse_string(row.get(column_map.get("cooling_type", "")))
        oil_type = self._parse_string(row.get(column_map.get("oil_type", "")))
        location = self._parse_string(row.get(column_map.get("location", "")))
        latitude = self._parse_float(row.get(column_map.get("latitude", "")))
        longitude = self._parse_float(row.get(column_map.get("longitude", "")))
        
        # Collect additional columns
        additional_info = {}
        mapped_columns = set(column_map.values())
        for col in row.index:
            if col not in mapped_columns and pd.notna(row[col]):
                additional_info[col] = row[col]
        
        return TransformerInfo(
            transformer_id=transformer_id,
            name=name,
            manufacturer=manufacturer,
            manufacture_year=manufacture_year,
            installation_year=installation_year,
            rated_mva=rated_mva,
            rated_kv=rated_kv,
            cooling_type=cooling_type,
            oil_type=oil_type,
            location=location,
            latitude=latitude,
            longitude=longitude,
            additional_info=additional_info,
        )
    
    def _parse_string(self, value: Any) -> Optional[str]:
        """Parse a string value."""
        if pd.isna(value):
            return None
        value_str = str(value).strip()
        if value_str.lower() in ("nan", "none", "", "-"):
            return None
        return value_str
    
    def _parse_int(self, value: Any) -> Optional[int]:
        """Parse an integer value."""
        if pd.isna(value):
            return None
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None
    
    def get_sheet_names(self, file_path: Union[str, Path]) -> List[str]:
        """
        Get all sheet names from an Excel file.
        
        Args:
            file_path: Path to the Excel file.
            
        Returns:
            List of sheet names.
        """
        xl = pd.ExcelFile(file_path)
        return xl.sheet_names
    
    def parse_to_dataframe(
        self, 
        file_path: Union[str, Path], 
        sheet_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Parse an Excel file and return a pandas DataFrame.
        
        Args:
            file_path: Path to the Excel file.
            sheet_name: Specific sheet name to parse.
            
        Returns:
            DataFrame with parsed DGA data.
        """
        records = self.parse(file_path, sheet_name)
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame([r.to_dict() for r in records])
        df["sample_date"] = pd.to_datetime(df["sample_date"])
        
        return df