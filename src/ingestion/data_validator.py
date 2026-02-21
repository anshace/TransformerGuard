"""
Data Validator for DGA Records

This module provides validation functionality for DGA (Dissolved Gas Analysis)
records, checking for data quality issues, physically impossible values,
and consistency problems.

Author: TransformerGuard Team
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .csv_parser import DGARecord

# Configure module logger
logger = logging.getLogger(__name__)


# Physical limits for gas concentrations (in ppm)
# Based on IEEE C57.104 and practical considerations
GAS_LIMITS = {
    "h2": {
        "min": 0,
        "max": 100000,  # Extremely high, but physically possible
        "warning_high": 1000,  # IEEE Level 3
        "typical_max": 500,  # Typical maximum for healthy transformers
    },
    "ch4": {
        "min": 0,
        "max": 100000,
        "warning_high": 1200,
        "typical_max": 500,
    },
    "c2h6": {
        "min": 0,
        "max": 100000,
        "warning_high": 1000,
        "typical_max": 300,
    },
    "c2h4": {
        "min": 0,
        "max": 100000,
        "warning_high": 1000,
        "typical_max": 400,
    },
    "c2h2": {
        "min": 0,
        "max": 50000,  # Acetylene is particularly concerning
        "warning_high": 50,
        "typical_max": 20,
    },
    "co": {
        "min": 0,
        "max": 100000,
        "warning_high": 1400,
        "typical_max": 700,
    },
    "co2": {
        "min": 0,
        "max": 200000,  # CO2 can be quite high
        "warning_high": 10000,
        "typical_max": 5000,
    },
}

# Temperature limits (°C)
TEMP_LIMITS = {
    "oil_temp": {"min": -20, "max": 150, "warning_high": 105},
    "ambient_temp": {"min": -60, "max": 60, "warning_high": 45},
}

# Load percentage limits
LOAD_LIMITS = {
    "min": 0,
    "max": 200,  # Emergency overload possible
    "warning_high": 100,
}


@dataclass
class ValidationIssue:
    """
    Represents a single validation issue.
    
    Attributes:
        severity: Issue severity ('error', 'warning', 'info')
        field: Field name that has the issue
        message: Human-readable description of the issue
        value: The actual value that caused the issue
        expected: Expected value or range (optional)
    """
    
    severity: str
    field: str
    message: str
    value: Any = None
    expected: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "severity": self.severity,
            "field": self.field,
            "message": self.message,
            "value": self.value,
            "expected": self.expected,
        }


@dataclass
class ValidationResult:
    """
    Result of validating a DGA record.
    
    Attributes:
        is_valid: True if record passes all critical validations
        errors: List of error-level issues (must be fixed)
        warnings: List of warning-level issues (should be reviewed)
        info: List of informational messages
        record_id: ID of the validated record
    """
    
    is_valid: bool
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    info: List[ValidationIssue] = field(default_factory=list)
    record_id: Optional[str] = None
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0
    
    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0
    
    def add_error(
        self, 
        field: str, 
        message: str, 
        value: Any = None, 
        expected: Optional[str] = None
    ) -> None:
        """Add an error-level issue."""
        self.errors.append(ValidationIssue(
            severity="error",
            field=field,
            message=message,
            value=value,
            expected=expected,
        ))
        self.is_valid = False
    
    def add_warning(
        self, 
        field: str, 
        message: str, 
        value: Any = None, 
        expected: Optional[str] = None
    ) -> None:
        """Add a warning-level issue."""
        self.warnings.append(ValidationIssue(
            severity="warning",
            field=field,
            message=message,
            value=value,
            expected=expected,
        ))
    
    def add_info(
        self, 
        field: str, 
        message: str, 
        value: Any = None
    ) -> None:
        """Add an informational message."""
        self.info.append(ValidationIssue(
            severity="info",
            field=field,
            message=message,
            value=value,
        ))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "is_valid": self.is_valid,
            "has_errors": self.has_errors,
            "has_warnings": self.has_warnings,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "info": [i.to_dict() for i in self.info],
            "record_id": self.record_id,
        }
    
    def __str__(self) -> str:
        """String representation of the result."""
        status = "VALID" if self.is_valid else "INVALID"
        parts = [f"Validation Result: {status}"]
        
        if self.errors:
            parts.append(f"  Errors: {len(self.errors)}")
            for e in self.errors[:5]:  # Show first 5
                parts.append(f"    - {e.field}: {e.message}")
        
        if self.warnings:
            parts.append(f"  Warnings: {len(self.warnings)}")
            for w in self.warnings[:5]:
                parts.append(f"    - {w.field}: {w.message}")
        
        return "\n".join(parts)


class DGADataValidator:
    """
    Validator for DGA records.
    
    Performs comprehensive validation including:
    - Non-negative gas concentrations
    - Physically impossible value detection
    - Date reasonability checks
    - Cross-field consistency checks
    - Gas ratio validation
    
    Example:
        >>> validator = DGADataValidator()
        >>> record = DGARecord(
        ...     transformer_id="TR-001",
        ...     sample_date=datetime.now(),
        ...     h2=50, ch4=100, c2h6=30, c2h4=20, c2h2=5, co=200, co2=1000
        ... )
        >>> result = validator.validate(record)
        >>> print(result.is_valid)
        True
    """
    
    def __init__(
        self,
        min_date: Optional[datetime] = None,
        max_future_days: int = 1,
        strict_mode: bool = False,
        custom_limits: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """
        Initialize the validator.
        
        Args:
            min_date: Minimum acceptable date. If None, uses 50 years ago.
            max_future_days: Maximum days in the future allowed for dates.
            strict_mode: If True, treat warnings as errors.
            custom_limits: Custom gas limits to override defaults.
        """
        self.min_date = min_date or (datetime.now() - timedelta(days=365 * 50))
        self.max_future_days = max_future_days
        self.strict_mode = strict_mode
        self.gas_limits = {**GAS_LIMITS}
        
        if custom_limits:
            self.gas_limits.update(custom_limits)
    
    def validate(self, record: DGARecord) -> ValidationResult:
        """
        Validate a DGA record.
        
        Args:
            record: DGARecord to validate.
            
        Returns:
            ValidationResult with validation status and any issues found.
        """
        result = ValidationResult(
            is_valid=True,
            record_id=record.transformer_id,
        )
        
        # Validate transformer ID
        self._validate_transformer_id(record, result)
        
        # Validate sample date
        self._validate_sample_date(record, result)
        
        # Validate gas concentrations
        self._validate_gases(record, result)
        
        # Validate temperatures
        self._validate_temperatures(record, result)
        
        # Validate load percentage
        self._validate_load(record, result)
        
        # Cross-field validations
        self._validate_gas_ratios(record, result)
        self._validate_gas_totals(record, result)
        
        # Apply strict mode
        if self.strict_mode and result.has_warnings:
            result.errors.extend(result.warnings)
            result.warnings = []
            result.is_valid = False
        
        return result
    
    def validate_batch(
        self, 
        records: List[DGARecord]
    ) -> Tuple[List[DGARecord], List[ValidationResult]]:
        """
        Validate a batch of records.
        
        Args:
            records: List of DGARecords to validate.
            
        Returns:
            Tuple of (valid_records, all_results).
        """
        valid_records = []
        all_results = []
        
        for record in records:
            result = self.validate(record)
            all_results.append(result)
            
            if result.is_valid:
                valid_records.append(record)
        
        logger.info(
            f"Validated {len(records)} records: "
            f"{len(valid_records)} valid, {len(records) - len(valid_records)} invalid"
        )
        
        return valid_records, all_results
    
    def _validate_transformer_id(
        self, 
        record: DGARecord, 
        result: ValidationResult
    ) -> None:
        """Validate transformer ID field."""
        if not record.transformer_id:
            result.add_error(
                field="transformer_id",
                message="Transformer ID is required",
                value=record.transformer_id,
            )
        elif not isinstance(record.transformer_id, str):
            result.add_error(
                field="transformer_id",
                message="Transformer ID must be a string",
                value=record.transformer_id,
                expected="string",
            )
        elif len(record.transformer_id.strip()) == 0:
            result.add_error(
                field="transformer_id",
                message="Transformer ID cannot be empty or whitespace",
                value=record.transformer_id,
            )
    
    def _validate_sample_date(
        self, 
        record: DGARecord, 
        result: ValidationResult
    ) -> None:
        """Validate sample date."""
        if not record.sample_date:
            result.add_error(
                field="sample_date",
                message="Sample date is required",
                value=record.sample_date,
            )
            return
        
        if not isinstance(record.sample_date, datetime):
            result.add_error(
                field="sample_date",
                message="Sample date must be a datetime object",
                value=record.sample_date,
            )
            return
        
        # Check if date is too old
        if record.sample_date < self.min_date:
            result.add_error(
                field="sample_date",
                message=f"Sample date is too old (before {self.min_date.strftime('%Y-%m-%d')})",
                value=record.sample_date.strftime('%Y-%m-%d'),
                expected=f">= {self.min_date.strftime('%Y-%m-%d')}",
            )
        
        # Check if date is in the future
        max_date = datetime.now() + timedelta(days=self.max_future_days)
        if record.sample_date > max_date:
            result.add_error(
                field="sample_date",
                message="Sample date is in the future",
                value=record.sample_date.strftime('%Y-%m-%d'),
                expected=f"<= {datetime.now().strftime('%Y-%m-%d')}",
            )
    
    def _validate_gases(
        self, 
        record: DGARecord, 
        result: ValidationResult
    ) -> None:
        """Validate all gas concentrations."""
        gas_fields = ["h2", "ch4", "c2h6", "c2h4", "c2h2", "co", "co2"]
        
        for gas in gas_fields:
            value = getattr(record, gas, None)
            self._validate_single_gas(gas, value, result)
    
    def _validate_single_gas(
        self, 
        gas_name: str, 
        value: Optional[float], 
        result: ValidationResult
    ) -> None:
        """Validate a single gas concentration."""
        limits = self.gas_limits.get(gas_name, {})
        
        if value is None:
            result.add_info(
                field=gas_name,
                message=f"No value provided for {gas_name.upper()}",
            )
            return
        
        # Check for non-negative
        if value < 0:
            result.add_error(
                field=gas_name,
                message=f"{gas_name.upper()} concentration cannot be negative",
                value=value,
                expected=">= 0",
            )
            return
        
        # Check for physically impossible values
        max_limit = limits.get("max", 100000)
        if value > max_limit:
            result.add_error(
                field=gas_name,
                message=f"{gas_name.upper()} concentration is physically impossible",
                value=value,
                expected=f"<= {max_limit} ppm",
            )
        
        # Check for warning-level high values
        warning_high = limits.get("warning_high", float("inf"))
        if value > warning_high:
            result.add_warning(
                field=gas_name,
                message=f"{gas_name.upper()} concentration is very high (IEEE Level 3+)",
                value=value,
                expected=f"<= {warning_high} ppm",
            )
        elif value > limits.get("typical_max", float("inf")):
            result.add_info(
                field=gas_name,
                message=f"{gas_name.upper()} concentration is above typical range",
                value=value,
            )
    
    def _validate_temperatures(
        self, 
        record: DGARecord, 
        result: ValidationResult
    ) -> None:
        """Validate temperature fields."""
        # Oil temperature
        if record.oil_temp is not None:
            limits = TEMP_LIMITS["oil_temp"]
            
            if record.oil_temp < limits["min"]:
                result.add_error(
                    field="oil_temp",
                    message="Oil temperature is below physical minimum",
                    value=record.oil_temp,
                    expected=f">= {limits['min']}°C",
                )
            elif record.oil_temp > limits["max"]:
                result.add_error(
                    field="oil_temp",
                    message="Oil temperature is above physical maximum",
                    value=record.oil_temp,
                    expected=f"<= {limits['max']}°C",
                )
            elif record.oil_temp > limits["warning_high"]:
                result.add_warning(
                    field="oil_temp",
                    message="Oil temperature is very high",
                    value=record.oil_temp,
                    expected=f"<= {limits['warning_high']}°C",
                )
        
        # Ambient temperature
        if record.ambient_temp is not None:
            limits = TEMP_LIMITS["ambient_temp"]
            
            if record.ambient_temp < limits["min"]:
                result.add_error(
                    field="ambient_temp",
                    message="Ambient temperature is below physical minimum",
                    value=record.ambient_temp,
                    expected=f">= {limits['min']}°C",
                )
            elif record.ambient_temp > limits["max"]:
                result.add_error(
                    field="ambient_temp",
                    message="Ambient temperature is above physical maximum",
                    value=record.ambient_temp,
                    expected=f"<= {limits['max']}°C",
                )
        
        # Cross-check: oil temp should typically be higher than ambient
        if (record.oil_temp is not None and 
            record.ambient_temp is not None and
            record.oil_temp < record.ambient_temp - 5):
            result.add_warning(
                field="oil_temp",
                message="Oil temperature is lower than ambient temperature (unusual)",
                value=record.oil_temp,
                expected=f">= ambient ({record.ambient_temp}°C)",
            )
    
    def _validate_load(
        self, 
        record: DGARecord, 
        result: ValidationResult
    ) -> None:
        """Validate load percentage."""
        if record.load_pct is None:
            return
        
        if record.load_pct < LOAD_LIMITS["min"]:
            result.add_error(
                field="load_pct",
                message="Load percentage cannot be negative",
                value=record.load_pct,
                expected=">= 0",
            )
        elif record.load_pct > LOAD_LIMITS["max"]:
            result.add_error(
                field="load_pct",
                message="Load percentage exceeds maximum possible value",
                value=record.load_pct,
                expected=f"<= {LOAD_LIMITS['max']}%",
            )
        elif record.load_pct > LOAD_LIMITS["warning_high"]:
            result.add_warning(
                field="load_pct",
                message="Load percentage indicates overload condition",
                value=record.load_pct,
                expected=f"<= {LOAD_LIMITS['warning_high']}%",
            )
    
    def _validate_gas_ratios(
        self, 
        record: DGARecord, 
        result: ValidationResult
    ) -> None:
        """Validate gas ratios for consistency."""
        # Check for unusual gas ratios that might indicate data entry errors
        
        # C2H2 should not be higher than C2H4 in most cases
        if record.c2h2 > 0 and record.c2h4 > 0:
            if record.c2h2 > record.c2h4 * 2:
                result.add_warning(
                    field="c2h2",
                    message="C2H2/C2H4 ratio is unusually high (possible data entry error)",
                    value=f"{record.c2h2}/{record.c2h4}",
                )
        
        # Check for all zeros (missing data)
        total_gas = (record.h2 + record.ch4 + record.c2h6 + 
                    record.c2h4 + record.c2h2 + record.co + record.co2)
        if total_gas == 0:
            result.add_error(
                field="all_gases",
                message="All gas concentrations are zero (data may be missing)",
                value=0,
            )
        
        # Check for very low combustible gas with high CO/CO2 (paper degradation only)
        tcg = record.total_combustible_gas
        if tcg < 10 and (record.co > 500 or record.co2 > 5000):
            result.add_info(
                field="gas_ratios",
                message="Low combustible gases with elevated CO/CO2 may indicate paper degradation",
                value=f"TCG={tcg}, CO={record.co}, CO2={record.co2}",
            )
    
    def _validate_gas_totals(
        self, 
        record: DGARecord, 
        result: ValidationResult
    ) -> None:
        """Validate gas totals and check for anomalies."""
        tcg = record.total_combustible_gas
        tdg = record.total_dissolved_gas
        
        # Check for extremely high total combustible gas
        if tcg > 10000:
            result.add_warning(
                field="total_combustible_gas",
                message="Total combustible gas is extremely high",
                value=tcg,
                expected="< 10000 ppm",
            )
        
        # Check CO/CO2 ratio (indicator of paper degradation)
        if record.co > 0 and record.co2 > 0:
            co_co2_ratio = record.co / record.co2
            if co_co2_ratio > 0.5:
                result.add_warning(
                    field="co_co2_ratio",
                    message="High CO/CO2 ratio may indicate paper degradation",
                    value=f"{co_co2_ratio:.2f}",
                    expected="< 0.5",
                )
            elif co_co2_ratio < 0.02:
                result.add_info(
                    field="co_co2_ratio",
                    message="Very low CO/CO2 ratio (normal aging pattern)",
                    value=f"{co_co2_ratio:.2f}",
                )
    
    def get_validation_summary(
        self, 
        results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """
        Generate a summary of validation results.
        
        Args:
            results: List of ValidationResult objects.
            
        Returns:
            Dictionary with validation statistics.
        """
        total = len(results)
        valid = sum(1 for r in results if r.is_valid)
        invalid = total - valid
        with_warnings = sum(1 for r in results if r.has_warnings)
        
        # Count issues by field
        field_issues: Dict[str, Dict[str, int]] = {}
        for result in results:
            for issue in result.errors + result.warnings:
                if issue.field not in field_issues:
                    field_issues[issue.field] = {"errors": 0, "warnings": 0}
                if issue.severity == "error":
                    field_issues[issue.field]["errors"] += 1
                else:
                    field_issues[issue.field]["warnings"] += 1
        
        return {
            "total_records": total,
            "valid_records": valid,
            "invalid_records": invalid,
            "records_with_warnings": with_warnings,
            "validation_rate": valid / total if total > 0 else 0,
            "field_issues": field_issues,
        }