"""
Rogers Ratio Method for DGA Fault Diagnosis

This module implements the Rogers Ratio Method for dissolved gas analysis,
based on IEC 60599 standard. It uses four gas ratios to determine fault type.

The method uses the following ratios:
- C2H2/C2H4 (Acetylene/Ethylene)
- CH4/H2 (Methane/Hydrogen)
- C2H4/C2H6 (Ethylene/Ethane)
- CO2/CO (Carbon Dioxide/Carbon Monoxide)

Author: TransformerGuard Team
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import yaml


class FaultType(Enum):
    """
    Enumeration of fault types detected by DGA methods.

    Based on IEEE C57.104 and IEC 60599 standards:
    - PD: Partial Discharge - Low temperature electrical discharge
    - D1: Low Energy Discharge - Spark discharge
    - D2: High Energy Discharge - Arc discharge
    - T1: Thermal Fault < 300°C - Low temperature overheating
    - T2: Thermal Fault 300-700°C - Medium temperature overheating
    - T3: Thermal Fault > 700°C - High temperature overheating
    - DT: Mixed Thermal/Electrical - Combination of discharge and thermal
    - NORMAL: Normal aging - No significant fault detected
    - UNDETERMINED: Unable to determine fault type
    """

    PD = "Partial Discharge"
    D1 = "Low Energy Discharge"
    D2 = "High Energy Discharge"
    T1 = "Thermal Fault <300°C"
    T2 = "Thermal Fault 300-700°C"
    T3 = "Thermal Fault >700°C"
    DT = "Mixed Discharge/Thermal"
    NORMAL = "Normal"
    UNDETERMINED = "Undetermined"


@dataclass
class RogersResult:
    """
    Result from Rogers Ratio analysis.

    Attributes:
        fault_type: Detected fault type
        confidence: Confidence score (0.0 to 1.0)
        explanation: Human-readable explanation of the diagnosis
        method_name: Name of the method used
        ratios: Dictionary of calculated gas ratios
        code: The 4-character code from ratio interpretation
    """

    fault_type: FaultType
    confidence: float
    explanation: str
    method_name: str
    ratios: Optional[Dict[str, float]] = None
    code: Optional[str] = None


class RogersRatios:
    """
    Rogers Ratio Method for DGA Fault Diagnosis.

    This method interprets four gas ratios to determine fault type:
    - C2H2/C2H4: Indicates electrical discharge severity
    - CH4/H2: Indicates thermal vs electrical nature
    - C2H4/C2H6: Indicates thermal fault temperature
    - CO2/CO: Indicates paper/insulation involvement

    Ratio codes (IEC 60599):
    0: Ratio < limit value
    1: Ratio between limit and 1
    2: Ratio between 1 and 3
    5: Ratio > 3

    Example:
        >>> analyzer = RogersRatios()
        >>> result = analyzer.diagnose(h2=50, ch4=150, c2h2=5, c2h4=30, c2h6=20, co=200, co2=1500)
        >>> print(result.fault_type)
        FaultType.T2
    """

    # Rogers Ratio interpretation table (IEC 60599)
    # Format: (C2H2/C2H4_code, CH4/H2_code, C2H4/C2H6_code, CO2/CO_code) -> FaultType
    INTERPRETATION_TABLE = {
        # PD - Partial Discharge
        (0, 0, 0, 0): FaultType.PD,
        # D1 - Low Energy Discharge
        (1, 0, 0, 0): FaultType.D1,
        (0, 0, 0, 1): FaultType.D1,
        # D2 - High Energy Discharge
        (1, 1, 0, 0): FaultType.D2,
        (1, 2, 0, 0): FaultType.D2,
        (2, 1, 0, 0): FaultType.D2,
        (2, 2, 0, 0): FaultType.D2,
        (0, 1, 0, 0): FaultType.D2,
        (0, 2, 0, 0): FaultType.D2,
        # T1 - Thermal Fault < 300°C
        (0, 0, 1, 0): FaultType.T1,
        (0, 0, 2, 0): FaultType.T1,
        # T2 - Thermal Fault 300-700°C
        (0, 1, 1, 0): FaultType.T2,
        (0, 1, 2, 0): FaultType.T2,
        (0, 2, 1, 0): FaultType.T2,
        (0, 2, 2, 0): FaultType.T2,
        # T3 - Thermal Fault > 700°C
        (0, 0, 2, 1): FaultType.T3,
        (0, 0, 2, 2): FaultType.T3,
        (0, 1, 2, 1): FaultType.T3,
        (0, 1, 2, 2): FaultType.T3,
        (0, 2, 2, 1): FaultType.T3,
        (0, 2, 2, 2): FaultType.T3,
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Rogers Ratio analyzer.

        Args:
            config_path: Optional path to configuration file.
        """
        self.config = self._load_config(config_path)
        self.method_name = "RogersRatios"

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "config",
                "dga_thresholds.yaml",
            )

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                return config.get("rogers_ratios", {})
        except (FileNotFoundError, yaml.YAMLError):
            return self._get_default_thresholds()

    def _get_default_thresholds(self) -> Dict[str, Any]:
        """Get default ratio thresholds."""
        return {
            "c2h2_c2h4": {"limit": 0.1, "high": 1.0, "low": 0.1},
            "ch4_h2": {"limit": 1.0, "high": 3.0, "low": 0.1},
            "c2h4_c2h6": {"limit": 1.0, "high": 3.0, "low": 1.0},
            "co2_co": {"limit": 0.1, "high": 3.0, "low": 0.1},
        }

    def _calculate_ratio(self, numerator: float, denominator: float) -> float:
        """
        Calculate gas ratio safely.

        Args:
            numerator: Numerator gas concentration
            denominator: Denominator gas concentration

        Returns:
            Ratio value, or 0.0 if denominator is zero
        """
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def _get_code(self, ratio: float, ratio_name: str) -> int:
        """
        Get the Rogers ratio code for a given ratio value.

        IEC 60599 standard codes:
        - Code 0: ratio < low_limit
        - Code 1: low_limit ≤ ratio < 1
        - Code 2: 1 ≤ ratio ≤ 3
        - Code 5: ratio > 3

        Args:
            ratio: The calculated ratio value
            ratio_name: Name of the ratio (e.g., 'c2h2_c2h4')

        Returns:
            Code: 0, 1, 2, or 5
        """
        thresholds = self.config.get(ratio_name, {})
        low = thresholds.get("low", 0.1)

        if ratio < low:
            return 0
        elif ratio < 1.0:
            return 1
        elif ratio <= 3.0:
            return 2
        else:
            return 5

    def _get_fault_type_from_code(self, code: Tuple[int, int, int, int]) -> FaultType:
        """Get fault type from Rogers ratio code."""
        return self.INTERPRETATION_TABLE.get(code, FaultType.UNDETERMINED)

    def _generate_explanation(
        self, fault_type: FaultType, ratios: Dict[str, float], code: str
    ) -> str:
        """Generate explanation for the diagnosis."""
        explanations = {
            FaultType.PD: (
                f"Partial Discharge detected. Ratios: C2H2/C2H4={ratios.get('c2h2_c2h4', 0):.3f}, "
                f"CH4/H2={ratios.get('ch4_h2', 0):.3f}, C2H4/C2H6={ratios.get('c2h4_c2h6', 0):.3f}, "
                f"CO2/CO={ratios.get('co2_co', 0):.3f}. Rogers code: {code}. "
                f"This indicates low-energy electrical discharge, typically partial breakdown."
            ),
            FaultType.D1: (
                f"Low Energy Discharge (D1) detected. Ratios: C2H2/C2H4={ratios.get('c2h2_c2h4', 0):.3f}, "
                f"CH4/H2={ratios.get('ch4_h2', 0):.3f}, C2H4/C2H6={ratios.get('c2h4_c2h6', 0):.3f}, "
                f"CO2/CO={ratios.get('co2_co', 0):.3f}. Rogers code: {code}. "
                f"This indicates spark discharge with limited energy."
            ),
            FaultType.D2: (
                f"High Energy Discharge (D2) detected. Ratios: C2H2/C2H4={ratios.get('c2h2_c2h4', 0):.3f}, "
                f"CH4/H2={ratios.get('ch4_h2', 0):.3f}, C2H4/C2H6={ratios.get('c2h4_c2h6', 0):.3f}, "
                f"CO2/CO={ratios.get('co2_co', 0):.3f}. Rogers code: {code}. "
                f"This indicates high-energy arc discharge, which is severe."
            ),
            FaultType.T1: (
                f"Low Temperature Thermal Fault (<300°C) detected. Ratios: "
                f"C2H2/C2H4={ratios.get('c2h2_c2h4', 0):.3f}, CH4/H2={ratios.get('ch4_h2', 0):.3f}, "
                f"C2H4/C2H6={ratios.get('c2h4_c2h6', 0):.3f}, CO2/CO={ratios.get('co2_co', 0):.3f}. "
                f" Rogers code: {code}. This indicates overheating below 300°C."
            ),
            FaultType.T2: (
                f"Medium Temperature Thermal Fault (300-700°C) detected. Ratios: "
                f"C2H2/C2H4={ratios.get('c2h2_c2h4', 0):.3f}, CH4/H2={ratios.get('ch4_h2', 0):.3f}, "
                f"C2H4/C2H6={ratios.get('c2h4_c2h6', 0):.3f}, CO2/CO={ratios.get('co2_co', 0):.3f}. "
                f" Rogers code: {code}. This indicates overheating between 300-700°C."
            ),
            FaultType.T3: (
                f"High Temperature Thermal Fault (>700°C) detected. Ratios: "
                f"C2H2/C2H4={ratios.get('c2h2_c2h4', 0):.3f}, CH4/H2={ratios.get('ch4_h2', 0):.3f}, "
                f"C2H4/C2H6={ratios.get('c2h4_c2h6', 0):.3f}, CO2/CO={ratios.get('co2_co', 0):.3f}. "
                f" Rogers code: {code}. This indicates severe overheating above 700°C."
            ),
            FaultType.DT: (
                f"Mixed Discharge and Thermal Fault detected. Ratios: "
                f"C2H2/C2H4={ratios.get('c2h2_c2h4', 0):.3f}, CH4/H2={ratios.get('ch4_h2', 0):.3f}, "
                f"C2H4/C2H6={ratios.get('c2h4_c2h6', 0):.3f}, CO2/CO={ratios.get('co2_co', 0):.3f}. "
                f" Rogers code: {code}. This indicates combination of faults."
            ),
            FaultType.NORMAL: (
                f"Normal operation indicated. Ratios: C2H2/C2H4={ratios.get('c2h2_c2h4', 0):.3f}, "
                f"CH4/H2={ratios.get('ch4_h2', 0):.3f}, C2H4/C2H6={ratios.get('c2h4_c2h6', 0):.3f}, "
                f"CO2/CO={ratios.get('co2_co', 0):.3f}. Rogers code: {code}. "
                f"All ratios are within normal limits."
            ),
            FaultType.UNDETERMINED: (
                f"Undetermined fault type. Ratios: C2H2/C2H4={ratios.get('c2h2_c2h4', 0):.3f}, "
                f"CH4/H2={ratios.get('ch4_h2', 0):.3f}, C2H4/C2H6={ratios.get('c2h4_c2h6', 0):.3f}, "
                f"CO2/CO={ratios.get('co2_co', 0):.3f}. Rogers code: {code}. "
                f"The ratio combination does not match any standard pattern."
            ),
        }
        return explanations.get(fault_type, explanations[FaultType.UNDETERMINED])

    def _calculate_confidence(
        self, fault_type: FaultType, ratios: Dict[str, float]
    ) -> float:
        """
        Calculate confidence based on ratio clarity.

        Args:
            fault_type: Detected fault type
            ratios: Dictionary of calculated ratios

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence by fault type
        base_confidence = {
            FaultType.PD: 0.75,
            FaultType.D1: 0.70,
            FaultType.D2: 0.80,
            FaultType.T1: 0.70,
            FaultType.T2: 0.75,
            FaultType.T3: 0.80,
            FaultType.DT: 0.65,
            FaultType.NORMAL: 0.85,
            FaultType.UNDETERMINED: 0.30,
        }

        base = base_confidence.get(fault_type, 0.50)

        # Adjust based on ratio clarity
        # Clear ratios (very high or very low) increase confidence
        c2h2_c2h4 = ratios.get("c2h2_c2h4", 0)
        ch4_h2 = ratios.get("ch4_h2", 0)
        c2h4_c2h6 = ratios.get("c2h4_c2h6", 0)
        co2_co = ratios.get("co2_co", 0)

        # Check for extreme values
        clear_indicators = 0
        if c2h2_c2h4 > 3 or c2h2_c2h4 < 0.01:
            clear_indicators += 1
        if ch4_h2 > 3 or ch4_h2 < 0.01:
            clear_indicators += 1
        if c2h4_c2h6 > 3 or c2h4_c2h6 < 0.1:
            clear_indicators += 1
        if co2_co > 3 or co2_co < 0.01:
            clear_indicators += 1

        base += clear_indicators * 0.03

        return min(base, 1.0)

    def diagnose(
        self,
        h2: float = 0,
        ch4: float = 0,
        c2h2: float = 0,
        c2h4: float = 0,
        c2h6: float = 0,
        co: float = 0,
        co2: float = 0,
        **kwargs,
    ) -> RogersResult:
        """
        Diagnose fault type using Rogers Ratio method.

        Args:
            h2: Hydrogen concentration in ppm
            ch4: Methane concentration in ppm
            c2h2: Acetylene concentration in ppm
            c2h4: Ethylene concentration in ppm
            c2h6: Ethane concentration in ppm
            co: Carbon monoxide concentration in ppm
            co2: Carbon dioxide concentration in ppm
            **kwargs: Additional gas concentrations (ignored)

        Returns:
            RogersResult containing fault type, confidence, and explanation
        """
        # Calculate ratios
        ratios = {
            "c2h2_c2h4": self._calculate_ratio(c2h2, c2h4),
            "ch4_h2": self._calculate_ratio(ch4, h2),
            "c2h4_c2h6": self._calculate_ratio(c2h4, c2h6),
            "co2_co": self._calculate_ratio(co2, co),
        }

        # Get codes for each ratio
        codes = (
            self._get_code(ratios["c2h2_c2h4"], "c2h2_c2h4"),
            self._get_code(ratios["ch4_h2"], "ch4_h2"),
            self._get_code(ratios["c2h4_c2h6"], "c2h4_c2h6"),
            self._get_code(ratios["co2_co"], "co2_co"),
        )

        # Convert to string code
        code_str = "".join(str(c) for c in codes)

        # Get fault type from table
        fault_type = self._get_fault_type_from_code(codes)

        # Handle special cases
        # If all ratios are 0 (all gases below limits), it's normal
        if all(c == 0 for c in codes):
            fault_type = FaultType.NORMAL
            code_str = "0000"

        # Calculate confidence
        confidence = self._calculate_confidence(fault_type, ratios)

        # Generate explanation
        explanation = self._generate_explanation(fault_type, ratios, code_str)

        return RogersResult(
            fault_type=fault_type,
            confidence=confidence,
            explanation=explanation,
            method_name=self.method_name,
            ratios=ratios,
            code=code_str,
        )
