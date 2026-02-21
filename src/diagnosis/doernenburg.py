"""
Doernenburg Ratio Method for DGA Fault Diagnosis

This module implements the Doernenburg Method for dissolved gas analysis,
which uses both key gas detection and ratio limits to determine fault type.

The method focuses on four key gases and their ratios:
- H2 (Hydrogen)
- CH4 (Methane)
- C2H2 (Acetylene)
- C2H4 (Ethylene)

Author: TransformerGuard Team
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

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
class DoernenburgResult:
    """
    Result from Doernenburg analysis.

    Attributes:
        fault_type: Detected fault type
        confidence: Confidence score (0.0 to 1.0)
        explanation: Human-readable explanation of the diagnosis
        method_name: Name of the method used
        detected_key_gases: List of key gases that exceeded limits
        exceeded_ratios: Dictionary of ratios that exceeded limits
    """

    fault_type: FaultType
    confidence: float
    explanation: str
    method_name: str
    detected_key_gases: Optional[List[str]] = None
    exceeded_ratios: Optional[Dict[str, float]] = None


class Doernenburg:
    """
    Doernenburg Ratio Method for DGA Fault Diagnosis.

    This method uses two approaches:
    1. Key Gas Method: Detects when specific gases exceed threshold values
    2. Ratio Method: Checks if specific gas ratios exceed limits

    Key Gas Limits (typical values from IEEE C57.104):
    - H2 > 100 ppm
    - CH4 > 120 ppm
    - C2H2 > 2 ppm
    - C2H4 > 50 ppm

    Ratio Limits:
    - CH4/H2 > 1.0
    - C2H2/C2H4 > 1.0
    - C2H2/CH4 > 0.1

    Example:
        >>> analyzer = Doernenburg()
        >>> result = analyzer.diagnose(h2=150, ch4=200, c2h2=10, c2h4=50)
        >>> print(result.fault_type)
        FaultType.D2
    """

    # Doernenburg ratio limits
    RATIO_LIMITS = {"ch4_h2": 1.0, "c2h2_c2h4": 1.0, "c2h2_ch4": 0.1, "c2h4_c2h6": 1.0}

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Doernenburg analyzer.

        Args:
            config_path: Optional path to configuration file.
        """
        self.config = self._load_config(config_path)
        self.method_name = "Doernenburg"

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
                doernenburg_config = config.get("doernenburg", {})
                return {
                    "key_gas_limits": doernenburg_config.get("key_gas_limits", {}),
                    "ratio_limits": self.RATIO_LIMITS,
                }
        except (FileNotFoundError, yaml.YAMLError):
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "key_gas_limits": {
                "h2": 100,
                "ch4": 120,
                "c2h6": 50,
                "c2h4": 50,
                "c2h2": 2,
                "co": 350,
                "co2": 2500,
            },
            "ratio_limits": self.RATIO_LIMITS,
        }

    def _check_key_gases(
        self,
        h2: float,
        ch4: float,
        c2h2: float,
        c2h4: float,
        c2h6: float,
        co: float,
        co2: float,
    ) -> List[str]:
        """
        Check which key gases exceed their limits.

        Args:
            h2: Hydrogen concentration in ppm
            ch4: Methane concentration in ppm
            c2h2: Acetylene concentration in ppm
            c2h4: Ethylene concentration in ppm
            c2h6: Ethane concentration in ppm
            co: Carbon monoxide concentration in ppm
            co2: Carbon dioxide concentration in ppm

        Returns:
            List of key gases that exceeded their limits
        """
        key_gas_limits = self.config.get("key_gas_limits", {})

        exceeded = []

        if h2 > key_gas_limits.get("h2", 100):
            exceeded.append("H2")
        if ch4 > key_gas_limits.get("ch4", 120):
            exceeded.append("CH4")
        if c2h2 > key_gas_limits.get("c2h2", 2):
            exceeded.append("C2H2")
        if c2h4 > key_gas_limits.get("c2h4", 50):
            exceeded.append("C2H4")
        if c2h6 > key_gas_limits.get("c2h6", 50):
            exceeded.append("C2H6")
        if co > key_gas_limits.get("co", 350):
            exceeded.append("CO")
        if co2 > key_gas_limits.get("co2", 2500):
            exceeded.append("CO2")

        return exceeded

    def _check_ratios(
        self, h2: float, ch4: float, c2h2: float, c2h4: float, c2h6: float
    ) -> Dict[str, float]:
        """
        Check which ratios exceed their limits.

        Args:
            h2: Hydrogen concentration in ppm
            ch4: Methane concentration in ppm
            c2h2: Acetylene concentration in ppm
            c2h4: Ethylene concentration in ppm
            c2h6: Ethane concentration in ppm

        Returns:
            Dictionary of ratios and their values (only exceeded ones)
        """
        ratio_limits = self.config.get("ratio_limits", self.RATIO_LIMITS)

        exceeded = {}

        # CH4/H2
        if h2 > 0:
            ch4_h2 = ch4 / h2
            if ch4_h2 > ratio_limits.get("ch4_h2", 1.0):
                exceeded["CH4/H2"] = ch4_h2

        # C2H2/C2H4
        if c2h4 > 0:
            c2h2_c2h4 = c2h2 / c2h4
            if c2h2_c2h4 > ratio_limits.get("c2h2_c2h4", 1.0):
                exceeded["C2H2/C2H4"] = c2h2_c2h4

        # C2H2/CH4
        if ch4 > 0:
            c2h2_ch4 = c2h2 / ch4
            if c2h2_ch4 > ratio_limits.get("c2h2_ch4", 0.1):
                exceeded["C2H2/CH4"] = c2h2_ch4

        # C2H4/C2H6
        if c2h6 > 0:
            c2h4_c2h6 = c2h4 / c2h6
            if c2h4_c2h6 > ratio_limits.get("c2h4_c2h6", 1.0):
                exceeded["C2H4/C2H6"] = c2h4_c2h6

        return exceeded

    def _determine_fault_type(
        self, key_gases: List[str], exceeded_ratios: Dict[str, float]
    ) -> FaultType:
        """
        Determine fault type based on key gases and exceeded ratios.

        Args:
            key_gases: List of key gases that exceeded limits
            exceeded_ratios: Dictionary of ratios that exceeded limits

        Returns:
            Detected fault type
        """
        # High energy discharge indicators
        if "C2H2" in key_gases and "C2H2/C2H4" in exceeded_ratios:
            if exceeded_ratios.get("C2H2/C2H4", 0) > 3:
                return FaultType.D2
            return FaultType.D1

        # Thermal fault indicators
        if "CH4" in key_gases and "C2H4" in key_gases:
            if "C2H4/C2H6" in exceeded_ratios:
                if exceeded_ratios["C2H4/C2H6"] > 3:
                    return FaultType.T3
                elif exceeded_ratios["C2H4/C2H6"] > 1:
                    return FaultType.T2
                return FaultType.T1

        # Low energy discharge
        if "C2H2" in key_gases:
            return FaultType.D1

        # Hydrogen alone indicates partial discharge
        if "H2" in key_gases and len(key_gases) == 1:
            return FaultType.PD

        # Thermal with CH4
        if "CH4" in key_gases:
            return FaultType.T1

        return FaultType.UNDETERMINED

    def _generate_explanation(
        self,
        fault_type: FaultType,
        key_gases: List[str],
        exceeded_ratios: Dict[str, float],
    ) -> str:
        """Generate explanation for the diagnosis."""
        key_gas_str = ", ".join(key_gases) if key_gases else "None"
        ratio_str = (
            ", ".join(f"{k}={v:.2f}" for k, v in exceeded_ratios.items())
            if exceeded_ratios
            else "None"
        )

        explanations = {
            FaultType.PD: (
                f"Partial Discharge detected. Exceeded key gases: {key_gas_str}. "
                f"Exceeded ratios: {ratio_str}. "
                f"Hydrogen is the primary indicator of partial discharge."
            ),
            FaultType.D1: (
                f"Low Energy Discharge (D1) detected. Exceeded key gases: {key_gas_str}. "
                f"Exceeded ratios: {ratio_str}. "
                f"Acetylene presence indicates spark discharge."
            ),
            FaultType.D2: (
                f"High Energy Discharge (D2) detected. Exceeded key gases: {key_gas_str}. "
                f"Exceeded ratios: {ratio_str}. "
                f"High acetylene/ethylene ratio indicates severe arc discharge."
            ),
            FaultType.T1: (
                f"Low Temperature Thermal Fault (<300°C) detected. "
                f"Exceeded key gases: {key_gas_str}. Exceeded ratios: {ratio_str}. "
                f"Methane is the primary indicator of low-temperature overheating."
            ),
            FaultType.T2: (
                f"Medium Temperature Thermal Fault (300-700°C) detected. "
                f"Exceeded key gases: {key_gas_str}. Exceeded ratios: {ratio_str}. "
                f"Ethylene presence indicates medium-temperature overheating."
            ),
            FaultType.T3: (
                f"High Temperature Thermal Fault (>700°C) detected. "
                f"Exceeded key gases: {key_gas_str}. Exceeded ratios: {ratio_str}. "
                f"High ethylene/ethane ratio indicates severe overheating."
            ),
            FaultType.DT: (
                f"Mixed Discharge and Thermal Fault detected. "
                f"Exceeded key gases: {key_gas_str}. Exceeded ratios: {ratio_str}. "
                f"Multiple indicators suggest combined faults."
            ),
            FaultType.NORMAL: (
                f"Normal operation indicated. All gas levels and ratios are within acceptable limits."
            ),
            FaultType.UNDETERMINED: (
                f"Undetermined fault type. Exceeded key gases: {key_gas_str}. "
                f"Exceeded ratios: {ratio_str}. "
                f"The gas pattern does not match standard Doernenburg fault signatures."
            ),
        }
        return explanations.get(fault_type, explanations[FaultType.UNDETERMINED])

    def _calculate_confidence(
        self,
        fault_type: FaultType,
        key_gases: List[str],
        exceeded_ratios: Dict[str, float],
    ) -> float:
        """
        Calculate confidence based on number of indicators.

        Args:
            fault_type: Detected fault type
            key_gases: List of key gases that exceeded limits
            exceeded_ratios: Dictionary of ratios that exceeded limits

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if fault_type == FaultType.NORMAL:
            return 0.90

        if fault_type == FaultType.UNDETERMINED:
            return 0.30

        # Base confidence
        base = 0.60

        # More key gases = higher confidence
        base += min(len(key_gases) * 0.05, 0.15)

        # More exceeded ratios = higher confidence
        base += min(len(exceeded_ratios) * 0.05, 0.15)

        # Specific strong indicators
        if "C2H2" in key_gases:
            base += 0.10  # Acetylene is a strong indicator

        if "C2H2/C2H4" in exceeded_ratios:
            base += 0.05

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
    ) -> DoernenburgResult:
        """
        Diagnose fault type using Doernenburg method.

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
            DoernenburgResult containing fault type, confidence, and explanation
        """
        # Check key gases
        key_gases = self._check_key_gases(h2, ch4, c2h2, c2h4, c2h6, co, co2)

        # Check ratios
        exceeded_ratios = self._check_ratios(h2, ch4, c2h2, c2h4, c2h6)

        # Determine fault type
        if not key_gases and not exceeded_ratios:
            fault_type = FaultType.NORMAL
        else:
            fault_type = self._determine_fault_type(key_gases, exceeded_ratios)

        # Calculate confidence
        confidence = self._calculate_confidence(fault_type, key_gases, exceeded_ratios)

        # Generate explanation
        explanation = self._generate_explanation(fault_type, key_gases, exceeded_ratios)

        return DoernenburgResult(
            fault_type=fault_type,
            confidence=confidence,
            explanation=explanation,
            method_name=self.method_name,
            detected_key_gases=key_gases,
            exceeded_ratios=exceeded_ratios,
        )
