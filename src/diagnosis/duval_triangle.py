"""
Duval Triangle Method for DGA Fault Diagnosis

This module implements the Duval Triangle method for dissolved gas analysis,
based on IEEE C57.104 and IEC 60599 standards. It includes both the classic
Triangle 1 method (for mineral oil transformers) and the Pentagon method.

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
class DuvalResult:
    """
    Result from Duval Triangle analysis.

    Attributes:
        fault_type: Detected fault type
        confidence: Confidence score (0.0 to 1.0)
        explanation: Human-readable explanation of the diagnosis
        method_name: Name of the method used (DuvalTriangle1 or DuvalPentagon)
        gas_percentages: The calculated gas percentages used for diagnosis
        zone: The specific zone within the triangle where the point falls
    """

    fault_type: FaultType
    confidence: float
    explanation: str
    method_name: str
    gas_percentages: Optional[Dict[str, float]] = None
    zone: Optional[str] = None


class DuvalTriangle1:
    """
    Duval Triangle 1 Method for Mineral Oil Transformers.

    This method uses the percentages of three key gases to determine fault type:
    - CH4 (Methane)
    - C2H4 (Ethylene)
    - C2H2 (Acetylene)

    The percentages are calculated relative to the sum of these three gases.

    Triangle Zones (IEEE C57.104 / IEC 60599):
    - PD: CH4 > 80%, C2H4 < 20%, C2H2 < 2%
    - D1: C2H2 > 13%, C2H4 < 23%
    - D2: C2H2 > 13%, C2H4 >= 23% and < 40%
    - T1: C2H2 < 4%, C2H4 < 20%, CH4 > 50%
    - T2: C2H2 < 4%, C2H4 >= 20% and < 50%
    - T3: C2H2 < 15%, C2H4 >= 50%
    - DT: C2H2 >= 4% and <= 13%

    Example:
        >>> analyzer = DuvalTriangle1()
        >>> result = analyzer.diagnose(h2=50, ch4=150, c2h2=5, c2h4=30, co=200, co2=1500)
        >>> print(result.fault_type)
        FaultType.T2
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Duval Triangle analyzer.

        Args:
            config_path: Optional path to configuration file.
                        If not provided, uses default config.
        """
        self.config = self._load_config(config_path)
        self.method_name = "DuvalTriangle1"

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            # Default config path
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "config",
                "dga_thresholds.yaml",
            )

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                return config.get("duval_triangle", {})
        except (FileNotFoundError, yaml.YAMLError):
            # Return default zones if config not found
            return self._get_default_zones()

    def _get_default_zones(self) -> Dict[str, Any]:
        """Get default triangle zone definitions."""
        return {
            "zones": {
                "PD": {"ch4_min": 80, "c2h4_max": 20, "c2h2_max": 2},
                "D1": {"ch4_max": 40, "c2h4_max": 23, "c2h2_min": 13},
                "D2": {"ch4_max": 40, "c2h4_min": 23, "c2h4_max": 40, "c2h2_min": 13},
                "T1": {"ch4_min": 50, "c2h4_max": 20, "c2h2_max": 4},
                "T2": {"ch4_min": 20, "c2h4_min": 20, "c2h4_max": 50, "c2h2_max": 4},
                "T3": {"ch4_min": 10, "c2h4_min": 50, "c2h2_max": 15},
                "DT": {"ch4_max": 80, "c2h4_max": 50, "c2h2_min": 4, "c2h2_max": 13},
            }
        }

    def _calculate_percentages(
        self, ch4: float, c2h4: float, c2h2: float
    ) -> Tuple[float, float, float]:
        """
        Calculate gas percentages relative to the sum of CH4, C2H4, C2H2.

        Args:
            ch4: Methane concentration in ppm
            c2h4: Ethylene concentration in ppm
            c2h2: Acetylene concentration in ppm

        Returns:
            Tuple of (ch4_pct, c2h4_pct, c2h2_pct)
        """
        total = ch4 + c2h4 + c2h2

        if total == 0:
            return (0.0, 0.0, 0.0)

        ch4_pct = (ch4 / total) * 100
        c2h4_pct = (c2h4 / total) * 100
        c2h2_pct = (c2h2 / total) * 100

        return (ch4_pct, c2h4_pct, c2h2_pct)

    def _check_zone(
        self,
        ch4_pct: float,
        c2h4_pct: float,
        c2h2_pct: float,
        zone_name: str,
        zone_config: Dict[str, Any],
    ) -> bool:
        """
        Check if the gas percentages fall within a specific zone.

        Args:
            ch4_pct: Methane percentage
            c2h4_pct: Ethylene percentage
            c2h2_pct: Acetylene percentage
            zone_name: Name of the zone to check
            zone_config: Zone configuration dictionary

        Returns:
            True if the point falls within the zone, False otherwise
        """
        # Check all defined constraints
        if "ch4_min" in zone_config and ch4_pct < zone_config["ch4_min"]:
            return False
        if "ch4_max" in zone_config and ch4_pct > zone_config["ch4_max"]:
            return False

        if "c2h4_min" in zone_config and c2h4_pct < zone_config["c2h4_min"]:
            return False
        if "c2h4_max" in zone_config and c2h4_pct > zone_config["c2h4_max"]:
            return False

        if "c2h2_min" in zone_config and c2h2_pct < zone_config["c2h2_min"]:
            return False
        if "c2h2_max" in zone_config and c2h2_pct > zone_config["c2h2_max"]:
            return False

        return True

    def _get_fault_type(self, zone_name: str) -> FaultType:
        """Map zone name to FaultType enum."""
        zone_to_fault = {
            "PD": FaultType.PD,
            "D1": FaultType.D1,
            "D2": FaultType.D2,
            "T1": FaultType.T1,
            "T2": FaultType.T2,
            "T3": FaultType.T3,
            "DT": FaultType.DT,
        }
        return zone_to_fault.get(zone_name, FaultType.UNDETERMINED)

    def _generate_explanation(
        self,
        fault_type: FaultType,
        ch4_pct: float,
        c2h4_pct: float,
        c2h2_pct: float,
        zone: str,
    ) -> str:
        """Generate a human-readable explanation of the diagnosis."""
        explanations = {
            FaultType.PD: (
                f"Partial Discharge detected. Gas percentages: CH4={ch4_pct:.1f}%, "
                f"C2H4={c2h4_pct:.1f}%, C2H2={c2h2_pct:.1f}%. This indicates low-energy "
                f"electrical discharge, typically caused by partial breakdown of insulation."
            ),
            FaultType.D1: (
                f"Low Energy Discharge (D1) detected. Gas percentages: CH4={ch4_pct:.1f}%, "
                f"C2H4={c2h4_pct:.1f}%, C2H2={c2h2_pct:.1f}%. This indicates spark discharge "
                f"with limited energy, typically caused by floating potentials or poor connections."
            ),
            FaultType.D2: (
                f"High Energy Discharge (D2) detected. Gas percentages: CH4={ch4_pct:.1f}%, "
                f"C2H4={c2h4_pct:.1f}%, C2H2={c2h2_pct:.1f}%. This indicates high-energy arc "
                f"discharge, which is severe and requires immediate attention."
            ),
            FaultType.T1: (
                f"Low Temperature Thermal Fault (<300°C) detected. Gas percentages: "
                f"CH4={ch4_pct:.1f}%, C2H4={c2h4_pct:.1f}%, C2H2={c2h2_pct:.1f}%. "
                f"This indicates overheating below 300°C, possibly due to poor connections."
            ),
            FaultType.T2: (
                f"Medium Temperature Thermal Fault (300-700°C) detected. Gas percentages: "
                f"CH4={ch4_pct:.1f}%, C2H4={c2h4_pct:.1f}%, C2H2={c2h2_pct:.1f}%. "
                f"This indicates overheating between 300-700°C, possibly due to circulating currents."
            ),
            FaultType.T3: (
                f"High Temperature Thermal Fault (>700°C) detected. Gas percentages: "
                f"CH4={ch4_pct:.1f}%, C2H4={c2h4_pct:.1f}%, C2H2={c2h2_pct:.1f}%. "
                f"This indicates severe overheating above 700°C, possibly due to core faults."
            ),
            FaultType.DT: (
                f"Mixed Discharge and Thermal Fault detected. Gas percentages: "
                f"CH4={ch4_pct:.1f}%, C2H4={c2h4_pct:.1f}%, C2H2={c2h2_pct:.1f}%. "
                f"This indicates a combination of thermal and electrical faults."
            ),
            FaultType.NORMAL: (
                f"Normal operation indicated. Gas percentages: CH4={ch4_pct:.1f}%, "
                f"C2H4={c2h4_pct:.1f}%, C2H2={c2h2_pct:.1f}%. All gases are within normal limits."
            ),
            FaultType.UNDETERMINED: (
                f"Undetermined fault type. Gas percentages: CH4={ch4_pct:.1f}%, "
                f"C2H4={c2h4_pct:.1f}%, C2H2={c2h2_pct:.1f}%. "
                f"The gas combination does not fall within any standard Duval Triangle zone."
            ),
        }
        return explanations.get(fault_type, explanations[FaultType.UNDETERMINED])

    def _calculate_confidence(
        self, ch4_pct: float, c2h4_pct: float, c2h2_pct: float, zone: str
    ) -> float:
        """
        Calculate confidence score based on how clearly the point fits a zone.

        Args:
            ch4_pct: Methane percentage
            c2h4_pct: Ethylene percentage
            c2h2_pct: Acetylene percentage
            zone: The identified zone

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence is higher for zones with more distinctive signatures
        zone_base_confidence = {
            "PD": 0.85,
            "D1": 0.80,
            "D2": 0.85,
            "T1": 0.75,
            "T2": 0.80,
            "T3": 0.85,
            "DT": 0.70,
        }

        base = zone_base_confidence.get(zone, 0.50)

        # Adjust based on how extreme the values are within the zone
        # More extreme values = higher confidence
        if c2h2_pct > 20:
            base += 0.05  # Very high acetylene is clearly discharge
        if c2h4_pct > 60:
            base += 0.05  # Very high ethylene is clearly thermal

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
    ) -> DuvalResult:
        """
        Diagnose fault type using Duval Triangle 1 method.

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
            DuvalResult containing fault type, confidence, and explanation
        """
        # Calculate gas percentages
        ch4_pct, c2h4_pct, c2h2_pct = self._calculate_percentages(ch4, c2h4, c2h2)

        # Check if gases are very low (normal operation) - check this FIRST
        # Duval Triangle doesn't have a NORMAL zone, so we use a threshold
        if ch4 + c2h4 + c2h2 < 10:  # Very low total
            fault_type = FaultType.NORMAL
            explanation = self._generate_explanation(
                fault_type, ch4_pct, c2h4_pct, c2h2_pct, "NORMAL"
            )
            return DuvalResult(
                fault_type=fault_type,
                confidence=0.90,
                explanation=explanation,
                method_name=self.method_name,
                gas_percentages={
                    "CH4": ch4_pct,
                    "C2H4": c2h4_pct,
                    "C2H2": c2h2_pct,
                },
                zone="NORMAL",
            )

        zones = self.config.get("zones", {})

        # Check each zone in order of priority
        # Priority: D1, D2, T1, T2, T3, DT, PD
        zone_order = ["D1", "D2", "T1", "T2", "T3", "DT", "PD"]

        detected_zone = None
        for zone_name in zone_order:
            if zone_name in zones:
                if self._check_zone(
                    ch4_pct, c2h4_pct, c2h2_pct, zone_name, zones[zone_name]
                ):
                    detected_zone = zone_name
                    break

        # If no zone matched, return undetermined
        if detected_zone is None:
            fault_type = FaultType.UNDETERMINED
            detected_zone = "UNDETERMINED"

        fault_type = self._get_fault_type(detected_zone)
        confidence = self._calculate_confidence(
            ch4_pct, c2h4_pct, c2h2_pct, detected_zone
        )
        explanation = self._generate_explanation(
            fault_type, ch4_pct, c2h4_pct, c2h2_pct, detected_zone
        )

        return DuvalResult(
            fault_type=fault_type,
            confidence=confidence,
            explanation=explanation,
            method_name=self.method_name,
            gas_percentages={"CH4": ch4_pct, "C2H4": c2h4_pct, "C2H2": c2h2_pct},
            zone=detected_zone,
        )


class DuvalPentagon:
    """
    Duval Pentagon Method for DGA Fault Diagnosis.

    This is an extended version of the Duval Triangle that uses five gases:
    - CH4 (Methane)
    - C2H4 (Ethylene)
    - C2H2 (Acetylene)
    - CO (Carbon Monoxide)
    - CO2 (Carbon Dioxide)

    The Pentagon method provides additional fault detection capabilities
    and is particularly useful for detecting thermal faults involving
    paper/insulation degradation.

    Note: This is a simplified implementation. For full pentagon analysis,
    refer to IEC 60599 and M. Duval's publications.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Duval Pentagon analyzer."""
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.method_name = "DuvalPentagon"

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
                return config
        except (FileNotFoundError, yaml.YAMLError):
            return {}

    def _calculate_pentagon_percentages(
        self, ch4: float, c2h4: float, c2h2: float, co: float, co2: float
    ) -> Dict[str, float]:
        """Calculate percentages for pentagon method."""
        total = ch4 + c2h4 + c2h2 + co + co2

        if total == 0:
            return {"CH4": 0, "C2H4": 0, "C2H2": 0, "CO": 0, "CO2": 0}

        return {
            "CH4": (ch4 / total) * 100,
            "C2H4": (c2h4 / total) * 100,
            "C2H2": (c2h2 / total) * 100,
            "CO": (co / total) * 100,
            "CO2": (co2 / total) * 100,
        }

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
    ) -> DuvalResult:
        """
        Diagnose fault type using Duval Pentagon method.

        This method primarily uses the triangle gases but also considers
        CO and CO2 for more accurate thermal fault classification.

        Args:
            h2: Hydrogen concentration in ppm
            ch4: Methane concentration in ppm
            c2h2: Acetylene concentration in ppm
            c2h4: Ethylene concentration in ppm
            c2h6: Ethane concentration in ppm
            co: Carbon monoxide concentration in ppm
            co2: Carbon dioxide concentration in ppm
            **kwargs: Additional gas concentrations

        Returns:
            DuvalResult containing fault type, confidence, and explanation
        """
        # Use triangle as base diagnosis
        triangle = DuvalTriangle1(self.config_path)
        base_result = triangle.diagnose(
            h2=h2, ch4=ch4, c2h2=c2h2, c2h4=c2h4, c2h6=c2h6, co=co, co2=co2
        )

        # Enhance with CO/CO2 information for thermal faults
        if base_result.fault_type in [FaultType.T1, FaultType.T2, FaultType.T3]:
            # Check for paper/insulation involvement
            pct = self._calculate_pentagon_percentages(ch4, c2h4, c2h2, co, co2)

            if pct["CO"] > 15 or pct["CO2"] > 30:
                base_result.explanation += (
                    f" High CO ({pct['CO']:.1f}%) and/or CO2 ({pct['CO2']:.1f}%) "
                    f"suggests involvement of paper/insulation degradation."
                )
                base_result.gas_percentages.update(pct)

        # Pentagon typically has slightly higher confidence
        base_result.confidence = min(base_result.confidence * 1.05, 1.0)
        base_result.method_name = self.method_name

        return base_result
