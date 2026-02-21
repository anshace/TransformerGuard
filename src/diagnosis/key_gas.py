"""
Key Gas Method for DGA Fault Diagnosis

This module implements the Key Gas Method for dissolved gas analysis,
which detects specific fault types based on individual gas concentrations.

The method identifies fault types based on which gases are dominant
and their concentration levels according to IEEE C57.104.

Author: TransformerGuard Team
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

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
class KeyGasResult:
    """
    Result from Key Gas analysis.

    Attributes:
        fault_type: Detected fault type
        confidence: Confidence score (0.0 to 1.0)
        explanation: Human-readable explanation of the diagnosis
        method_name: Name of the method used
        dominant_gas: The gas with highest concentration
        dominant_gas_concentration: Concentration of dominant gas in ppm
        gas_rankings: Sorted list of (gas, concentration) tuples
    """

    fault_type: FaultType
    confidence: float
    explanation: str
    method_name: str
    dominant_gas: Optional[str] = None
    dominant_gas_concentration: Optional[float] = None
    gas_rankings: Optional[List[Tuple[str, float]]] = None


class KeyGasMethod:
    """
    Key Gas Method for DGA Fault Diagnosis.

    This method identifies fault types based on which individual gases
    are present at significant levels:

    Key Gas Associations (IEEE C57.104):
    - H2: Partial Discharge (PD)
    - C2H2: Electrical Discharge (D1, D2)
    - CH4, C2H6: Thermal Faults < 300°C (T1)
    - C2H4: Thermal Faults 300-700°C (T2)
    - CO, CO2: Paper/Insulation involvement

    Gas Limits (IEEE C57.104-2019 Level 1 - Generally Acceptable):
    - H2: 100 ppm
    - CH4: 120 ppm
    - C2H2: 2 ppm
    - C2H4: 50 ppm
    - C2H6: 50 ppm
    - CO: 350 ppm
    - CO2: 2500 ppm

    Example:
        >>> analyzer = KeyGasMethod()
        >>> result = analyzer.diagnose(h2=50, ch4=150, c2h2=5, c2h4=30, co=200, co2=1500)
        >>> print(result.fault_type)
        FaultType.T1
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Key Gas analyzer.

        Args:
            config_path: Optional path to configuration file.
        """
        self.config = self._load_config(config_path)
        self.method_name = "KeyGas"

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
                return config.get("key_gases", {})
        except (FileNotFoundError, yaml.YAMLError):
            return self._get_default_limits()

    def _get_default_limits(self) -> Dict[str, Any]:
        """Get default key gas limits."""
        return {
            "h2": {"typical": 100, "alarm": 100, "warning": 50},
            "ch4": {"typical": 120, "alarm": 120, "warning": 60},
            "c2h6": {"typical": 50, "alarm": 50, "warning": 25},
            "c2h4": {"typical": 50, "alarm": 50, "warning": 20},
            "c2h2": {"typical": 2, "alarm": 2, "warning": 1},
            "co": {"typical": 350, "alarm": 350, "warning": 100},
            "co2": {"typical": 2500, "alarm": 2500, "warning": 1000},
        }

    def _rank_gases(
        self,
        h2: float,
        ch4: float,
        c2h2: float,
        c2h4: float,
        c2h6: float,
        co: float,
        co2: float,
    ) -> List[Tuple[str, float]]:
        """
        Rank gases by concentration.

        Args:
            h2: Hydrogen concentration in ppm
            ch4: Methane concentration in ppm
            c2h2: Acetylene concentration in ppm
            c2h4: Ethylene concentration in ppm
            c2h6: Ethane concentration in ppm
            co: Carbon monoxide concentration in ppm
            co2: Carbon dioxide concentration in ppm

        Returns:
            Sorted list of (gas_name, concentration) tuples, highest first
        """
        gases = [
            ("H2", h2),
            ("CH4", ch4),
            ("C2H2", c2h2),
            ("C2H4", c2h4),
            ("C2H6", c2h6),
            ("CO", co),
            ("CO2", co2),
        ]

        # Sort by concentration (highest first)
        return sorted(gases, key=lambda x: x[1], reverse=True)

    def _determine_fault_type(
        self, gas_rankings: List[Tuple[str, float]], limits: Dict[str, Any]
    ) -> Tuple[FaultType, float]:
        """
        Determine fault type based on dominant gases.

        Args:
            gas_rankings: Sorted list of (gas, concentration) tuples
            limits: Key gas limits

        Returns:
            Tuple of (fault_type, confidence)
        """
        if not gas_rankings or gas_rankings[0][1] == 0:
            return FaultType.NORMAL, 0.90

        dominant_gas, dominant_conc = gas_rankings[0]

        # Check if any gas exceeds alarm level
        alarm_levels = {
            "H2": limits.get("h2", {}).get("alarm", 100),
            "CH4": limits.get("ch4", {}).get("alarm", 120),
            "C2H2": limits.get("c2h2", {}).get("alarm", 2),
            "C2H4": limits.get("c2h4", {}).get("alarm", 50),
            "C2H6": limits.get("c2h6", {}).get("alarm", 50),
            "CO": limits.get("co", {}).get("alarm", 350),
            "CO2": limits.get("co2", {}).get("alarm", 2500),
        }

        # Get top gases that exceed their limits
        exceeding_gases = [
            (gas, conc)
            for gas, conc in gas_rankings
            if gas in alarm_levels and conc > alarm_levels.get(gas, 0)
        ]

        if not exceeding_gases:
            # Check if any gas exceeds warning level
            warning_levels = {
                "H2": limits.get("h2", {}).get("warning", 50),
                "CH4": limits.get("ch4", {}).get("warning", 60),
                "C2H2": limits.get("c2h2", {}).get("warning", 1),
                "C2H4": limits.get("c2h4", {}).get("warning", 20),
                "C2H6": limits.get("c2h6", {}).get("warning", 25),
                "CO": limits.get("co", {}).get("warning", 100),
                "CO2": limits.get("co2", {}).get("warning", 1000),
            }

            warning_gases = [
                (gas, conc)
                for gas, conc in gas_rankings
                if gas in warning_levels and conc > warning_levels.get(gas, 0)
            ]

            if not warning_gases:
                return FaultType.NORMAL, 0.85
            else:
                exceeding_gases = warning_gases

        # Determine fault type based on dominant exceeding gas
        fault_type = FaultType.UNDETERMINED
        base_confidence = 0.60

        # Get all exceeding gases for combined analysis
        exceeding_names = [g for g, _ in exceeding_gases]

        # Electrical discharge (C2H2 is the key indicator)
        if "C2H2" in exceeding_names:
            c2h2_value = next((conc for gas, conc in gas_rankings if gas == "C2H2"), 0)
            if c2h2_value > 20:
                fault_type = FaultType.D2
                base_confidence = 0.85
            elif c2h2_value > 5:
                fault_type = FaultType.D1
                base_confidence = 0.75
            else:
                fault_type = FaultType.D1
                base_confidence = 0.70

        # Thermal faults - check combination of gases
        elif dominant_gas == "H2" and len(exceeding_names) == 1:
            fault_type = FaultType.PD
            base_confidence = 0.70

        # Thermal < 300°C - CH4 and C2H6 dominant
        elif "CH4" in exceeding_names or "C2H6" in exceeding_names:
            c2h4_value = next((conc for gas, conc in gas_rankings if gas == "C2H4"), 0)
            if c2h4_value > 30:
                fault_type = FaultType.T2
                base_confidence = 0.75
            else:
                fault_type = FaultType.T1
                base_confidence = 0.70

        # Thermal 300-700°C - C2H4 dominant
        elif dominant_gas == "C2H4":
            c2h4_value = next((conc for gas, conc in gas_rankings if gas == "C2H4"), 0)
            c2h2_value = next((conc for gas, conc in gas_rankings if gas == "C2H2"), 0)
            if c2h4_value > 50 and c2h2_value < 5:
                fault_type = FaultType.T3
                base_confidence = 0.80
            else:
                fault_type = FaultType.T2
                base_confidence = 0.75

        # CO and CO2 indicate paper/insulation involvement
        elif "CO" in exceeding_names or "CO2" in exceeding_names:
            ch4_value = next((conc for gas, conc in gas_rankings if gas == "CH4"), 0)
            c2h4_value = next((conc for gas, conc in gas_rankings if gas == "C2H4"), 0)

            if c2h4_value > ch4_value:
                fault_type = FaultType.T3
            else:
                fault_type = FaultType.T2
            base_confidence = 0.65

        # Mixed faults
        elif len(exceeding_names) >= 3:
            fault_type = FaultType.DT
            base_confidence = 0.60

        return fault_type, base_confidence

    def _generate_explanation(
        self,
        fault_type: FaultType,
        gas_rankings: List[Tuple[str, float]],
        confidence: float,
    ) -> str:
        """Generate explanation for the diagnosis."""
        if not gas_rankings:
            return "No gas data available for analysis."

        dominant_gas, dominant_conc = gas_rankings[0]

        # Format gas rankings
        gas_str = ", ".join(f"{g}:{c:.0f}ppm" for g, c in gas_rankings[:5])

        explanations = {
            FaultType.PD: (
                f"Partial Discharge detected. Dominant gas: {dominant_gas}={dominant_conc:.0f}ppm. "
                f"Gas concentrations: {gas_str}. "
                f"Hydrogen is the primary indicator of partial discharge in oil."
            ),
            FaultType.D1: (
                f"Low Energy Discharge (D1) detected. Dominant gas: {dominant_gas}={dominant_conc:.0f}ppm. "
                f"Gas concentrations: {gas_str}. "
                f"Acetylene presence indicates spark or low-energy discharge."
            ),
            FaultType.D2: (
                f"High Energy Discharge (D2) detected. Dominant gas: {dominant_gas}={dominant_conc:.0f}ppm. "
                f"Gas concentrations: {gas_str}. "
                f"High acetylene levels indicate severe arc discharge."
            ),
            FaultType.T1: (
                f"Low Temperature Thermal Fault (<300°C) detected. "
                f"Dominant gas: {dominant_gas}={dominant_conc:.0f}ppm. "
                f"Gas concentrations: {gas_str}. "
                f"Methane and ethane indicate low-temperature overheating."
            ),
            FaultType.T2: (
                f"Medium Temperature Thermal Fault (300-700°C) detected. "
                f"Dominant gas: {dominant_gas}={dominant_conc:.0f}ppm. "
                f"Gas concentrations: {gas_str}. "
                f"Ethylene indicates medium-temperature overheating."
            ),
            FaultType.T3: (
                f"High Temperature Thermal Fault (>700°C) detected. "
                f"Dominant gas: {dominant_gas}={dominant_conc:.0f}ppm. "
                f"Gas concentrations: {gas_str}. "
                f"High ethylene indicates severe thermal fault."
            ),
            FaultType.DT: (
                f"Mixed Discharge and Thermal Fault detected. "
                f"Dominant gas: {dominant_gas}={dominant_conc:.0f}ppm. "
                f"Gas concentrations: {gas_str}. "
                f"Multiple gas types indicate combined fault modes."
            ),
            FaultType.NORMAL: (
                f"Normal operation indicated. Dominant gas: {dominant_gas}={dominant_conc:.0f}ppm. "
                f"Gas concentrations: {gas_str}. "
                f"All gas levels are within acceptable limits."
            ),
            FaultType.UNDETERMINED: (
                f"Undetermined fault type. Dominant gas: {dominant_gas}={dominant_conc:.0f}ppm. "
                f"Gas concentrations: {gas_str}. "
                f"The gas pattern does not match standard key gas signatures."
            ),
        }

        return explanations.get(fault_type, explanations[FaultType.UNDETERMINED])

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
    ) -> KeyGasResult:
        """
        Diagnose fault type using Key Gas method.

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
            KeyGasResult containing fault type, confidence, and explanation
        """
        # Rank gases
        gas_rankings = self._rank_gases(h2, ch4, c2h2, c2h4, c2h6, co, co2)

        # Determine fault type
        fault_type, base_confidence = self._determine_fault_type(
            gas_rankings, self.config
        )

        # Adjust confidence based on how dominant the key gas is
        if gas_rankings:
            dominant_gas, dominant_conc = gas_rankings[0]
            if dominant_conc > 0:
                # Calculate dominance ratio
                total = sum(conc for _, conc in gas_rankings)
                dominance = dominant_conc / total if total > 0 else 0

                # Higher dominance = higher confidence
                if dominance > 0.7:
                    base_confidence += 0.10
                elif dominance > 0.5:
                    base_confidence += 0.05

        confidence = min(base_confidence, 1.0)

        # Generate explanation
        explanation = self._generate_explanation(fault_type, gas_rankings, confidence)

        dominant_gas = gas_rankings[0][0] if gas_rankings else None
        dominant_conc = gas_rankings[0][1] if gas_rankings else None

        return KeyGasResult(
            fault_type=fault_type,
            confidence=confidence,
            explanation=explanation,
            method_name=self.method_name,
            dominant_gas=dominant_gas,
            dominant_gas_concentration=dominant_conc,
            gas_rankings=gas_rankings,
        )
