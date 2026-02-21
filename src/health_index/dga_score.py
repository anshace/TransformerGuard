"""
DGA (Dissolved Gas Analysis) Score Calculator
Calculates DGA contribution to health index based on Duval fault type and TDCG
Uses IEEE C57.104 limits for severity levels
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from src.diagnosis.multi_method import MultiMethodDiagnosis


@dataclass
class DGAScoreResult:
    """Result of DGA score calculation"""

    score: float  # 0-100
    rating: str  # EXCELLENT, GOOD, FAIR, POOR, CRITICAL
    fault_type: str
    tdcg: float  # Total Dissolved Combustible Gas
    contributing_gases: Dict[str, float]
    confidence: float = 1.0


class DGAScoreCalculator:
    """Calculator for DGA factor in health index"""

    # Duval fault type codes
    FAULT_TYPES = {
        "NORMAL": "Normal/None",
        "PD": "Partial Discharge",
        "D1": "Low-energy Discharge",
        "D2": "High-energy Discharge",
        "T1": "Low-temperature Thermal (<300°C)",
        "T2": "Medium-temperature Thermal (300-700°C)",
        "T3": "High-temperature Thermal (>700°C)",
        "DT": "Thermal Discharge",
    }

    # Default score mapping if config not available
    DEFAULT_SCORE_MAP = {
        4: {"fault_types": ["NORMAL"], "max_tdcg": 720},
        3: {"fault_types": ["PD", "T1"], "max_tdcg": 1920},
        2: {"fault_types": ["T2", "DT"], "max_tdcg": 4630},
        1: {"fault_types": ["D1"], "max_tdcg": 7190},
        0: {"fault_types": ["D2", "T3"], "max_tdcg": 99999},
    }

    def __init__(self, config_path: Optional[str] = None):
        """Initialize DGA calculator with optional config"""
        self.score_map = self.DEFAULT_SCORE_MAP.copy()
        self.config = None

        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path: str):
        """Load configuration from YAML file"""
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
                if "dga_score" in self.config:
                    self.score_map = self.config["dga_score"]
        except FileNotFoundError:
            pass  # Use defaults

    def calculate_tdcg(self, gases: Dict[str, float]) -> float:
        """
        Calculate Total Dissolved Combustible Gas

        Args:
            gases: Dictionary of gas concentrations in ppm
                   (H2, CH4, C2H2, C2H4, C2H6, CO, CO2)

        Returns:
            Total Dissolved Combustible Gas in ppm
        """
        # Key combustible gases per IEEE C57.104
        combustible_gases = ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2"]
        return sum(gases.get(gas, 0) for gas in combustible_gases)

    def determine_fault_type(self, gases: Dict[str, float]) -> str:
        """
        Determine fault type based on gas ratios using multi-method diagnosis

        Args:
            gases: Dictionary of gas concentrations in ppm

        Returns:
            Fault type code (NORMAL, PD, D1, D2, T1, T2, T3, DT)
        """
        # Get gas values (default to 0 if not present)
        # Support both uppercase and lowercase gas keys
        h2 = gases.get("H2", gases.get("h2", 0))
        ch4 = gases.get("CH4", gases.get("ch4", 0))
        c2h2 = gases.get("C2H2", gases.get("c2h2", 0))
        c2h4 = gases.get("C2H4", gases.get("c2h4", 0))
        c2h6 = gases.get("C2H6", gases.get("c2h6", 0))
        co = gases.get("CO", gases.get("co", 0))
        co2 = gases.get("CO2", gases.get("co2", 0))

        # Use multi-method diagnosis for fault detection
        diagnosis = MultiMethodDiagnosis()
        result = diagnosis.diagnose(
            h2=h2, ch4=ch4, c2h2=c2h2, c2h4=c2h4, c2h6=c2h6, co=co, co2=co2
        )

        # Map severity to fault type for scoring
        # The severity gives us a simpler classification
        severity_to_fault = {
            "NORMAL": "NORMAL",
            "LOW": "PD",  # Low severity typically indicates PD or T1
            "MEDIUM": "T2",  # Medium severity indicates T2
            "HIGH": "D1",  # High severity indicates D1 or DT
            "CRITICAL": "D2",  # Critical indicates D2 or T3
            "UNKNOWN": "NORMAL",
        }

        # Get the fault type name from the FaultType enum
        fault_type_name = result.fault_type.name

        # Return the actual fault type from diagnosis
        return fault_type_name

    def calculate_score(
        self,
        gases: Dict[str, float],
        fault_type: Optional[str] = None,
        tdcg: Optional[float] = None,
    ) -> DGAScoreResult:
        """
        Calculate DGA score (0-100) based on fault type and gas levels

        Args:
            gases: Dictionary of gas concentrations in ppm
            fault_type: Optional pre-determined fault type
            tdcg: Optional pre-calculated Total Dissolved Combustible Gas

        Returns:
            DGAScoreResult with score and rating
        """
        # Calculate TDCG if not provided
        if tdcg is None:
            tdcg = self.calculate_tdcg(gases)

        # Determine fault type if not provided
        if fault_type is None:
            fault_type = self.determine_fault_type(gases)

        # Find the appropriate score level based on fault type
        score = 0
        rating = "CRITICAL"

        for level, config in sorted(self.score_map.items(), reverse=True):
            if fault_type in config.get("fault_types", []):
                score = level * 25  # Convert 0-4 scale to 0-100

                # Adjust score based on TDCG within the level
                max_tdcg = config.get("max_tdcg", 99999)
                if tdcg < max_tdcg:
                    # Scale down score if approaching limit
                    if tdcg > max_tdcg * 0.5:
                        score = score * (1 - 0.3 * (tdcg / max_tdcg - 0.5) * 2)
                    break

        # Adjust for normal operation with elevated gases
        if fault_type == "NORMAL" and tdcg > 0:
            if tdcg < 720:
                score = 100 - (tdcg / 720) * 10  # Slight penalty
            elif tdcg < 1920:
                score = 90 - ((tdcg - 720) / 1200) * 20
            else:
                score = 70

        # Ensure score is in valid range
        score = max(0, min(100, score))

        # Determine rating
        rating = self._get_rating(score)

        return DGAScoreResult(
            score=score,
            rating=rating,
            fault_type=fault_type,
            tdcg=tdcg,
            contributing_gases=gases,
            confidence=0.95 if fault_type != "NORMAL" else 0.85,
        )

    def _get_rating(self, score: float) -> str:
        """Get rating string from score"""
        if score >= 85:
            return "EXCELLENT"
        elif score >= 70:
            return "GOOD"
        elif score >= 50:
            return "FAIR"
        elif score >= 25:
            return "POOR"
        else:
            return "CRITICAL"

    def get_fault_description(self, fault_type: str) -> str:
        """Get description for fault type"""
        return self.FAULT_TYPES.get(fault_type, "Unknown")


def calculate_dga_score(
    gases: Dict[str, float],
    fault_type: Optional[str] = None,
    config_path: Optional[str] = "config/health_index_weights.yaml",
) -> DGAScoreResult:
    """
    Convenience function to calculate DGA score

    Args:
        gases: Dictionary of gas concentrations in ppm
        fault_type: Optional pre-determined fault type
        config_path: Path to configuration file

    Returns:
        DGAScoreResult with score and rating
    """
    calculator = DGAScoreCalculator(config_path)
    return calculator.calculate_score(gases, fault_type)
