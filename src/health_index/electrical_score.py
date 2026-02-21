"""
Electrical Test Score Calculator
Calculates electrical test factor based on power factor, capacitance, and insulation resistance
"""

from dataclasses import dataclass
from typing import List, Optional

import yaml


@dataclass
class ElectricalResult:
    """Result of electrical test score calculation"""

    score: float  # 0-100
    rating: str  # EXCELLENT, GOOD, FAIR, POOR, CRITICAL
    power_factor: float  # Dissipation factor (%)
    capacitance: float  # Capacitance in pF
    insulation_resistance: float  # Resistance in MΩ
    issues: List[str] = None
    confidence: float = 1.0

    def __post_init__(self):
        if self.issues is None:
            self.issues = []


class ElectricalScore:
    """Calculator for electrical test factor in health index"""

    # Default thresholds per IEEE standards
    DEFAULT_THRESHOLDS = {
        "power_factor": {
            "excellent": 0.1,  # %
            "good": 0.5,
            "fair": 1.0,
            "poor": 2.0,
        },
        "capacitance_change": {
            "excellent": 2,  # % change from baseline
            "good": 5,
            "fair": 10,
            "poor": 15,
        },
        "insulation_resistance": {
            "excellent": 10000,  # MΩ
            "good": 5000,
            "fair": 2000,
            "poor": 1000,
        },
    }

    # Weight for each parameter
    PARAM_WEIGHTS = {
        "power_factor": 0.50,
        "capacitance_change": 0.25,
        "insulation_resistance": 0.25,
    }

    def __init__(self, config_path: Optional[str] = None):
        """Initialize electrical score calculator with optional config"""
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()
        self.param_weights = self.PARAM_WEIGHTS.copy()

        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path: str):
        """Load configuration from YAML file"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if "electrical_score" in config:
                    self.thresholds = config["electrical_score"]
        except FileNotFoundError:
            pass  # Use defaults

    def _score_parameter(
        self, value: float, thresholds: dict, reverse: bool = False
    ) -> float:
        """
        Score a single parameter (0-100)

        Args:
            value: Measured value
            thresholds: Dictionary with excellent/good/fair/poor thresholds
            reverse: If True, lower is better (e.g., for power factor)

        Returns:
            Score from 0-100
        """
        if reverse:
            # Lower is better
            if value <= thresholds["excellent"]:
                return 100
            elif value <= thresholds["good"]:
                return (
                    90
                    - (
                        (value - thresholds["excellent"])
                        / (thresholds["good"] - thresholds["excellent"])
                    )
                    * 10
                )
            elif value <= thresholds["fair"]:
                return (
                    75
                    - (
                        (value - thresholds["good"])
                        / (thresholds["fair"] - thresholds["good"])
                    )
                    * 15
                )
            elif value <= thresholds["poor"]:
                return (
                    50
                    - (
                        (value - thresholds["fair"])
                        / (thresholds["poor"] - thresholds["fair"])
                    )
                    * 25
                )
            else:
                return max(0, 25 - ((value - thresholds["poor"]) / thresholds["poor"]))
        else:
            # Higher is better (e.g., insulation resistance)
            if value >= thresholds["excellent"]:
                return 100
            elif value >= thresholds["good"]:
                return (
                    90
                    - (
                        (thresholds["excellent"] - value)
                        / (thresholds["excellent"] - thresholds["good"])
                    )
                    * 10
                )
            elif value >= thresholds["fair"]:
                return (
                    75
                    - (
                        (thresholds["good"] - value)
                        / (thresholds["good"] - thresholds["fair"])
                    )
                    * 15
                )
            elif value >= thresholds["poor"]:
                return (
                    50
                    - (
                        (thresholds["fair"] - value)
                        / (thresholds["fair"] - thresholds["poor"])
                    )
                    * 25
                )
            else:
                return max(0, 25 - ((thresholds["poor"] - value) / thresholds["poor"]))

    def calculate_score(
        self,
        power_factor: float,
        capacitance: float,
        insulation_resistance: float,
        baseline_capacitance: Optional[float] = None,
    ) -> ElectricalResult:
        """
        Calculate electrical test score (0-100)

        Args:
            power_factor: Power factor / dissipation factor in %
            capacitance: Measured capacitance in pF
            insulation_resistance: Insulation resistance in MΩ
            baseline_capacitance: Original capacitance for comparison (optional)

        Returns:
            ElectricalResult with score and rating
        """
        issues = []

        # Score power factor (lower is better)
        pf_score = self._score_parameter(
            power_factor, self.thresholds["power_factor"], reverse=True
        )

        # Score capacitance change if baseline provided
        if baseline_capacitance and baseline_capacitance > 0:
            pct_change = (
                abs((capacitance - baseline_capacitance) / baseline_capacitance) * 100
            )
            cap_score = self._score_parameter(
                pct_change, self.thresholds["capacitance_change"], reverse=True
            )
        else:
            # If no baseline, use absolute capacitance values
            # Assume typical transformer capacitance 1000-5000 pF
            if 1000 <= capacitance <= 5000:
                cap_score = 90
            elif capacitance < 1000:
                cap_score = 75
            else:
                cap_score = 70

        # Score insulation resistance (higher is better)
        ir_score = self._score_parameter(
            insulation_resistance,
            self.thresholds["insulation_resistance"],
            reverse=False,
        )

        # Calculate weighted average
        score = (
            pf_score * self.param_weights["power_factor"]
            + cap_score * self.param_weights["capacitance_change"]
            + ir_score * self.param_weights["insulation_resistance"]
        )

        # Identify issues
        if power_factor > self.thresholds["power_factor"]["fair"]:
            issues.append(f"High power factor: {power_factor}%")

        if baseline_capacitance and baseline_capacitance > 0:
            pct_change = (
                abs((capacitance - baseline_capacitance) / baseline_capacitance) * 100
            )
            if pct_change > self.thresholds["capacitance_change"]["fair"]:
                issues.append(f"Capacitance change: {pct_change:.1f}%")

        if insulation_resistance < self.thresholds["insulation_resistance"]["fair"]:
            issues.append(f"Low insulation resistance: {insulation_resistance} MΩ")

        # Ensure score is in valid range
        score = max(0, min(100, score))

        # Determine rating
        rating = self._get_rating(score)

        return ElectricalResult(
            score=score,
            rating=rating,
            power_factor=power_factor,
            capacitance=capacitance,
            insulation_resistance=insulation_resistance,
            issues=issues,
            confidence=0.95 if len(issues) == 0 else 0.85,
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


def calculate_electrical_score(
    power_factor: float,
    capacitance: float,
    insulation_resistance: float,
    baseline_capacitance: Optional[float] = None,
    config_path: Optional[str] = "config/health_index_weights.yaml",
) -> ElectricalResult:
    """
    Convenience function to calculate electrical test score

    Args:
        power_factor: Power factor / dissipation factor in %
        capacitance: Measured capacitance in pF
        insulation_resistance: Insulation resistance in MΩ
        baseline_capacitance: Original capacitance for comparison (optional)
        config_path: Path to configuration file

    Returns:
        ElectricalResult with score and rating
    """
    calculator = ElectricalScore(config_path)
    return calculator.calculate_score(
        power_factor, capacitance, insulation_resistance, baseline_capacitance
    )
