"""
Oil Quality Score Calculator
Calculates oil quality factor based on dielectric strength, moisture, acidity, and IFT
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml


@dataclass
class OilQualityResult:
    """Result of oil quality score calculation"""

    score: float  # 0-100
    rating: str  # EXCELLENT, GOOD, FAIR, POOR, CRITICAL
    dielectric_strength: float  # kV
    moisture_content: float  # ppm
    acidity: float  # mgKOH/g
    interfacial_tension: Optional[float] = None  # mN/m
    issues: List[str] = None
    confidence: float = 1.0

    def __post_init__(self):
        if self.issues is None:
            self.issues = []


class OilQualityScore:
    """Calculator for oil quality factor in health index"""

    # Default thresholds from config/IEEE standards
    DEFAULT_THRESHOLDS = {
        "dielectric_strength": {
            "excellent": 60,  # kV
            "good": 50,
            "fair": 40,
            "poor": 30,
        },
        "moisture_content": {
            "excellent": 10,  # ppm
            "good": 15,
            "fair": 25,
            "poor": 35,
        },
        "acidity": {
            "excellent": 0.1,  # mgKOH/g
            "good": 0.2,
            "fair": 0.3,
            "poor": 0.5,
        },
        "interfacial_tension": {
            "excellent": 40,  # mN/m
            "good": 35,
            "fair": 28,
            "poor": 20,
        },
    }

    # Weight for each parameter
    PARAM_WEIGHTS = {
        "dielectric_strength": 0.35,
        "moisture_content": 0.30,
        "acidity": 0.25,
        "interfacial_tension": 0.10,
    }

    def __init__(self, config_path: Optional[str] = None):
        """Initialize oil quality calculator with optional config"""
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()
        self.param_weights = self.PARAM_WEIGHTS.copy()

        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path: str):
        """Load configuration from YAML file"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if "oil_quality_score" in config:
                    # Merge config with defaults to ensure all keys exist
                    for key, value in config["oil_quality_score"].items():
                        self.thresholds[key] = value
        except FileNotFoundError:
            pass  # Use defaults

    def _score_parameter(
        self, value: float, thresholds: Dict[str, float], reverse: bool = False
    ) -> float:
        """
        Score a single parameter (0-100)

        Args:
            value: Measured value
            thresholds: Dictionary with excellent/good/fair/poor thresholds
            reverse: If True, lower is better (e.g., for moisture)

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
                return max(0, 25 - ((value - thresholds["poor"]) / 10))
        else:
            # Higher is better (e.g., dielectric strength)
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
                return max(0, 25 - ((thresholds["poor"] - value) / 5))

    def calculate_score(
        self,
        dielectric_strength: float,
        moisture_content: float,
        acidity: float,
        interfacial_tension: Optional[float] = None,
    ) -> OilQualityResult:
        """
        Calculate oil quality score (0-100)

        Args:
            dielectric_strength: Dielectric breakdown voltage in kV
            moisture_content: Moisture content in ppm
            acidity: Acid number in mgKOH/g
            interfacial_tension: IFT in mN/m (optional)

        Returns:
            OilQualityResult with score and rating
        """
        issues = []

        # Score each parameter
        ds_score = self._score_parameter(
            dielectric_strength, self.thresholds["dielectric_strength"], reverse=False
        )

        moisture_score = self._score_parameter(
            moisture_content, self.thresholds["moisture_content"], reverse=True
        )

        acidity_score = self._score_parameter(
            acidity, self.thresholds["acidity"], reverse=True
        )

        # Calculate weighted average
        score = (
            ds_score * self.param_weights["dielectric_strength"]
            + moisture_score * self.param_weights["moisture_content"]
            + acidity_score * self.param_weights["acidity"]
        )

        # Include IFT if provided
        if interfacial_tension is not None:
            ift_score = self._score_parameter(
                interfacial_tension,
                self.thresholds["interfacial_tension"],
                reverse=False,
            )
            # Normalize weights for 4 parameters
            total_weight = (
                self.param_weights["dielectric_strength"]
                + self.param_weights["moisture_content"]
                + self.param_weights["acidity"]
                + self.param_weights["interfacial_tension"]
            )

            score = (
                (
                    ds_score * self.param_weights["dielectric_strength"]
                    + moisture_score * self.param_weights["moisture_content"]
                    + acidity_score * self.param_weights["acidity"]
                    + ift_score * self.param_weights["interfacial_tension"]
                )
                / total_weight
                * 100
            )
        else:
            # Normalize weights for 3 parameters
            total_weight = (
                self.param_weights["dielectric_strength"]
                + self.param_weights["moisture_content"]
                + self.param_weights["acidity"]
            )
            score = (
                (
                    ds_score * self.param_weights["dielectric_strength"]
                    + moisture_score * self.param_weights["moisture_content"]
                    + acidity_score * self.param_weights["acidity"]
                )
                / total_weight
                * 100
            )

        # Identify issues
        if dielectric_strength < self.thresholds["dielectric_strength"]["fair"]:
            issues.append(f"Low dielectric strength: {dielectric_strength} kV")
        if moisture_content > self.thresholds["moisture_content"]["fair"]:
            issues.append(f"High moisture content: {moisture_content} ppm")
        if acidity > self.thresholds["acidity"]["fair"]:
            issues.append(f"High acidity: {acidity} mgKOH/g")
        if interfacial_tension is not None:
            if interfacial_tension < self.thresholds["interfacial_tension"]["fair"]:
                issues.append(f"Low interfacial tension: {interfacial_tension} mN/m")

        # Ensure score is in valid range
        score = max(0, min(100, score))

        # Determine rating
        rating = self._get_rating(score)

        return OilQualityResult(
            score=score,
            rating=rating,
            dielectric_strength=dielectric_strength,
            moisture_content=moisture_content,
            acidity=acidity,
            interfacial_tension=interfacial_tension,
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


def calculate_oil_quality_score(
    dielectric_strength: float,
    moisture_content: float,
    acidity: float,
    interfacial_tension: Optional[float] = None,
    config_path: Optional[str] = "config/health_index_weights.yaml",
) -> OilQualityResult:
    """
    Convenience function to calculate oil quality score

    Args:
        dielectric_strength: Dielectric breakdown voltage in kV
        moisture_content: Moisture content in ppm
        acidity: Acid number in mgKOH/g
        interfacial_tension: IFT in mN/m (optional)
        config_path: Path to configuration file

    Returns:
        OilQualityResult with score and rating
    """
    calculator = OilQualityScore(config_path)
    return calculator.calculate_score(
        dielectric_strength, moisture_content, acidity, interfacial_tension
    )
