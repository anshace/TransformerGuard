"""
Composite Health Index Calculator
Combines all factors into weighted composite score
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

from .age_score import AgeResult, AgeScore
from .dga_score import DGAScoreCalculator, DGAScoreResult
from .electrical_score import ElectricalResult, ElectricalScore
from .loading_score import LoadingResult, LoadingScore
from .oil_quality_score import OilQualityResult, OilQualityScore


@dataclass
class HealthIndexResult:
    """Result of composite health index calculation"""

    health_index: float  # 0-100
    category: str  # EXCELLENT, GOOD, FAIR, POOR, CRITICAL
    category_color: str  # Color code
    component_scores: Dict[str, float] = field(default_factory=dict)
    weights_used: Dict[str, float] = field(default_factory=dict)
    risk_level: str = "LOW"
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 1.0


class CompositeHealthIndex:
    """Calculator for composite health index"""

    # Default weights (must sum to 1.0)
    DEFAULT_WEIGHTS = {
        "dga": 0.35,
        "oil_quality": 0.20,
        "electrical": 0.15,
        "age": 0.15,
        "loading": 0.15,
    }

    # Category definitions
    CATEGORIES = {
        "excellent": {"min": 85, "max": 100, "color": "#2ecc71"},
        "good": {"min": 70, "max": 84, "color": "#27ae60"},
        "fair": {"min": 50, "max": 69, "color": "#f39c12"},
        "poor": {"min": 25, "max": 49, "color": "#e67e22"},
        "critical": {"min": 0, "max": 24, "color": "#e74c3c"},
    }

    # Risk level thresholds
    RISK_THRESHOLDS = {"LOW": 70, "MODERATE": 50, "HIGH": 25, "CRITICAL": 0}

    def __init__(self, config_path: Optional[str] = None):
        """Initialize composite health index calculator"""
        self.weights = self.DEFAULT_WEIGHTS.copy()
        self.categories = self.CATEGORIES.copy()
        self.config = None

        # Initialize component calculators
        self.dga_calculator = DGAScoreCalculator(config_path)
        self.oil_calculator = OilQualityScore(config_path)
        self.electrical_calculator = ElectricalScore(config_path)
        self.age_calculator = AgeScore(config_path)
        self.loading_calculator = LoadingScore(config_path)

        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path: str):
        """Load configuration from YAML file"""
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
                if "weights" in self.config:
                    self.weights = self.config["weights"]
                if "categories" in self.config:
                    self.categories = self.config["categories"]
        except FileNotFoundError:
            pass  # Use defaults

    def _validate_weights(self) -> bool:
        """Validate that weights sum to 1.0"""
        total = sum(self.weights.values())
        return abs(total - 1.0) < 0.001

    def calculate_dga(
        self,
        gases: Dict[str, float],
        fault_type: Optional[str] = None,
        tdcg: Optional[float] = None,
    ) -> DGAScoreResult:
        """Calculate DGA component score"""
        return self.dga_calculator.calculate_score(gases, fault_type, tdcg)

    def calculate_oil_quality(
        self,
        dielectric_strength: float,
        moisture_content: float,
        acidity: float,
        interfacial_tension: Optional[float] = None,
    ) -> OilQualityResult:
        """Calculate oil quality component score"""
        return self.oil_calculator.calculate_score(
            dielectric_strength, moisture_content, acidity, interfacial_tension
        )

    def calculate_electrical(
        self,
        power_factor: float,
        capacitance: float,
        insulation_resistance: float,
        baseline_capacitance: Optional[float] = None,
    ) -> ElectricalResult:
        """Calculate electrical component score"""
        return self.electrical_calculator.calculate_score(
            power_factor, capacitance, insulation_resistance, baseline_capacitance
        )

    def calculate_age(
        self, age_years: float, expected_life_years: Optional[float] = None
    ) -> AgeResult:
        """Calculate age component score"""
        return self.age_calculator.calculate_score(age_years, expected_life_years)

    def calculate_loading(
        self,
        average_load_percent: float,
        peak_load_percent: Optional[float] = None,
        overload_hours: int = 0,
    ) -> LoadingResult:
        """Calculate loading component score"""
        return self.loading_calculator.calculate_score(
            average_load_percent, peak_load_percent, overload_hours
        )

    def calculate(
        self,
        dga_gases: Optional[Dict[str, float]] = None,
        dielectric_strength: Optional[float] = None,
        moisture_content: Optional[float] = None,
        acidity: Optional[float] = None,
        interfacial_tension: Optional[float] = None,
        power_factor: Optional[float] = None,
        capacitance: Optional[float] = None,
        insulation_resistance: Optional[float] = None,
        baseline_capacitance: Optional[float] = None,
        age_years: Optional[float] = None,
        expected_life_years: Optional[float] = None,
        average_load_percent: Optional[float] = None,
        peak_load_percent: Optional[float] = None,
        overload_hours: int = 0,
        fault_type: Optional[str] = None,
        tdcg: Optional[float] = None,
    ) -> HealthIndexResult:
        """
        Calculate composite health index

        Args:
            dga_gases: DGA gas concentrations (optional)
            dielectric_strength: Oil dielectric strength in kV (optional)
            moisture_content: Moisture content in ppm (optional)
            acidity: Acidity in mgKOH/g (optional)
            interfacial_tension: IFT in mN/m (optional)
            power_factor: Power factor in % (optional)
            capacitance: Capacitance in pF (optional)
            insulation_resistance: Insulation resistance in MΩ (optional)
            baseline_capacitance: Baseline capacitance for comparison (optional)
            age_years: Transformer age in years (optional)
            expected_life_years: Expected life in years (optional)
            average_load_percent: Average load % (optional)
            peak_load_percent: Peak load % (optional)
            overload_hours: Overload hours (optional)
            fault_type: Pre-determined fault type (optional)
            tdcg: Pre-calculated TDCG (optional)

        Returns:
            HealthIndexResult with composite score
        """
        # Calculate component scores
        component_scores = {}

        # DGA score
        if dga_gases is not None:
            dga_result = self.calculate_dga(dga_gases, fault_type, tdcg)
            component_scores["dga"] = dga_result.score
        else:
            component_scores["dga"] = None

        # Oil quality score
        if all(x is not None for x in [dielectric_strength, moisture_content, acidity]):
            oil_result = self.calculate_oil_quality(
                dielectric_strength, moisture_content, acidity, interfacial_tension
            )
            component_scores["oil_quality"] = oil_result.score
        else:
            component_scores["oil_quality"] = None

        # Electrical score
        if all(
            x is not None for x in [power_factor, capacitance, insulation_resistance]
        ):
            electrical_result = self.calculate_electrical(
                power_factor, capacitance, insulation_resistance, baseline_capacitance
            )
            component_scores["electrical"] = electrical_result.score
        else:
            component_scores["electrical"] = None

        # Age score
        if age_years is not None:
            age_result = self.calculate_age(age_years, expected_life_years)
            component_scores["age"] = age_result.score
        else:
            component_scores["age"] = None

        # Loading score
        if average_load_percent is not None:
            loading_result = self.calculate_loading(
                average_load_percent, peak_load_percent, overload_hours
            )
            component_scores["loading"] = loading_result.score
        else:
            component_scores["loading"] = None

        # Calculate weighted health index
        health_index = self._calculate_weighted_score(component_scores)

        # Get category
        category, category_color = self._get_category(health_index)

        # Determine risk level
        risk_level = self._get_risk_level(health_index)

        # Generate recommendations
        recommendations = self._generate_recommendations(component_scores, health_index)

        # Calculate confidence
        available_components = sum(
            1 for v in component_scores.values() if v is not None
        )
        confidence = available_components / len(component_scores)

        return HealthIndexResult(
            health_index=health_index,
            category=category,
            category_color=category_color,
            component_scores=component_scores,
            weights_used=self.weights.copy(),
            risk_level=risk_level,
            recommendations=recommendations,
            confidence=confidence,
        )

    def _calculate_weighted_score(self, component_scores: Dict[str, float]) -> float:
        """Calculate weighted average of available component scores"""
        total_weight = 0.0
        weighted_sum = 0.0

        for key, weight in self.weights.items():
            score = component_scores.get(key)
            if score is not None:
                weighted_sum += score * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        # Normalize if not all components are available
        return weighted_sum / total_weight

    def _get_category(self, health_index: float) -> tuple:
        """Get category and color from health index"""
        for cat_name, cat_def in self.categories.items():
            if cat_def["min"] <= health_index <= cat_def["max"]:
                return cat_name.upper(), cat_def["color"]

        return "CRITICAL", "#e74c3c"

    def _get_risk_level(self, health_index: float) -> str:
        """Determine risk level from health index"""
        if health_index >= self.RISK_THRESHOLDS["LOW"]:
            return "LOW"
        elif health_index >= self.RISK_THRESHOLDS["MODERATE"]:
            return "MODERATE"
        elif health_index >= self.RISK_THRESHOLDS["HIGH"]:
            return "HIGH"
        else:
            return "CRITICAL"

    def _generate_recommendations(
        self, component_scores: Dict[str, float], health_index: float
    ) -> List[str]:
        """Generate recommendations based on component scores"""
        recommendations = []

        # Check each component
        if component_scores.get("dga") is not None:
            if component_scores["dga"] < 50:
                recommendations.append(
                    "Urgent: Address DGA issues - consider dissolved gas analysis and fault diagnosis"
                )
            elif component_scores["dga"] < 70:
                recommendations.append(
                    "Monitor DGA trends closely and investigate fault type"
                )

        if component_scores.get("oil_quality") is not None:
            if component_scores["oil_quality"] < 50:
                recommendations.append(
                    "Critical: Oil treatment or replacement recommended"
                )
            elif component_scores["oil_quality"] < 70:
                recommendations.append("Consider oil regeneration or drying")

        if component_scores.get("electrical") is not None:
            if component_scores["electrical"] < 50:
                recommendations.append(
                    "Urgent: Electrical testing and inspection required"
                )
            elif component_scores["electrical"] < 70:
                recommendations.append("Schedule detailed electrical diagnostics")

        if component_scores.get("age") is not None:
            if component_scores["age"] < 50:
                recommendations.append("Consider transformer replacement planning")
            elif component_scores["age"] < 70:
                recommendations.append(
                    "Include transformer in capital replacement planning"
                )

        if component_scores.get("loading") is not None:
            if component_scores["loading"] < 50:
                recommendations.append("Reduce loading to extend transformer life")
            elif component_scores["loading"] < 70:
                recommendations.append("Avoid sustained overload conditions")

        # Overall recommendations
        if health_index >= 85:
            recommendations.append("Continue routine maintenance and monitoring")
        elif health_index >= 70:
            recommendations.append("Increase monitoring frequency")
        elif health_index >= 50:
            recommendations.append("Develop maintenance action plan")
        else:
            recommendations.append(
                "Immediate attention required - prioritize maintenance"
            )

        return recommendations


def calculate_health_index(
    dga_gases: Optional[Dict[str, float]] = None,
    dielectric_strength: Optional[float] = None,
    moisture_content: Optional[float] = None,
    acidity: Optional[float] = None,
    interfacial_tension: Optional[float] = None,
    power_factor: Optional[float] = None,
    capacitance: Optional[float] = None,
    insulation_resistance: Optional[float] = None,
    age_years: Optional[float] = None,
    average_load_percent: Optional[float] = None,
    config_path: str = "config/health_index_weights.yaml",
) -> HealthIndexResult:
    """
    Convenience function to calculate composite health index

    Args:
        Various component parameters
        config_path: Path to configuration file

    Returns:
        HealthIndexResult with composite score
    """
    calculator = CompositeHealthIndex(config_path)
    return calculator.calculate(
        dga_gases=dga_gases,
        dielectric_strength=dielectric_strength,
        moisture_content=moisture_content,
        acidity=acidity,
        interfacial_tension=interfacial_tension,
        power_factor=power_factor,
        capacitance=capacitance,
        insulation_resistance=insulation_resistance,
        age_years=age_years,
        average_load_percent=average_load_percent,
    )
