"""
Failure Probability Calculator
Calculates probability of failure at 30/60/90 day horizons
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class FailureProbabilityResult:
    """
    Result of failure probability calculation

    Attributes:
        p_failure_30_days: Probability of failure within 30 days (0.0-1.0)
        p_failure_60_days: Probability of failure within 60 days (0.0-1.0)
        p_failure_90_days: Probability of failure within 90 days (0.0-1.0)
        risk_category: Risk category (LOW, MEDIUM, HIGH, CRITICAL)
        contributing_factors: List of contributing factors
        confidence: Confidence level (0.0-1.0)
    """

    p_failure_30_days: float
    p_failure_60_days: float
    p_failure_90_days: float
    risk_category: str
    contributing_factors: List[str] = field(default_factory=list)
    confidence: float = 0.75

    def __str__(self) -> str:
        return (
            f"FailureProbabilityResult(30d={self.p_failure_30_days:.3f}, "
            f"60d={self.p_failure_60_days:.3f}, 90d={self.p_failure_90_days:.3f}, "
            f"risk={self.risk_category})"
        )


class FailureProbability:
    """
    Failure Probability Calculator

    Calculates probability of failure at 30/60/90 day horizons based on:
    - Health index and trend
    - Fault type severity weighting
    - Historical failure rates
    """

    # Risk category thresholds
    RISK_THRESHOLDS = {
        "LOW": 0.05,  # < 5%
        "MEDIUM": 0.20,  # 5-20%
        "HIGH": 0.50,  # 20-50%
        "CRITICAL": 1.0,  # > 50%
    }

    # Fault type severity weights (increases failure probability)
    FAULT_SEVERITY_WEIGHTS = {
        "thermal_fault_low": 1.5,
        "thermal_fault_medium": 2.5,
        "thermal_fault_high": 4.0,
        "partial_discharge": 3.0,
        "arcing": 5.0,
        "overheating": 2.0,
        "hot_spot": 3.5,
        "moisture": 1.8,
        "oxygen": 1.3,
        "acid": 1.5,
        "unknown": 1.2,
    }

    # Historical failure rates by health index category (per year)
    HISTORICAL_FAILURE_RATES = {
        "excellent": 0.001,  # 0.1% per year
        "good": 0.005,  # 0.5% per year
        "fair": 0.02,  # 2% per year
        "poor": 0.08,  # 8% per year
        "critical": 0.25,  # 25% per year
    }

    # Trend severity multipliers
    TREND_SEVERITY = {
        "improving": 0.5,
        "stable": 1.0,
        "declining": 1.5,
        "rapid_decline": 3.0,
    }

    def __init__(
        self,
        base_failure_rate: float = 0.01,
        min_confidence: float = 0.3,
    ):
        """
        Initialize Failure Probability Calculator

        Args:
            base_failure_rate: Base annual failure rate (default: 1%)
            min_confidence: Minimum confidence level
        """
        self.base_failure_rate = base_failure_rate
        self.min_confidence = min_confidence

    def calculate_probability(
        self,
        health_index: float,
        health_index_trend: Optional[str] = None,
        fault_types: Optional[List[str]] = None,
        gas_rates: Optional[Dict[str, float]] = None,
        loading_percent: Optional[float] = None,
        hotspot_temp: Optional[float] = None,
        moisture_content: Optional[float] = None,
    ) -> FailureProbabilityResult:
        """
        Calculate failure probability at 30/60/90 day horizons

        Args:
            health_index: Current health index (0-100)
            health_index_trend: Trend direction (improving, stable, declining, rapid_decline)
            fault_types: List of detected fault types
            gas_rates: Dictionary of gas rates of change (ppm/day)
            loading_percent: Current loading percentage
            hotspot_temp: Hotspot temperature (°C)
            moisture_content: Moisture content in oil (%)

        Returns:
            FailureProbabilityResult with probabilities and risk category
        """
        contributing_factors = []

        # Get base failure probability from health index
        base_prob = self._get_base_failure_probability(health_index)
        contributing_factors.append(f"Base probability from HI={health_index:.1f}")

        # Apply trend adjustment
        if health_index_trend:
            trend_multiplier = self.TREND_SEVERITY.get(health_index_trend, 1.0)
            base_prob *= trend_multiplier
            contributing_factors.append(
                f"Trend: {health_index_trend} (x{trend_multiplier})"
            )

        # Apply fault type severity
        if fault_types:
            fault_multiplier = self._get_fault_severity_multiplier(fault_types)
            base_prob *= fault_multiplier
            contributing_factors.append(f"Fault types: {', '.join(fault_types)}")

        # Apply gas rate increases
        if gas_rates:
            gas_multiplier = self._get_gas_rate_multiplier(gas_rates)
            base_prob *= gas_multiplier
            contributing_factors.append(f"Gas rate multiplier: {gas_multiplier:.2f}")

        # Apply loading stress
        if loading_percent is not None:
            loading_factor = self._get_loading_factor(loading_percent)
            base_prob *= loading_factor
            if loading_factor > 1.0:
                contributing_factors.append(f"High loading: {loading_percent:.0f}%")

        # Apply hotspot temperature
        if hotspot_temp is not None:
            temp_factor = self._get_temperature_factor(hotspot_temp)
            base_prob *= temp_factor
            if temp_factor > 1.0:
                contributing_factors.append(f"High hotspot: {hotspot_temp:.0f}°C")

        # Apply moisture content
        if moisture_content is not None:
            moisture_factor = self._get_moisture_factor(moisture_content)
            base_prob *= moisture_factor
            if moisture_factor > 1.0:
                contributing_factors.append(f"High moisture: {moisture_content:.1f}%")

        # Convert annual probability to short-term (30, 60, 90 days)
        p_annual = min(base_prob, 0.99)  # Cap at 99%

        # Calculate probabilities for different horizons
        # Using exponential model: P(t) = 1 - e^(-λt)
        # Where λ = -ln(1-P) for annual rate
        if p_annual < 1.0:
            lambda_rate = -np.log(1 - p_annual)
        else:
            lambda_rate = 10.0  # Very high rate

        days_30 = 30 / 365.0
        days_60 = 60 / 365.0
        days_90 = 90 / 365.0

        p_30 = 1 - np.exp(-lambda_rate * days_30)
        p_60 = 1 - np.exp(-lambda_rate * days_60)
        p_90 = 1 - np.exp(-lambda_rate * days_90)

        # Determine risk category
        risk_category = self._get_risk_category(p_90)

        # Calculate confidence based on data availability
        confidence = self._calculate_confidence(health_index, fault_types, gas_rates)

        # Ensure minimum confidence
        confidence = max(confidence, self.min_confidence)

        return FailureProbabilityResult(
            p_failure_30_days=min(1.0, p_30),
            p_failure_60_days=min(1.0, p_60),
            p_failure_90_days=min(1.0, p_90),
            risk_category=risk_category,
            contributing_factors=contributing_factors,
            confidence=confidence,
        )

    def _get_base_failure_probability(self, health_index: float) -> float:
        """
        Get base annual failure probability from health index

        Args:
            health_index: Current health index

        Returns:
            Annual failure probability
        """
        # Use historical failure rates mapped to health index
        if health_index >= 85:
            return self.HISTORICAL_FAILURE_RATES["excellent"]
        elif health_index >= 70:
            return self.HISTORICAL_FAILURE_RATES["good"]
        elif health_index >= 50:
            return self.HISTORICAL_FAILURE_RATES["fair"]
        elif health_index >= 25:
            return self.HISTORICAL_FAILURE_RATES["poor"]
        else:
            return self.HISTORICAL_FAILURE_RATES["critical"]

    def _get_fault_severity_multiplier(self, fault_types: List[str]) -> float:
        """
        Get severity multiplier for fault types

        Args:
            fault_types: List of detected fault types

        Returns:
            Multiplier for failure probability
        """
        max_severity = 1.0

        for fault in fault_types:
            fault_lower = fault.lower()
            for key, value in self.FAULT_SEVERITY_WEIGHTS.items():
                if key in fault_lower:
                    max_severity = max(max_severity, value)
                    break

        return max_severity

    def _get_gas_rate_multiplier(self, gas_rates: Dict[str, float]) -> float:
        """
        Get multiplier based on gas rates of change

        Args:
            gas_rates: Dictionary of gas rates (ppm/day)

        Returns:
            Multiplier for failure probability
        """
        if not gas_rates:
            return 1.0

        # Key gases that indicate serious problems
        critical_gases = {
            "C2H2": 2.0,  # Acetylene - very serious
            "H2": 1.5,  # Hydrogen
            "C2H4": 1.3,  # Ethylene
            "CH4": 1.2,  # Methane
        }

        multiplier = 1.0

        for gas, rate in gas_rates.items():
            if gas in critical_gases and rate > 0:
                # Normalize rate - significant if > 10% of threshold per month
                gas_threshold = critical_gases.get(gas, 1.0)
                if rate > gas_threshold * 0.1:  # More than 10% of threshold per day
                    multiplier *= 1.2

        return min(multiplier, 5.0)  # Cap at 5x

    def _get_loading_factor(self, loading_percent: float) -> float:
        """
        Get loading stress factor

        Args:
            loading_percent: Loading percentage

        Returns:
            Stress factor
        """
        if loading_percent >= 100:
            return 2.5
        elif loading_percent >= 80:
            return 1.5
        elif loading_percent >= 60:
            return 1.1
        else:
            return 1.0

    def _get_temperature_factor(self, hotspot_temp: float) -> float:
        """
        Get temperature stress factor

        Args:
            hotspot_temp: Hotspot temperature in °C

        Returns:
            Stress factor
        """
        if hotspot_temp >= 140:
            return 3.0  # Severe overheating
        elif hotspot_temp >= 120:
            return 2.0  # High temperature
        elif hotspot_temp >= 100:
            return 1.3  # Moderate
        else:
            return 1.0

    def _get_moisture_factor(self, moisture_content: float) -> float:
        """
        Get moisture stress factor

        Args:
            moisture_content: Moisture content in %

        Returns:
            Stress factor
        """
        if moisture_content >= 0.5:
            return 2.0  # Very high moisture
        elif moisture_content >= 0.25:
            return 1.5  # High moisture
        elif moisture_content >= 0.1:
            return 1.1  # Moderate
        else:
            return 1.0

    def _calculate_confidence(
        self,
        health_index: float,
        fault_types: Optional[List[str]],
        gas_rates: Optional[Dict[str, float]],
    ) -> float:
        """
        Calculate confidence based on data availability

        Args:
            health_index: Health index value
            fault_types: Fault types detected
            gas_rates: Gas rates available

        Returns:
            Confidence level (0-1)
        """
        confidence = 0.5  # Base confidence

        # More confidence with fault types identified
        if fault_types:
            confidence += 0.15

        # More confidence with gas rate data
        if gas_rates:
            confidence += 0.15

        # Health index gives baseline confidence
        if health_index < 25 or health_index > 85:
            confidence += 0.1  # Clear indication
        else:
            confidence += 0.05

        return min(confidence, 0.95)

    def _get_risk_category(self, p_90: float) -> str:
        """
        Get risk category from 90-day failure probability

        Args:
            p_90: 90-day failure probability

        Returns:
            Risk category string
        """
        if p_90 < self.RISK_THRESHOLDS["LOW"]:
            return "LOW"
        elif p_90 < self.RISK_THRESHOLDS["MEDIUM"]:
            return "MEDIUM"
        elif p_90 < self.RISK_THRESHOLDS["HIGH"]:
            return "HIGH"
        else:
            return "CRITICAL"

    def get_recommendations(self, result: FailureProbabilityResult) -> List[str]:
        """
        Get recommendations based on failure probability

        Args:
            result: Failure probability result

        Returns:
            List of recommendations
        """
        recommendations = []

        if result.risk_category == "CRITICAL":
            recommendations.append("URGENT: Immediate inspection required")
            recommendations.append("Consider load reduction to minimize stress")
            recommendations.append(
                "Plan for immediate replacement or major maintenance"
            )
        elif result.risk_category == "HIGH":
            recommendations.append("Schedule detailed inspection within 30 days")
            recommendations.append("Monitor DGA trends weekly")
            recommendations.append("Evaluate loading conditions")
        elif result.risk_category == "MEDIUM":
            recommendations.append("Continue regular monitoring")
            recommendations.append("Review maintenance schedule")
        else:
            recommendations.append("Maintain routine monitoring")

        return recommendations
