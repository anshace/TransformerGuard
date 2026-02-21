"""
Remaining Useful Life (RUL) Estimator
Estimates RUL based on health index, degradation rate, and historical data
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

# IEEE C57.12.00 baseline
IEEE_LIFE_EXPECTANCY_HOURS = 180000  # 20.5 years
IEEE_LIFE_EXPECTANCY_YEARS = IEEE_LIFE_EXPECTANCY_HOURS / 8760


@dataclass
class RULResult:
    """
    Result of RUL estimation

    Attributes:
        rul_years: Remaining useful life in years
        rul_days: Remaining useful life in days
        confidence: Confidence level (0.0-1.0)
        method: Method used for estimation (linear, exponential, ml-based)
        end_of_life_date: Estimated end-of-life date
        assumptions: List of assumptions made in the calculation
    """

    rul_years: float
    rul_days: float
    confidence: float
    method: str
    end_of_life_date: Optional[datetime]
    assumptions: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"RULResult(years={self.rul_years:.2f}, days={self.rul_days:.0f}, "
            f"confidence={self.confidence:.2f}, method={self.method})"
        )


class RULEstimator:
    """
    Remaining Useful Life Estimator

    Estimates RUL based on:
    - Current health index
    - Rate of degradation
    - Historical failure data
    - IEEE C57.12.00 baseline
    """

    # Degradation rate thresholds (percent per year)
    DEGRADATION_RATE = {
        "slow": 1.0,  # < 1% per year
        "moderate": 2.5,  # 1-2.5% per year
        "fast": 5.0,  # 2.5-5% per year
        "rapid": 10.0,  # > 5% per year
    }

    # Confidence factors based on data availability
    CONFIDENCE_FACTORS = {
        "high": {"min_history_years": 3, "min_data_points": 24, "confidence": 0.85},
        "medium": {"min_history_years": 1, "min_data_points": 12, "confidence": 0.70},
        "low": {"min_history_years": 0, "min_data_points": 3, "confidence": 0.50},
    }

    def __init__(
        self,
        life_expectancy_hours: float = IEEE_LIFE_EXPECTANCY_HOURS,
        min_confidence: float = 0.3,
    ):
        """
        Initialize RUL Estimator

        Args:
            life_expectancy_hours: IEEE baseline life expectancy (default: 180000 hours)
            min_confidence: Minimum confidence level to return
        """
        self.life_expectancy_hours = life_expectancy_hours
        self.life_expectancy_years = life_expectancy_hours / 8760
        self.min_confidence = min_confidence

    def estimate_rul(
        self,
        current_health_index: float,
        age_years: float,
        degradation_rate: Optional[float] = None,
        health_index_history: Optional[List[float]] = None,
        health_index_dates: Optional[List[datetime]] = None,
        fault_types_present: Optional[List[str]] = None,
    ) -> RULResult:
        """
        Estimate remaining useful life

        Args:
            current_health_index: Current health index (0-100)
            age_years: Age of transformer in years
            degradation_rate: Optional degradation rate (% per year)
            health_index_history: Optional list of historical health indices
            health_index_dates: Optional dates for history (for calculating degradation)
            fault_types_present: Optional list of fault types detected

        Returns:
            RULResult with estimated remaining life
        """
        assumptions = []

        # Validate inputs
        if not 0 <= current_health_index <= 100:
            raise ValueError("Health index must be between 0 and 100")

        if age_years < 0:
            raise ValueError("Age cannot be negative")

        # Calculate degradation rate from history if not provided
        if degradation_rate is None and health_index_history is not None:
            degradation_rate = self._calculate_degradation_rate(
                health_index_history, health_index_dates
            )
            assumptions.append(
                f"Degradation rate calculated from {len(health_index_history)} data points"
            )
        elif degradation_rate is None:
            degradation_rate = self._estimate_degradation_from_hi(
                current_health_index, age_years
            )
            assumptions.append("Degradation rate estimated from health index and age")

        # Determine estimation method and calculate RUL
        if health_index_history is not None and len(health_index_history) >= 5:
            method = (
                "exponential"
                if self._has_exponential_degradation(health_index_history)
                else "linear"
            )
            rul_years, confidence = self._calculate_rul_with_history(
                current_health_index, degradation_rate, age_years, method
            )
        elif degradation_rate is not None:
            method = "linear"
            rul_years, confidence = self._calculate_rul_linear(
                current_health_index, degradation_rate, age_years
            )
        else:
            method = "ml-based"
            rul_years, confidence = self._calculate_rul_ml_based(
                current_health_index, age_years, fault_types_present
            )

        # Apply fault type penalties
        if fault_types_present:
            rul_years = self._apply_fault_penalties(rul_years, fault_types_present)
            assumptions.append(
                f"Fault penalties applied for: {', '.join(fault_types_present)}"
            )

        # Calculate end of life date
        end_of_life_date = datetime.now() + timedelta(days=int(rul_years * 365))

        # Ensure minimum confidence
        confidence = max(confidence, self.min_confidence)

        # Add method-specific assumptions
        if method == "linear":
            assumptions.append("Assuming linear degradation based on current HI")
        elif method == "exponential":
            assumptions.append("Assuming exponential degradation pattern")
        else:
            assumptions.append("Using ML-based estimation with limited historical data")

        return RULResult(
            rul_years=max(0, rul_years),
            rul_days=max(0, rul_years * 365),
            confidence=confidence,
            method=method,
            end_of_life_date=end_of_life_date if rul_years > 0 else None,
            assumptions=assumptions,
        )

    def _calculate_degradation_rate(
        self,
        health_index_history: List[float],
        health_index_dates: Optional[List[datetime]] = None,
    ) -> float:
        """
        Calculate degradation rate from historical data

        Args:
            health_index_history: List of historical health indices
            health_index_dates: Optional dates for the history

        Returns:
            Degradation rate in percent per year
        """
        if len(health_index_history) < 2:
            return 2.0  # Default moderate degradation

        # Use linear regression to find trend
        y = np.array(health_index_history)
        x = np.arange(len(y))

        # Fit linear regression
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]  # Change per data point

        # Convert to percentage per year (assuming monthly data)
        if health_index_dates and len(health_index_dates) >= 2:
            days_span = (health_index_dates[-1] - health_index_dates[0]).days
            if days_span > 0:
                points_per_year = len(y) / (days_span / 365)
                slope_per_year = slope * points_per_year
            else:
                slope_per_year = slope * 12  # Default monthly
        else:
            slope_per_year = slope * 12  # Assume monthly data

        # Convert to percentage
        avg_hi = np.mean(y)
        if avg_hi > 0:
            degradation_rate = abs(slope_per_year / avg_hi) * 100
        else:
            degradation_rate = 2.0

        return max(0.1, min(degradation_rate, 20.0))

    def _estimate_degradation_from_hi(
        self, current_health_index: float, age_years: float
    ) -> float:
        """
        Estimate degradation rate from health index and age

        Args:
            current_health_index: Current HI
            age_years: Transformer age

        Returns:
            Estimated degradation rate (% per year)
        """
        if age_years <= 0 or current_health_index >= 100:
            return 1.0  # Slow degradation for new/healthy transformers

        # Calculate implied degradation
        total_degradation = 100 - current_health_index
        implied_rate = total_degradation / age_years

        return max(0.5, min(implied_rate, 15.0))

    def _has_exponential_degradation(self, history: List[float]) -> bool:
        """
        Check if degradation pattern is exponential

        Args:
            history: Health index history

        Returns:
            True if pattern appears exponential
        """
        if len(history) < 5:
            return False

        # Check if degradation is accelerating
        y = np.array(history)
        x = np.arange(len(y))

        # Fit linear and exponential models
        try:
            linear_residuals = np.polyfit(x, y, 1)
            exp_log = np.log(y + 1)  # Add 1 to avoid log(0)
            exp_coeffs = np.polyfit(x, exp_log, 1)

            # If exponential coefficient is negative and significant
            return exp_coeffs[0] < -0.05
        except:
            return False

    def _calculate_rul_with_history(
        self,
        current_hi: float,
        degradation_rate: float,
        age_years: float,
        method: str,
    ) -> tuple:
        """
        Calculate RUL using historical data

        Args:
            current_hi: Current health index
            degradation_rate: Degradation rate (% per year)
            age_years: Current age
            method: 'linear' or 'exponential'

        Returns:
            Tuple of (rul_years, confidence)
        """
        if method == "exponential":
            # Exponential degradation model
            # HI = HI0 * e^(-k * t)
            # Solve for t when HI = 0
            k = degradation_rate / 100
            if k > 0:
                # Time to reach HI=0
                total_life = -np.log(current_hi / 100) / k
                rul = max(0, total_life - age_years)
            else:
                rul = self.life_expectancy_years - age_years
            confidence = 0.80
        else:
            rul, confidence = self._calculate_rul_linear(
                current_hi, degradation_rate, age_years
            )

        return rul, confidence

    def _calculate_rul_linear(
        self,
        current_hi: float,
        degradation_rate: float,
        age_years: float,
    ) -> tuple:
        """
        Calculate RUL using linear degradation model

        Args:
            current_hi: Current health index
            degradation_rate: Degradation rate (% per year)
            age_years: Current age

        Returns:
            Tuple of (rul_years, confidence)
        """
        # Linear: HI decreases at constant rate
        # Rate in absolute HI points per year
        rate_absolute = (
            degradation_rate  # % per year = points per year when HI is 100-scale
        )

        if rate_absolute <= 0:
            rul = self.life_expectancy_years - age_years
            return rul, 0.60

        # Years until HI reaches 0
        years_to_failure = current_hi / rate_absolute

        # RUL = min(calculated, IEEE baseline - age)
        rul = min(years_to_failure, self.life_expectancy_years - age_years)

        confidence = 0.75

        return max(0, rul), confidence

    def _calculate_rul_ml_based(
        self,
        current_hi: float,
        age_years: float,
        fault_types: Optional[List[str]] = None,
    ) -> tuple:
        """
        Estimate RUL using ML-based approach (simplified)

        Args:
            current_hi: Current health index
            age_years: Current age
            fault_types: Optional fault types

        Returns:
            Tuple of (rul_years, confidence)
        """
        # Simplified ML estimation using IEEE baseline and health index
        hi_factor = current_hi / 100
        age_factor = 1 - (age_years / self.life_expectancy_years)

        # Base RUL from IEEE
        base_rul = self.life_expectancy_years * hi_factor * max(0.3, age_factor)

        # Apply fault penalties
        if fault_types:
            fault_penalty = self._get_fault_penalty_factor(fault_types)
            base_rul *= fault_penalty

        confidence = 0.55  # Lower confidence for ML-based

        return max(0, base_rul), confidence

    def _apply_fault_penalties(self, rul_years: float, fault_types: List[str]) -> float:
        """
        Apply penalties for detected fault types

        Args:
            rul_years: Current RUL estimate
            fault_types: List of fault types

        Returns:
            Adjusted RUL
        """
        penalty_factor = self._get_fault_penalty_factor(fault_types)
        return rul_years * penalty_factor

    def _get_fault_penalty_factor(self, fault_types: List[str]) -> float:
        """
        Get penalty factor based on fault types

        Args:
            fault_types: List of detected fault types

        Returns:
            Penalty factor (0-1)
        """
        # Fault type severity multipliers
        fault_penalties = {
            "thermal_fault_low": 0.90,
            "thermal_fault_medium": 0.75,
            "thermal_fault_high": 0.60,
            "partial_discharge": 0.70,
            "arcing": 0.50,
            "overheating": 0.80,
            "unknown": 0.85,
        }

        penalty = 1.0
        for fault in fault_types:
            fault_lower = fault.lower()
            for key, value in fault_penalties.items():
                if key in fault_lower:
                    penalty = min(penalty, value)
                    break

        return max(0.1, penalty)

    def get_risk_category(self, rul_result: RULResult) -> str:
        """
        Get risk category based on RUL

        Args:
            rul_result: RUL estimation result

        Returns:
            Risk category string
        """
        if rul_result.rul_years <= 1:
            return "CRITICAL"
        elif rul_result.rul_years <= 3:
            return "HIGH"
        elif rul_result.rul_years <= 7:
            return "MEDIUM"
        else:
            return "LOW"
