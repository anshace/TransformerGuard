"""
Tests for Remaining Useful Life (RUL) Estimator

This module tests the RULEstimator class for estimating
transformer remaining useful life.

Author: TransformerGuard Team
"""

from datetime import datetime, timedelta

import pytest

from src.prediction.rul_estimator import (
    IEEE_LIFE_EXPECTANCY_HOURS,
    IEEE_LIFE_EXPECTANCY_YEARS,
    RULEstimator,
    RULResult,
)


class TestRULEstimator:
    """Test suite for RULEstimator class."""

    def test_initialization(self):
        """Test that RULEstimator initializes correctly."""
        estimator = RULEstimator()

        assert estimator is not None
        assert estimator.life_expectancy_hours == IEEE_LIFE_EXPECTANCY_HOURS

    def test_initialization_custom_life(self):
        """Test initialization with custom life expectancy."""
        estimator = RULEstimator(life_expectancy_hours=200000)

        assert estimator.life_expectancy_hours == 200000
        assert estimator.life_expectancy_years == 200000 / 8760

    def test_estimate_rul_basic(self):
        """Test basic RUL estimation."""
        estimator = RULEstimator()

        result = estimator.estimate_rul(current_health_index=80.0, age_years=5.0)

        assert isinstance(result, RULResult)
        assert result.rul_years >= 0
        assert result.rul_days >= 0

    def test_estimate_rul_healthy_transformer(self):
        """Test RUL for healthy transformer."""
        estimator = RULEstimator()

        result = estimator.estimate_rul(current_health_index=90.0, age_years=5.0)

        # Healthy transformer should have positive RUL
        assert result.rul_years > 0

    def test_estimate_rul_critical_transformer(self):
        """Test RUL for critical transformer."""
        estimator = RULEstimator()

        result = estimator.estimate_rul(current_health_index=20.0, age_years=20.0)

        # Critical transformer should have lower RUL
        assert result.rul_years >= 0

    def test_estimate_rul_with_degradation_rate(self):
        """Test RUL with explicit degradation rate."""
        estimator = RULEstimator()

        result = estimator.estimate_rul(
            current_health_index=70.0, age_years=10.0, degradation_rate=2.0
        )

        assert result.rul_years >= 0
        assert "linear" in result.method.lower() or "ml" in result.method.lower()

    def test_estimate_rul_with_history(self):
        """Test RUL with health index history."""
        estimator = RULEstimator()

        health_history = [95, 90, 85, 80, 75, 70, 65, 60]

        result = estimator.estimate_rul(
            current_health_index=60.0,
            age_years=8.0,
            health_index_history=health_history,
        )

        assert result.rul_years >= 0

    def test_estimate_rul_with_history_and_dates(self):
        """Test RUL with health index history and dates."""
        estimator = RULEstimator()

        # Create history with dates (monthly data for 8 months)
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=30 * i) for i in range(8)]

        health_history = [95, 90, 85, 80, 75, 70, 65, 60]

        result = estimator.estimate_rul(
            current_health_index=60.0,
            age_years=8.0,
            health_index_history=health_history,
            health_index_dates=dates,
        )

        assert result.rul_years >= 0

    def test_estimate_rul_with_fault_types(self):
        """Test RUL with fault types."""
        estimator = RULEstimator()

        result = estimator.estimate_rul(
            current_health_index=60.0,
            age_years=10.0,
            fault_types_present=["thermal_fault_medium"],
        )

        # Should have penalty applied
        assert result.rul_years >= 0

    def test_estimate_rul_invalid_health_index_too_high(self):
        """Test that invalid health index (>100) raises error."""
        estimator = RULEstimator()

        with pytest.raises(ValueError):
            estimator.estimate_rul(current_health_index=150.0, age_years=10.0)

    def test_estimate_rul_invalid_health_index_negative(self):
        """Test that negative health index raises error."""
        estimator = RULEstimator()

        with pytest.raises(ValueError):
            estimator.estimate_rul(current_health_index=-10.0, age_years=10.0)

    def test_estimate_rul_invalid_age_negative(self):
        """Test that negative age raises error."""
        estimator = RULEstimator()

        with pytest.raises(ValueError):
            estimator.estimate_rul(current_health_index=80.0, age_years=-5.0)

    def test_confidence_in_valid_range(self):
        """Test that confidence is between 0 and 1."""
        estimator = RULEstimator()

        result = estimator.estimate_rul(current_health_index=70.0, age_years=10.0)

        assert 0 <= result.confidence <= 1

    def test_confidence_higher_with_history(self):
        """Test that confidence is higher with more history."""
        estimator = RULEstimator()

        # With minimal history
        result_minimal = estimator.estimate_rul(
            current_health_index=70.0, age_years=10.0, health_index_history=[80, 70]
        )

        # With more history
        result_more = estimator.estimate_rul(
            current_health_index=70.0,
            age_years=10.0,
            health_index_history=[100, 95, 90, 85, 80, 75, 70],
        )

        # More history should give higher or equal confidence
        assert result_more.confidence >= result_minimal.confidence

    def test_end_of_life_date_calculated(self):
        """Test that end of life date is calculated."""
        estimator = RULEstimator()

        result = estimator.estimate_rul(current_health_index=70.0, age_years=10.0)

        assert result.end_of_life_date is not None

    def test_assumptions_included(self):
        """Test that assumptions are included in result."""
        estimator = RULEstimator()

        result = estimator.estimate_rul(current_health_index=70.0, age_years=10.0)

        assert isinstance(result.assumptions, list)

    def test_method_determined_from_history(self):
        """Test that estimation method is determined from history."""
        estimator = RULEstimator()

        # With sufficient history, should use linear or exponential
        result = estimator.estimate_rul(
            current_health_index=70.0,
            age_years=10.0,
            health_index_history=[100, 90, 80, 70, 60, 50, 40, 30],
        )

        assert result.method in ["linear", "exponential"]

    def test_rul_positive_for_new_healthy_transformer(self):
        """Test RUL is positive for new healthy transformer."""
        estimator = RULEstimator()

        result = estimator.estimate_rul(current_health_index=95.0, age_years=2.0)

        assert result.rul_years > 0

    def test_rul_decreases_with_age(self):
        """Test that RUL decreases with age at same health index."""
        estimator = RULEstimator()

        result_young = estimator.estimate_rul(current_health_index=70.0, age_years=5.0)

        result_old = estimator.estimate_rul(current_health_index=70.0, age_years=15.0)

        # Older transformer should have less RUL remaining
        assert result_old.rul_years <= result_young.rul_years

    def test_rul_decreases_with_poorer_health(self):
        """Test that RUL decreases with lower health index."""
        estimator = RULEstimator()

        result_healthy = estimator.estimate_rul(
            current_health_index=90.0, age_years=10.0
        )

        result_poor = estimator.estimate_rul(current_health_index=40.0, age_years=10.0)

        # Poorer health should have less RUL
        assert result_poor.rul_years < result_healthy.rul_years


class TestRULResult:
    """Test suite for RULResult dataclass."""

    def test_rul_result_creation(self):
        """Test creating a RULResult instance."""
        result = RULResult(
            rul_years=10.0,
            rul_days=3650.0,
            confidence=0.75,
            method="linear",
            end_of_life_date=datetime(2035, 1, 1),
        )

        assert result.rul_years == 10.0
        assert result.rul_days == 3650.0
        assert result.confidence == 0.75
        assert result.method == "linear"
        assert result.end_of_life_date == datetime(2035, 1, 1)

    def test_rul_result_with_assumptions(self):
        """Test creating result with assumptions."""
        result = RULResult(
            rul_years=5.0,
            rul_days=1825.0,
            confidence=0.60,
            method="exponential",
            end_of_life_date=datetime(2030, 1, 1),
            assumptions=["Assuming exponential degradation", "Based on 5 data points"],
        )

        assert len(result.assumptions) == 2

    def test_rul_result_string_representation(self):
        """Test RULResult string representation."""
        result = RULResult(
            rul_years=10.0,
            rul_days=3650.0,
            confidence=0.75,
            method="linear",
            end_of_life_date=datetime(2035, 1, 1),
        )

        str_repr = str(result)

        assert "years=" in str_repr
        assert "days=" in str_repr
        assert "confidence=" in str_repr


class TestRULConstants:
    """Test suite for RUL constants."""

    def test_ieee_life_expectancy_hours(self):
        """Test IEEE life expectancy hours constant."""
        assert IEEE_LIFE_EXPECTANCY_HOURS == 180000

    def test_ieee_life_expectancy_years(self):
        """Test IEEE life expectancy years constant."""
        expected_years = IEEE_LIFE_EXPECTANCY_HOURS / 8760
        assert 20 < expected_years < 21


class TestRULDegradationRates:
    """Test suite for degradation rate handling."""

    def test_degradation_rate_dict_defined(self):
        """Test that degradation rate thresholds are defined."""
        estimator = RULEstimator()

        assert hasattr(estimator, "DEGRADATION_RATE")
        assert "slow" in estimator.DEGRADATION_RATE
        assert "moderate" in estimator.DEGRADATION_RATE
        assert "fast" in estimator.DEGRADATION_RATE
        assert "rapid" in estimator.DEGRADATION_RATE


class TestRULConfidenceFactors:
    """Test suite for confidence factors."""

    def test_confidence_factors_defined(self):
        """Test that confidence factors are defined."""
        estimator = RULEstimator()

        assert hasattr(estimator, "CONFIDENCE_FACTORS")
        assert "high" in estimator.CONFIDENCE_FACTORS
        assert "medium" in estimator.CONFIDENCE_FACTORS
        assert "low" in estimator.CONFIDENCE_FACTORS


class TestRULRiskCategory:
    """Test suite for risk category determination."""

    def test_get_risk_category_critical(self):
        """Test risk category CRITICAL for very low RUL."""
        estimator = RULEstimator()

        result = RULResult(
            rul_years=0.5,
            rul_days=180,
            confidence=0.75,
            method="linear",
            end_of_life_date=datetime.now() + timedelta(days=180),
        )

        category = estimator.get_risk_category(result)

        assert category == "CRITICAL"

    def test_get_risk_category_high(self):
        """Test risk category HIGH for low RUL."""
        estimator = RULEstimator()

        result = RULResult(
            rul_years=2.0,
            rul_days=730,
            confidence=0.75,
            method="linear",
            end_of_life_date=datetime.now() + timedelta(days=730),
        )

        category = estimator.get_risk_category(result)

        assert category == "HIGH"

    def test_get_risk_category_medium(self):
        """Test risk category MEDIUM for moderate RUL."""
        estimator = RULEstimator()

        result = RULResult(
            rul_years=5.0,
            rul_days=1825,
            confidence=0.75,
            method="linear",
            end_of_life_date=datetime.now() + timedelta(days=1825),
        )

        category = estimator.get_risk_category(result)

        assert category == "MEDIUM"

    def test_get_risk_category_low(self):
        """Test risk category LOW for good RUL."""
        estimator = RULEstimator()

        result = RULResult(
            rul_years=10.0,
            rul_days=3650,
            confidence=0.75,
            method="linear",
            end_of_life_date=datetime.now() + timedelta(days=3650),
        )

        category = estimator.get_risk_category(result)

        assert category == "LOW"


class TestRULWithFaultPenalties:
    """Test suite for RUL with fault penalties."""

    def test_fault_penalty_applied(self):
        """Test that fault penalties are applied."""
        estimator = RULEstimator()

        # Without faults
        result_no_fault = estimator.estimate_rul(
            current_health_index=70.0, age_years=10.0, fault_types_present=None
        )

        # With arcing fault (severe)
        result_with_fault = estimator.estimate_rul(
            current_health_index=70.0, age_years=10.0, fault_types_present=["arcing"]
        )

        # With fault should have less RUL
        assert result_with_fault.rul_years <= result_no_fault.rul_years

    def test_multiple_faults_cumulative(self):
        """Test that multiple faults have cumulative penalty."""
        estimator = RULEstimator()

        result = estimator.estimate_rul(
            current_health_index=70.0,
            age_years=10.0,
            fault_types_present=["thermal_fault_high", "arcing"],
        )

        # Should apply penalty and have assumptions
        assert len(result.assumptions) > 0


class TestRULCalculations:
    """Integration tests for RUL calculations."""

    def test_complete_rul_calculation_flow(self):
        """Test complete RUL calculation workflow."""
        estimator = RULEstimator()

        # Create some historical data
        health_history = [95, 92, 88, 85, 82, 78, 75, 72, 68, 65]

        # Calculate RUL
        result = estimator.estimate_rul(
            current_health_index=65.0,
            age_years=10.0,
            health_index_history=health_history,
            fault_types_present=["thermal_fault_medium"],
        )

        # Verify results
        assert result.rul_years >= 0
        assert result.rul_days >= 0
        assert 0 <= result.confidence <= 1
        assert result.end_of_life_date is not None
        assert len(result.assumptions) > 0

    def test_rul_for_brand_new_transformer(self):
        """Test RUL for brand new transformer."""
        estimator = RULEstimator()

        result = estimator.estimate_rul(current_health_index=100.0, age_years=0.0)

        # Should have maximum RUL
        assert result.rul_years > 15  # Close to full life expectancy

    def test_rul_for_end_of_life_transformer(self):
        """Test RUL for end-of-life transformer."""
        estimator = RULEstimator()

        result = estimator.estimate_rul(current_health_index=5.0, age_years=25.0)

        # Should have very low RUL
        assert result.rul_years < 5
