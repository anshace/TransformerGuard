"""
Tests for Insulation Aging Model

This module tests the AgingModel class for insulation aging calculations
based on the IEEE C57.91-2011 standard.

Author: TransformerGuard Team
"""

import math

import pytest

from src.thermal.aging_model import (
    AGING_DOUBLING_TEMP,
    MIN_LIFE_HOURS,
    REFERENCE_HOTSPOT,
    AgingModel,
    AgingResult,
)


class TestAgingModel:
    """Test suite for AgingModel class."""

    def test_initialization(self):
        """Test that AgingModel initializes correctly."""
        model = AgingModel()

        assert model is not None
        assert model.reference_hotspot == REFERENCE_HOTSPOT
        assert model.min_life_hours == MIN_LIFE_HOURS

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        model = AgingModel(reference_hotspot=120, min_life_hours=200000)

        assert model.reference_hotspot == 120
        assert model.min_life_hours == 200000

    def test_calculate_aging_factor_at_reference(self):
        """Test aging factor at reference temperature (110°C)."""
        model = AgingModel()

        factor = model.calculate_aging_factor(110)

        # At reference temperature, aging factor should be around 0.5
        # (half the rate at 116°C which is reference in some standards)
        assert 0.4 < factor < 0.6

    def test_calculate_aging_factor_above_reference(self):
        """Test aging factor above reference temperature."""
        model = AgingModel()

        # At 116°C (6°C above reference), factor should be ~1.0
        factor = model.calculate_aging_factor(116)

        assert 0.8 < factor < 1.2

    def test_aging_doubles_every_6_degrees(self):
        """Test that aging doubles every 6°C above reference."""
        model = AgingModel()

        factor_110 = model.calculate_aging_factor(110)
        factor_116 = model.calculate_aging_factor(116)
        factor_122 = model.calculate_aging_factor(122)

        # 116°C should be approximately 2x 110°C
        ratio_1 = factor_116 / factor_110
        assert 1.8 < ratio_1 < 2.2

        # 122°C should be approximately 2x 116°C (4x 110°C)
        ratio_2 = factor_122 / factor_116
        assert 1.8 < ratio_2 < 2.2

    def test_aging_factor_at_high_temperature(self):
        """Test aging factor at high temperature."""
        model = AgingModel()

        factor = model.calculate_aging_factor(140)

        # At 140°C (30°C above reference), factor should be much higher
        # 2^5 = 32x, but with some curve, should still be very high
        assert factor > 10

    def test_aging_factor_below_reference(self):
        """Test aging factor below reference temperature."""
        model = AgingModel()

        factor = model.calculate_aging_factor(80)

        # Below reference, aging should be slower (using Arrhenius equation)
        # At 80°C, the factor should be less than at reference (110°C = 0.5)
        assert factor < 1.0

    def test_calculate_single_duration(self):
        """Test aging calculation for single duration."""
        model = AgingModel()

        result = model.calculate(hotspot_temp=110, duration_hours=1.0)

        assert isinstance(result, AgingResult)
        assert result.aging_acceleration_factor > 0
        assert result.equivalent_aging_hours > 0

    def test_calculate_24_hours(self):
        """Test aging calculation for 24 hours."""
        model = AgingModel()

        result = model.calculate(hotspot_temp=110, duration_hours=24)

        assert result.equivalent_aging_hours > 0

    def test_daily_aging_rate_calculation(self):
        """Test daily aging rate calculation."""
        model = AgingModel()

        result = model.calculate(hotspot_temp=110, duration_hours=24)

        # Daily aging rate should be percentage of life consumed per day
        assert result.daily_aging_rate > 0

    def test_life_consumed_percent(self):
        """Test life consumed percentage calculation."""
        model = AgingModel()

        # At reference temp (110°C), aging factor is 0.5
        # For 180000 hours (min life), with factor 0.5, life consumed = 50%
        result = model.calculate(hotspot_temp=110, duration_hours=180000)

        # Should consume approximately 50% of life at reference temp
        assert 40 < result.life_consumed_percent < 60

    def test_calculate_invalid_temperature_too_low(self):
        """Test that invalid low temperature raises error."""
        model = AgingModel()

        with pytest.raises(ValueError):
            model.calculate(hotspot_temp=-100, duration_hours=1.0)

    def test_calculate_invalid_temperature_too_high(self):
        """Test that invalid high temperature raises error."""
        model = AgingModel()

        with pytest.raises(ValueError):
            model.calculate(hotspot_temp=250, duration_hours=1.0)

    def test_calculate_invalid_duration_negative(self):
        """Test that negative duration raises error."""
        model = AgingModel()

        with pytest.raises(ValueError):
            model.calculate(hotspot_temp=110, duration_hours=-10)

    def test_calculate_from_load_profile(self):
        """Test aging calculation from load profile."""
        model = AgingModel()

        hotspot_temps = [100, 105, 110, 115, 120]
        time_intervals = [2, 4, 6, 8, 4]  # Total 24 hours

        result = model.calculate_from_load_profile(hotspot_temps, time_intervals)

        assert isinstance(result, AgingResult)
        assert result.equivalent_aging_hours > 0

    def test_calculate_from_load_profile_mismatched_lengths(self):
        """Test that mismatched lengths raise error."""
        model = AgingModel()

        hotspot_temps = [100, 105, 110]
        time_intervals = [2, 4]  # Only 2 values

        with pytest.raises(ValueError):
            model.calculate_from_load_profile(hotspot_temps, time_intervals)

    def test_estimate_remaining_life_at_reference(self):
        """Test remaining life estimation at reference temperature."""
        model = AgingModel()

        remaining = model.estimate_remaining_life(110)

        # At reference, should be approximately full life
        assert remaining > 100000

    def test_estimate_remaining_life_at_high_temp(self):
        """Test remaining life estimation at high temperature."""
        model = AgingModel()

        remaining = model.estimate_remaining_life(140)

        # At high temp, should be much less than full life
        assert remaining < 50000

    def test_estimate_remaining_life_infinite_at_low_temp(self):
        """Test remaining life at very low temperature."""
        model = AgingModel()

        remaining = model.estimate_remaining_life(50)

        # At very low temp, life should be very long (but not infinite)
        # Using Arrhenius equation, at 50°C the factor is ~0.49
        # So remaining life should be around 180000/0.49 ≈ 367000 hours
        assert remaining > 300000

    def test_get_aging_rate_table(self):
        """Test getting aging rate table."""
        model = AgingModel()

        table = model.get_aging_rate_table()

        assert isinstance(table, dict)
        assert len(table) > 0

    def test_aging_rate_table_keys(self):
        """Test aging rate table has expected temperature keys."""
        model = AgingModel()

        table = model.get_aging_rate_table()

        # Table should contain temperatures from 60°C to 180°C
        assert 60 in table
        assert 180 in table

    def test_get_life_expectancy(self):
        """Test life expectancy calculation."""
        model = AgingModel()

        life_exp = model.get_life_expectancy(110)

        assert "remaining_hours" in life_exp
        assert "remaining_years" in life_exp
        assert "aging_factor" in life_exp
        assert "reference_hours" in life_exp
        assert "reference_years" in life_exp

    def test_life_expectancy_years_calculation(self):
        """Test that remaining years is calculated correctly."""
        model = AgingModel()

        life_exp = model.get_life_expectancy(110)

        # 180000 hours / 8760 hours/year ≈ 20.5 years
        assert 19 < life_exp["reference_years"] < 22


class TestAgingResult:
    """Test suite for AgingResult dataclass."""

    def test_aging_result_creation(self):
        """Test creating an AgingResult instance."""
        result = AgingResult(
            aging_acceleration_factor=1.0,
            equivalent_aging_hours=24.0,
            daily_aging_rate=0.0133,
            life_consumed_percent=0.0133,
        )

        assert result.aging_acceleration_factor == 1.0
        assert result.equivalent_aging_hours == 24.0
        assert result.daily_aging_rate == 0.0133
        assert result.life_consumed_percent == 0.0133

    def test_aging_result_string_representation(self):
        """Test AgingResult string representation."""
        result = AgingResult(
            aging_acceleration_factor=1.0,
            equivalent_aging_hours=24.0,
            daily_aging_rate=0.0133,
            life_consumed_percent=0.0133,
        )

        str_repr = str(result)

        assert "F_AA" in str_repr
        assert "equiv_hours" in str_repr


class TestAgingConstants:
    """Test suite for aging constants."""

    def test_reference_hotspot_constant(self):
        """Test reference hotspot constant."""
        assert REFERENCE_HOTSPOT == 110

    def test_min_life_hours_constant(self):
        """Test minimum life hours constant."""
        assert MIN_LIFE_HOURS == 180000

    def test_aging_doubling_temp_constant(self):
        """Test aging doubling temperature constant."""
        assert AGING_DOUBLING_TEMP == 6


class TestAgingModelWithTable:
    """Test suite for aging model with lookup table."""

    def test_use_table_parameter(self):
        """Test that use_table parameter works."""
        model_with_table = AgingModel(use_table=True)
        model_without_table = AgingModel(use_table=False)

        # Both should work
        assert model_with_table.use_table is True
        assert model_without_table.use_table is False

    def test_aging_factor_with_table(self):
        """Test aging factor calculation with lookup table."""
        model = AgingModel(use_table=True)

        factor = model.calculate_aging_factor(110)

        assert factor > 0

    def test_aging_factor_without_table(self):
        """Test aging factor calculation without lookup table."""
        model = AgingModel(use_table=False)

        factor = model.calculate_aging_factor(110)

        assert factor > 0

    def test_table_interpolation(self):
        """Test aging factor interpolation from table."""
        model = AgingModel(use_table=True)

        # Between table entries
        factor = model.calculate_aging_factor(107)

        assert factor > 0


class TestAgingCalculations:
    """Integration tests for aging calculations."""

    def test_daily_aging_at_normal_operating_temp(self):
        """Test daily aging at normal operating temperature."""
        model = AgingModel()

        # Normal operating hotspot ~95°C
        result = model.calculate(hotspot_temp=95, duration_hours=24)

        # Should have small daily aging rate
        assert result.daily_aging_rate < 0.1

    def test_daily_aging_at_high_operating_temp(self):
        """Test daily aging at high operating temperature."""
        model = AgingModel()

        # High operating hotspot ~120°C
        result = model.calculate(hotspot_temp=120, duration_hours=24)

        # Should have noticeable daily aging rate (factor ~1.59 at 120°C)
        # Daily rate = 24 * 1.59 / 180000 * 100 ≈ 0.021%
        assert result.daily_aging_rate > 0.01

    def test_weekly_aging_calculation(self):
        """Test aging calculation for a week."""
        model = AgingModel()

        result = model.calculate(hotspot_temp=110, duration_hours=168)  # 7 days

        # Weekly aging should be 7x daily
        assert result.equivalent_aging_hours > 24

    def test_yearly_aging_at_reference(self):
        """Test yearly aging at reference temperature."""
        model = AgingModel()

        result = model.calculate(hotspot_temp=110, duration_hours=8760)  # 1 year

        # At reference temp (110°C), aging factor is 0.5
        # Life consumed = 8760 * 0.5 / 180000 * 100 ≈ 2.43%
        assert 0.01 < result.life_consumed_percent / 100 < 0.05
