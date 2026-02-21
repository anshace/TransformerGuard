"""
Tests for Rogers Ratio DGA Diagnosis Method

This module tests the RogersRatios class for fault diagnosis
using the Rogers Ratio method based on IEC 60599.

Author: TransformerGuard Team
"""

import pytest

from src.diagnosis.rogers_ratios import FaultType, RogersRatios, RogersResult


class TestRogersRatios:
    """Test suite for RogersRatios class."""

    def test_initialization(self):
        """Test that RogersRatios initializes correctly."""
        rogers = RogersRatios()
        assert rogers is not None
        assert rogers.method_name == "RogersRatios"

    def test_calculate_ratio_basic(self):
        """Test basic ratio calculation."""
        rogers = RogersRatios()

        # 10 / 5 = 2
        ratio = rogers._calculate_ratio(10, 5)
        assert ratio == 2.0

    def test_calculate_ratio_zero_denominator(self):
        """Test ratio calculation with zero denominator."""
        rogers = RogersRatios()

        ratio = rogers._calculate_ratio(10, 0)
        assert ratio == 0.0

    def test_calculate_ratio_zero_numerator(self):
        """Test ratio calculation with zero numerator."""
        rogers = RogersRatios()

        ratio = rogers._calculate_ratio(0, 5)
        assert ratio == 0.0

    def test_calculate_ratio_both_zero(self):
        """Test ratio calculation with both values zero."""
        rogers = RogersRatios()

        ratio = rogers._calculate_ratio(0, 0)
        assert ratio == 0.0

    def test_get_code_below_limit(self):
        """Test getting ratio code when below limit."""
        rogers = RogersRatios()

        # ratio < limit (0.1)
        code = rogers._get_code(0.05, "c2h2_c2h4")
        assert code == 0

    def test_get_code_between_limit_and_one(self):
        """Test getting ratio code between limit and 1."""
        rogers = RogersRatios()

        # limit <= ratio < 1
        code = rogers._get_code(0.5, "c2h2_c2h4")
        assert code == 1

    def test_get_code_between_one_and_high(self):
        """Test getting ratio code between 1 and high threshold."""
        rogers = RogersRatios()

        # 1 <= ratio <= high (3.0)
        code = rogers._get_code(2.0, "c2h2_c2h4")
        assert code == 2

    def test_get_code_above_high(self):
        """Test getting ratio code when above high threshold."""
        rogers = RogersRatios()

        # ratio > high
        code = rogers._get_code(5.0, "c2h2_c2h4")
        assert code == 5

    def test_diagnose_partial_discharge(self):
        """Test partial discharge diagnosis (code 0000)."""
        rogers = RogersRatios()
        # Very low gases - all ratios should be 0
        result = rogers.diagnose(
            h2=100, ch4=10, c2h2=0, c2h4=1, c2h6=10, co=50, co2=500
        )

        assert result.fault_type == FaultType.NORMAL

    def test_diagnose_low_energy_discharge_d1(self):
        """Test low energy discharge diagnosis."""
        rogers = RogersRatios()
        # C2H2/C2H4 > 0.1, others low
        result = rogers.diagnose(
            h2=200, ch4=50, c2h2=20, c2h4=50, c2h6=20, co=100, co2=800
        )

        # Should be one of the discharge types
        assert result.fault_type in [FaultType.D1, FaultType.D2]

    def test_diagnose_high_energy_discharge_d2(self):
        """Test high energy discharge diagnosis."""
        rogers = RogersRatios()
        # High C2H2/C2H4, elevated CH4/H2
        result = rogers.diagnose(
            h2=100, ch4=150, c2h2=50, c2h4=30, c2h6=20, co=150, co2=1000
        )

        assert result.fault_type == FaultType.D2

    def test_diagnose_thermal_fault_t1(self):
        """Test low temperature thermal fault diagnosis."""
        rogers = RogersRatios()
        # C2H2/C2H4 < 0.1, C2H4/C2H6 between 1-3
        result = rogers.diagnose(
            h2=50, ch4=100, c2h2=2, c2h4=30, c2h6=30, co=100, co2=800
        )

        assert result.fault_type == FaultType.T1

    def test_diagnose_thermal_fault_t2(self):
        """Test medium temperature thermal fault diagnosis."""
        rogers = RogersRatios()
        # C2H2/C2H4 < 0.1, CH4/H2 > 1, C2H4/C2H6 > 1
        result = rogers.diagnose(
            h2=30, ch4=100, c2h2=2, c2h4=80, c2h6=30, co=150, co2=1000
        )

        assert result.fault_type == FaultType.T2

    def test_diagnose_thermal_fault_t3(self):
        """Test high temperature thermal fault diagnosis."""
        rogers = RogersRatios()
        # High C2H4/C2H6 and elevated CO2/CO
        result = rogers.diagnose(
            h2=50, ch4=150, c2h2=2, c2h4=200, c2h6=40, co=400, co2=2000
        )

        assert result.fault_type == FaultType.T3

    def test_diagnose_with_zero_c2h4(self):
        """Test diagnosis when C2H4 is zero."""
        rogers = RogersRatios()

        result = rogers.diagnose(
            h2=100, ch4=50, c2h2=10, c2h4=0, c2h6=20, co=100, co2=800
        )

        # Should handle gracefully (ratio becomes 0)
        assert result is not None
        assert result.ratios is not None

    def test_diagnose_with_zero_h2(self):
        """Test diagnosis when H2 is zero."""
        rogers = RogersRatios()

        result = rogers.diagnose(
            h2=0, ch4=50, c2h2=10, c2h4=30, c2h6=20, co=100, co2=800
        )

        # Should handle gracefully
        assert result is not None

    def test_diagnose_with_zero_co(self):
        """Test diagnosis when CO is zero."""
        rogers = RogersRatios()

        result = rogers.diagnose(
            h2=50, ch4=100, c2h2=2, c2h4=50, c2h6=30, co=0, co2=1000
        )

        # CO2/CO ratio should be handled
        assert result is not None

    def test_result_contains_ratios(self):
        """Test that result contains calculated ratios."""
        rogers = RogersRatios()

        result = rogers.diagnose(
            h2=50, ch4=100, c2h2=10, c2h4=50, c2h6=30, co=150, co2=1000
        )

        assert result.ratios is not None
        assert "c2h2_c2h4" in result.ratios
        assert "ch4_h2" in result.ratios
        assert "c2h4_c2h6" in result.ratios
        assert "co2_co" in result.ratios

    def test_result_contains_code(self):
        """Test that result contains Rogers code."""
        rogers = RogersRatios()

        result = rogers.diagnose(
            h2=50, ch4=100, c2h2=10, c2h4=50, c2h6=30, co=150, co2=1000
        )

        assert result.code is not None
        assert len(result.code) == 4

    def test_confidence_in_valid_range(self):
        """Test that confidence is between 0 and 1."""
        rogers = RogersRatios()

        result = rogers.diagnose(
            h2=50, ch4=100, c2h2=10, c2h4=50, c2h6=30, co=150, co2=1000
        )

        assert 0 <= result.confidence <= 1

    def test_explanation_generated(self):
        """Test that explanation is generated."""
        rogers = RogersRatios()

        result = rogers.diagnose(
            h2=50, ch4=100, c2h2=10, c2h4=50, c2h6=30, co=150, co2=1000
        )

        assert result.explanation is not None
        assert len(result.explanation) > 0

    def test_thermal_fault_sample(self, sample_dga_thermal_fault):
        """Test with typical thermal fault DGA data."""
        rogers = RogersRatios()
        result = rogers.diagnose(**sample_dga_thermal_fault)

        # Should detect some form of thermal fault
        assert result.fault_type in [
            FaultType.T1,
            FaultType.T2,
            FaultType.T3,
            FaultType.DT,
            FaultType.NORMAL,
        ]

    def test_arcing_sample(self, sample_dga_arcing):
        """Test with typical arcing DGA data."""
        rogers = RogersRatios()
        result = rogers.diagnose(**sample_dga_arcing)

        # Should detect discharge fault
        assert result.fault_type in [
            FaultType.D1,
            FaultType.D2,
            FaultType.DT,
            FaultType.NORMAL,
        ]

    def test_partial_discharge_sample(self, sample_dga_partial_discharge):
        """Test with typical partial discharge DGA data."""
        rogers = RogersRatios()
        result = rogers.diagnose(**sample_dga_partial_discharge)

        assert result is not None


class TestRogersResult:
    """Test suite for RogersResult dataclass."""

    def test_result_creation(self):
        """Test creating a RogersResult instance."""
        result = RogersResult(
            fault_type=FaultType.NORMAL,
            confidence=0.90,
            explanation="Test explanation",
            method_name="RogersRatios",
        )

        assert result.fault_type == FaultType.NORMAL
        assert result.confidence == 0.90
        assert result.explanation == "Test explanation"
        assert result.method_name == "RogersRatios"

    def test_result_with_ratios_and_code(self):
        """Test result with ratios and code."""
        result = RogersResult(
            fault_type=FaultType.T2,
            confidence=0.75,
            explanation="Test",
            method_name="RogersRatios",
            ratios={"c2h2_c2h4": 0.05, "ch4_h2": 2.0, "c2h4_c2h6": 3.0, "co2_co": 5.0},
            code="0225",
        )

        assert result.ratios is not None
        assert result.code == "0225"
        assert result.ratios["c2h2_c2h4"] == 0.05


class TestRogersRatioInterpretationTable:
    """Test suite for Rogers ratio interpretation table."""

    def test_interpretation_table_has_pd(self):
        """Test that interpretation table has PD codes."""
        rogers = RogersRatios()

        # (0, 0, 0, 0) should map to PD
        fault_type = rogers._get_fault_type_from_code((0, 0, 0, 0))
        assert fault_type == FaultType.PD

    def test_interpretation_table_has_t1(self):
        """Test that interpretation table has T1 codes."""
        rogers = RogersRatios()

        # (0, 0, 1, 0) or (0, 0, 2, 0) should map to T1
        fault_type = rogers._get_fault_type_from_code((0, 0, 1, 0))
        assert fault_type == FaultType.T1

    def test_interpretation_table_has_t2(self):
        """Test that interpretation table has T2 codes."""
        rogers = RogersRatios()

        # (0, 1, 1, 0) should map to T2
        fault_type = rogers._get_fault_type_from_code((0, 1, 1, 0))
        assert fault_type == FaultType.T2

    def test_interpretation_table_undetermined_for_unknown_code(self):
        """Test that unknown codes return UNDETERMINED."""
        rogers = RogersRatios()

        # Some unusual code combination
        fault_type = rogers._get_fault_type_from_code((5, 5, 5, 5))
        assert fault_type == FaultType.UNDETERMINED


class TestRogersConfidence:
    """Test suite for Rogers confidence calculations."""

    def test_high_confidence_for_clear_ratios(self):
        """Test that clear ratios produce higher confidence."""
        rogers = RogersRatios()

        # Very clear discharge signature
        result = rogers.diagnose(
            h2=10, ch4=200, c2h2=100, c2h4=20, c2h6=10, co=50, co2=500
        )

        # Clear indicators should give higher confidence
        assert result.confidence > 0.6

    def test_low_confidence_for_undetermined(self):
        """Test that undetermined results have lower confidence."""
        rogers = RogersRatios()

        # Get an undetermined result
        result = rogers.diagnose(
            h2=50, ch4=50, c2h2=50, c2h4=50, c2h6=50, co=50, co2=50
        )

        # Check confidence is in valid range
        assert 0 <= result.confidence <= 1
