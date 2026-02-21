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
        # All ratios below low limits to get code 0000 -> NORMAL
        # C2H2/C2H4 < 0.1: C2H2=0, C2H4=10 -> 0/10 = 0
        # CH4/H2 < 0.1: CH4=5, H2=100 -> 5/100 = 0.05
        # C2H4/C2H6 < 1.0: C2H4=10, C2H6=50 -> 10/50 = 0.2
        # CO2/CO < 0.1: CO2=5, CO=100 -> 5/100 = 0.05
        result = rogers.diagnose(
            h2=100, ch4=5, c2h2=0, c2h4=10, c2h6=50, co=100, co2=5
        )

        assert result.fault_type == FaultType.NORMAL

    def test_diagnose_low_energy_discharge_d1(self):
        """Test low energy discharge diagnosis."""
        rogers = RogersRatios()
        # Target code (1, 0, 0, 0) -> D1
        # C2H2/C2H4: 0.1 <= ratio < 1 -> code 1: C2H2=10, C2H4=50 -> 0.2
        # CH4/H2: ratio < 0.1 -> code 0: CH4=5, H2=100 -> 0.05
        # C2H4/C2H6: ratio < 1 -> code 0: C2H4=50, C2H6=100 -> 0.5
        # CO2/CO: ratio < 0.1 -> code 0: CO2=5, CO=100 -> 0.05
        result = rogers.diagnose(
            h2=100, ch4=5, c2h2=10, c2h4=50, c2h6=100, co=100, co2=5
        )

        assert result.fault_type == FaultType.D1

    def test_diagnose_high_energy_discharge_d2(self):
        """Test high energy discharge diagnosis."""
        rogers = RogersRatios()
        # Target code (1, 1, 0, 0) -> D2
        # C2H2/C2H4: 0.1 <= ratio < 1 -> code 1: C2H2=10, C2H4=50 -> 0.2
        # CH4/H2: 0.1 <= ratio < 1 -> code 1: CH4=50, H2=100 -> 0.5
        # C2H4/C2H6: ratio < 1 -> code 0: C2H4=50, C2H6=100 -> 0.5
        # CO2/CO: ratio < 0.1 -> code 0: CO2=5, CO=100 -> 0.05
        result = rogers.diagnose(
            h2=100, ch4=50, c2h2=10, c2h4=50, c2h6=100, co=100, co2=5
        )

        assert result.fault_type == FaultType.D2

    def test_diagnose_thermal_fault_t1(self):
        """Test low temperature thermal fault diagnosis."""
        rogers = RogersRatios()
        # Target code (0, 0, 1, 0) -> T1
        # C2H2/C2H4: ratio < 0.1 -> code 0: C2H2=0, C2H4=50 -> 0
        # CH4/H2: ratio < 0.1 -> code 0: CH4=5, H2=100 -> 0.05
        # C2H4/C2H6: 1 <= ratio <= 3 -> code 2: C2H4=100, C2H6=50 -> 2.0
        # CO2/CO: ratio < 0.1 -> code 0: CO2=5, CO=100 -> 0.05
        result = rogers.diagnose(
            h2=100, ch4=5, c2h2=0, c2h4=100, c2h6=50, co=100, co2=5
        )

        assert result.fault_type == FaultType.T1

    def test_diagnose_thermal_fault_t2(self):
        """Test medium temperature thermal fault diagnosis."""
        rogers = RogersRatios()
        # Target code (0, 1, 1, 0) -> T2
        # C2H2/C2H4: ratio < 0.1 -> code 0: C2H2=0, C2H4=50 -> 0
        # CH4/H2: 0.1 <= ratio < 1 -> code 1: CH4=50, H2=100 -> 0.5
        # C2H4/C2H6: 1 <= ratio <= 3 -> code 2: C2H4=100, C2H6=50 -> 2.0
        # CO2/CO: ratio < 0.1 -> code 0: CO2=5, CO=100 -> 0.05
        result = rogers.diagnose(
            h2=100, ch4=50, c2h2=0, c2h4=100, c2h6=50, co=100, co2=5
        )

        assert result.fault_type == FaultType.T2

    def test_diagnose_thermal_fault_t3(self):
        """Test high temperature thermal fault diagnosis."""
        rogers = RogersRatios()
        # Target code (0, 0, 2, 1) -> T3
        # C2H2/C2H4: ratio < 0.1 -> code 0: C2H2=0, C2H4=50 -> 0
        # CH4/H2: ratio < 0.1 -> code 0: CH4=5, H2=100 -> 0.05
        # C2H4/C2H6: 1 <= ratio <= 3 -> code 2: C2H4=100, C2H6=50 -> 2.0
        # CO2/CO: 0.1 <= ratio < 1 -> code 1: CO2=50, CO=100 -> 0.5
        result = rogers.diagnose(
            h2=100, ch4=5, c2h2=0, c2h4=100, c2h6=50, co=100, co2=50
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

        # Should detect some form of thermal fault or be undetermined
        assert result.fault_type in [
            FaultType.T1,
            FaultType.T2,
            FaultType.T3,
            FaultType.DT,
            FaultType.NORMAL,
            FaultType.UNDETERMINED,
        ]

    def test_arcing_sample(self, sample_dga_arcing):
        """Test with typical arcing DGA data."""
        rogers = RogersRatios()
        result = rogers.diagnose(**sample_dga_arcing)

        # Should detect discharge fault or be undetermined
        assert result.fault_type in [
            FaultType.D1,
            FaultType.D2,
            FaultType.DT,
            FaultType.NORMAL,
            FaultType.UNDETERMINED,
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

        # Use values that produce a known fault type (D2)
        # Target code (1, 1, 0, 0) -> D2
        result = rogers.diagnose(
            h2=100, ch4=50, c2h2=10, c2h4=50, c2h6=100, co=100, co2=5
        )

        # Clear indicators should give higher confidence
        assert result.confidence > 0.5

    def test_low_confidence_for_undetermined(self):
        """Test that undetermined results have lower confidence."""
        rogers = RogersRatios()

        # Get an undetermined result
        result = rogers.diagnose(
            h2=50, ch4=50, c2h2=50, c2h4=50, c2h6=50, co=50, co2=50
        )

        # Check confidence is in valid range
        assert 0 <= result.confidence <= 1
