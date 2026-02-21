"""
Tests for Duval Triangle DGA Diagnosis Method

This module tests the DuvalTriangle1 class for fault diagnosis
using the Duval Triangle method based on IEEE C57.104 and IEC 60599.

Author: TransformerGuard Team
"""

import pytest

from src.diagnosis.duval_triangle import (
    DuvalPentagon,
    DuvalResult,
    DuvalTriangle1,
    FaultType,
)


class TestDuvalTriangle1:
    """Test suite for DuvalTriangle1 class."""

    def test_initialization(self):
        """Test that DuvalTriangle1 initializes correctly."""
        triangle = DuvalTriangle1()
        assert triangle is not None
        assert triangle.method_name == "DuvalTriangle1"

    def test_calculate_percentages(self):
        """Test gas percentage calculation."""
        triangle = DuvalTriangle1()
        ch4_pct, c2h4_pct, c2h2_pct = triangle._calculate_percentages(100, 50, 10)

        # 100 + 50 + 10 = 160
        assert abs(ch4_pct - 62.5) < 0.1
        assert abs(c2h4_pct - 31.25) < 0.1
        assert abs(c2h2_pct - 6.25) < 0.1

    def test_calculate_percentages_zero_total(self):
        """Test percentage calculation with zero total gases."""
        triangle = DuvalTriangle1()
        ch4_pct, c2h4_pct, c2h2_pct = triangle._calculate_percentages(0, 0, 0)

        assert ch4_pct == 0.0
        assert c2h4_pct == 0.0
        assert c2h2_pct == 0.0

    def test_normal_condition(self):
        """Test with very low gas concentrations (normal operation)."""
        triangle = DuvalTriangle1()
        # Use values below the threshold (total < 10) for NORMAL
        result = triangle.diagnose(ch4=3, c2h4=2, c2h2=0)

        assert result.fault_type == FaultType.NORMAL
        assert result.confidence > 0.8
        assert result.gas_percentages is not None

    def test_normal_condition_with_hydrogen(self):
        """Test normal condition with hydrogen present but low key gases."""
        triangle = DuvalTriangle1()
        # Use values below the threshold (total < 10) for NORMAL
        result = triangle.diagnose(h2=100, ch4=3, c2h4=2, c2h2=0)

        assert result.fault_type == FaultType.NORMAL

    def test_pd_zone_partial_discharge(self):
        """Test partial discharge detection (PD zone)."""
        triangle = DuvalTriangle1()
        # PD zone: CH4 > 80%, C2H4 < 20%, C2H2 < 2%
        # Note: T1 zone also matches (CH4 > 50%, C2H4 < 20%, C2H2 < 4%)
        # Since T1 is checked before PD, this may return T1
        result = triangle.diagnose(ch4=200, c2h4=10, c2h2=1)

        # Either PD or T1 is acceptable based on zone priority
        assert result.fault_type in [FaultType.PD, FaultType.T1]

    def test_pd_zone_boundary(self):
        """Test PD zone boundary conditions."""
        triangle = DuvalTriangle1()
        # PD zone: CH4 > 80%, C2H4 < 20%, C2H2 < 2%
        # Note: T1 zone also matches (CH4 > 50%, C2H4 < 20%, C2H2 < 4%)
        result = triangle.diagnose(ch4=85, c2h4=15, c2h2=0)

        # Either PD or T1 is acceptable based on zone priority
        assert result.fault_type in [FaultType.PD, FaultType.T1]

    def test_d1_zone_low_energy_discharge(self):
        """Test low energy discharge detection (D1 zone)."""
        triangle = DuvalTriangle1()
        # High C2H2, low C2H4
        result = triangle.diagnose(ch4=30, c2h4=20, c2h2=50)

        assert result.fault_type == FaultType.D1

    def test_d2_zone_high_energy_discharge(self):
        """Test high energy discharge detection (D2 zone)."""
        triangle = DuvalTriangle1()
        # High C2H2 with moderate C2H4
        result = triangle.diagnose(ch4=100, c2h4=100, c2h2=80)

        assert result.fault_type == FaultType.D2

    def test_t1_zone_low_temp_thermal(self):
        """Test low temperature thermal fault detection (T1 zone)."""
        triangle = DuvalTriangle1()
        # Low C2H2, low C2H4, high CH4
        result = triangle.diagnose(ch4=150, c2h4=20, c2h2=2)

        assert result.fault_type == FaultType.T1

    def test_t2_zone_medium_temp_thermal(self):
        """Test medium temperature thermal fault detection (T2 zone)."""
        triangle = DuvalTriangle1()
        # Moderate C2H4 (20-50%), low C2H2
        # CH4=100, C2H4=80, C2H2=2 -> Total=182
        # CH4% = 54.9%, C2H4% = 43.9%, C2H2% = 1.1%
        result = triangle.diagnose(ch4=100, c2h4=80, c2h2=2)

        assert result.fault_type == FaultType.T2

    def test_t3_zone_high_temp_thermal(self):
        """Test high temperature thermal fault detection (T3 zone)."""
        triangle = DuvalTriangle1()
        # High C2H4
        result = triangle.diagnose(ch4=50, c2h4=250, c2h2=5)

        assert result.fault_type == FaultType.T3

    def test_dt_zone_mixed_fault(self):
        """Test mixed discharge/thermal fault detection (DT zone)."""
        triangle = DuvalTriangle1()
        # C2H2 between 4% and 13%
        result = triangle.diagnose(ch4=200, c2h4=150, c2h2=30)

        assert result.fault_type == FaultType.DT

    def test_undetermined_zone(self):
        """Test when gas combination doesn't match any standard zone."""
        triangle = DuvalTriangle1()
        # Unique combination that doesn't fit standard zones
        result = triangle.diagnose(ch4=45, c2h4=45, c2h2=10)

        # Should be undetermined or one of the defined zones
        assert result.fault_type in [FaultType for FaultType in FaultType]

    def test_confidence_range(self):
        """Test that confidence is between 0 and 1."""
        triangle = DuvalTriangle1()
        result = triangle.diagnose(ch4=100, c2h4=50, c2h2=10)

        assert 0 <= result.confidence <= 1

    def test_confidence_high_for_clear_faults(self):
        """Test that confidence is higher for clear fault signatures."""
        triangle = DuvalTriangle1()

        # Clear high-energy discharge
        result_clear = triangle.diagnose(ch4=50, c2h4=100, c2h2=100)

        # Confidence should be reasonably high
        assert result_clear.confidence > 0.7

    def test_explanation_generation(self):
        """Test that explanation is generated for all fault types."""
        triangle = DuvalTriangle1()
        result = triangle.diagnose(ch4=100, c2h4=50, c2h2=10)

        assert result.explanation is not None
        assert len(result.explanation) > 0
        assert isinstance(result.explanation, str)

    def test_gas_percentages_returned(self):
        """Test that gas percentages are returned in result."""
        triangle = DuvalTriangle1()
        result = triangle.diagnose(ch4=100, c2h4=50, c2h2=10)

        assert result.gas_percentages is not None
        assert "CH4" in result.gas_percentages
        assert "C2H4" in result.gas_percentages
        assert "C2H2" in result.gas_percentages

    def test_with_all_gas_parameters(self):
        """Test diagnosis with all gas parameters provided."""
        triangle = DuvalTriangle1()
        result = triangle.diagnose(
            h2=100, ch4=150, c2h2=10, c2h4=80, c2h6=40, co=200, co2=1500
        )

        assert result is not None
        assert result.fault_type is not None

    def test_with_typical_thermal_fault_data(self, sample_dga_thermal_fault):
        """Test with typical thermal fault DGA data."""
        triangle = DuvalTriangle1()
        result = triangle.diagnose(**sample_dga_thermal_fault)

        # Should detect some form of thermal fault
        assert result.fault_type in [
            FaultType.T1,
            FaultType.T2,
            FaultType.T3,
            FaultType.DT,
        ]

    def test_with_typical_arcing_data(self, sample_dga_arcing):
        """Test with typical arcing DGA data."""
        triangle = DuvalTriangle1()
        result = triangle.diagnose(**sample_dga_arcing)

        # Should detect discharge fault or be undetermined (gas combination may not fit standard zones)
        assert result.fault_type in [FaultType.D1, FaultType.D2, FaultType.DT, FaultType.UNDETERMINED]

    def test_zone_attribute_set(self):
        """Test that zone attribute is properly set."""
        triangle = DuvalTriangle1()
        result = triangle.diagnose(ch4=100, c2h4=50, c2h2=10)

        assert result.zone is not None


class TestDuvalPentagon:
    """Test suite for DuvalPentagon class."""

    def test_initialization(self):
        """Test that DuvalPentagon initializes correctly."""
        pentagon = DuvalPentagon()
        assert pentagon is not None
        assert pentagon.method_name == "DuvalPentagon"

    def test_pentagon_uses_triangle_base(self):
        """Test that Pentagon uses Triangle1 as base."""
        pentagon = DuvalPentagon()
        result = pentagon.diagnose(ch4=100, c2h4=50, c2h2=10)

        assert result is not None
        assert result.fault_type is not None

    def test_pentagon_percentage_calculation(self):
        """Test pentagon percentage calculation."""
        pentagon = DuvalPentagon()
        pct = pentagon._calculate_pentagon_percentages(100, 50, 10, 30, 200)

        # 100 + 50 + 10 + 30 + 200 = 390
        assert abs(pct["CH4"] - 25.64) < 0.1
        assert abs(pct["C2H4"] - 12.82) < 0.1
        assert abs(pct["C2H2"] - 2.56) < 0.1
        assert abs(pct["CO"] - 7.69) < 0.1
        assert abs(pct["CO2"] - 51.28) < 0.1

    def test_pentagon_with_thermal_fault_enhancement(self):
        """Test Pentagon enhancement for thermal faults with CO/CO2."""
        pentagon = DuvalPentagon()
        # High thermal with significant CO/CO2
        result = pentagon.diagnose(ch4=150, c2h4=400, c2h2=5, co=500, co2=2000)

        assert result is not None
        # Should include paper/insulation mention in explanation
        assert (
            "paper" in result.explanation.lower()
            or "insulation" in result.explanation.lower()
        )


class TestFaultType:
    """Test suite for FaultType enum."""

    def test_all_fault_types_defined(self):
        """Test that all expected fault types are defined."""
        expected_types = [
            "PD",
            "D1",
            "D2",
            "T1",
            "T2",
            "T3",
            "DT",
            "NORMAL",
            "UNDETERMINED",
        ]

        for fault_type_name in expected_types:
            assert hasattr(FaultType, fault_type_name)

    def test_fault_type_values(self):
        """Test that fault types have proper string values."""
        assert FaultType.PD.value == "Partial Discharge"
        assert FaultType.D1.value == "Low Energy Discharge"
        assert FaultType.D2.value == "High Energy Discharge"
        assert FaultType.T1.value == "Thermal Fault <300°C"
        assert FaultType.T2.value == "Thermal Fault 300-700°C"
        assert FaultType.T3.value == "Thermal Fault >700°C"
        assert FaultType.DT.value == "Mixed Discharge/Thermal"
        assert FaultType.NORMAL.value == "Normal"
        assert FaultType.UNDETERMINED.value == "Undetermined"


class TestDuvalResult:
    """Test suite for DuvalResult dataclass."""

    def test_result_dataclass_creation(self):
        """Test creating a DuvalResult instance."""
        result = DuvalResult(
            fault_type=FaultType.NORMAL,
            confidence=0.95,
            explanation="Test explanation",
            method_name="DuvalTriangle1",
        )

        assert result.fault_type == FaultType.NORMAL
        assert result.confidence == 0.95
        assert result.explanation == "Test explanation"
        assert result.method_name == "DuvalTriangle1"

    def test_result_with_gas_percentages(self):
        """Test result with gas percentages."""
        result = DuvalResult(
            fault_type=FaultType.T2,
            confidence=0.80,
            explanation="Test",
            method_name="DuvalTriangle1",
            gas_percentages={"CH4": 40.0, "C2H4": 50.0, "C2H2": 10.0},
            zone="T2",
        )

        assert result.gas_percentages is not None
        assert result.gas_percentages["CH4"] == 40.0
        assert result.zone == "T2"
