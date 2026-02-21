"""
Tests for IEEE C57.91-2011 Thermal Model

This module tests the IEEEC57_91 class for thermal calculations
based on the IEEE C57.91-2011 standard.

Author: TransformerGuard Team
"""

import pytest

from src.thermal.ieee_c57_91 import COOLING_MODES, IEEEC57_91, TransformerParameters


class TestIEEEC57_91:
    """Test suite for IEEEC57_91 thermal model class."""

    def test_initialization(self):
        """Test that IEEEC57_91 initializes correctly."""
        model = IEEEC57_91(
            rated_mva=25.0,
            rated_voltage=138000.0,
            rated_current=1200.0,
            cooling_mode="ONAF",
        )

        assert model is not None
        assert model.params.rated_mva == 25.0
        assert model.params.cooling_mode == "ONAF"

    def test_initialization_with_default_cooling(self):
        """Test initialization with default cooling mode."""
        model = IEEEC57_91(rated_mva=25.0, rated_voltage=138000.0, rated_current=1200.0)

        # Default should be ONAN
        assert model.params.cooling_mode == "ONAN"

    def test_invalid_cooling_mode_raises_error(self):
        """Test that invalid cooling mode raises ValueError."""
        with pytest.raises(ValueError):
            IEEEC57_91(
                rated_mva=25.0,
                rated_voltage=138000.0,
                rated_current=1200.0,
                cooling_mode="INVALID",
            )

    def test_calculate_load_factor(self):
        """Test load factor calculation."""
        model = IEEEC57_91(rated_mva=25.0, rated_voltage=138000.0, rated_current=1200.0)

        # Load of 20 MVA on 25 MVA rated = 0.8
        load_factor = model.calculate_load_factor(20.0)

        assert load_factor == 0.8

    def test_calculate_load_factor_at_rated(self):
        """Test load factor at rated load."""
        model = IEEEC57_91(rated_mva=25.0, rated_voltage=138000.0, rated_current=1200.0)

        load_factor = model.calculate_load_factor(25.0)

        assert load_factor == 1.0

    def test_calculate_load_factor_zero(self):
        """Test load factor at no load."""
        model = IEEEC57_91(rated_mva=25.0, rated_voltage=138000.0, rated_current=1200.0)

        load_factor = model.calculate_load_factor(0.0)

        assert load_factor == 0.0

    def test_calculate_top_oil_rise(self):
        """Test top-oil temperature rise calculation."""
        model = IEEEC57_91(
            rated_mva=25.0,
            rated_voltage=138000.0,
            rated_current=1200.0,
            cooling_mode="ONAF",
        )

        # At rated load (K=1.0), rise should equal rated value
        rise = model.calculate_top_oil_rise(1.0)

        # ONAF rated top-oil rise is 40°C
        assert abs(rise - 40.0) < 1.0

    def test_calculate_top_oil_rise_at_half_load(self):
        """Test top-oil rise at half load."""
        model = IEEEC57_91(
            rated_mva=25.0,
            rated_voltage=138000.0,
            rated_current=1200.0,
            cooling_mode="ONAF",
        )

        # At half load (K=0.5), rise should be less
        rise = model.calculate_top_oil_rise(0.5)

        # Should be less than rated
        assert rise < 40.0

    def test_calculate_top_oil_rise_overload(self):
        """Test top-oil rise at overload."""
        model = IEEEC57_91(
            rated_mva=25.0,
            rated_voltage=138000.0,
            rated_current=1200.0,
            cooling_mode="ONAF",
        )

        # At 120% load (K=1.2), rise should be more
        rise = model.calculate_top_oil_rise(1.2)

        # Should be more than rated
        assert rise > 40.0

    def test_calculate_winding_gradient(self):
        """Test winding gradient calculation."""
        model = IEEEC57_91(
            rated_mva=25.0,
            rated_voltage=138000.0,
            rated_current=1200.0,
            cooling_mode="ONAF",
        )

        # At rated load
        gradient = model.calculate_winding_gradient(1.0)

        # ONAF rated hotspot gradient is 55°C
        assert abs(gradient - 55.0) < 1.0

    def test_calculate_top_oil_temperature(self):
        """Test absolute top-oil temperature calculation."""
        model = IEEEC57_91(
            rated_mva=25.0,
            rated_voltage=138000.0,
            rated_current=1200.0,
            cooling_mode="ONAF",
        )

        # At 25°C ambient, rated load
        temp = model.calculate_top_oil_temperature(25.0, 1.0)

        # Should be ambient + rise (25 + 40 = 65)
        assert abs(temp - 65.0) < 2.0

    def test_calculate_hotspot_temperature(self):
        """Test hot-spot temperature calculation."""
        model = IEEEC57_91(
            rated_mva=25.0,
            rated_voltage=138000.0,
            rated_current=1200.0,
            cooling_mode="ONAF",
        )

        # At 25°C ambient, rated load
        hotspot = model.calculate_hotspot_temperature(25.0, 1.0)

        # Should be ambient + top-oil rise + gradient (25 + 40 + 55 = 120)
        assert abs(hotspot - 120.0) < 2.0

    def test_hotspot_above_ambient(self):
        """Test that hot-spot is always above ambient."""
        model = IEEEC57_91(
            rated_mva=25.0,
            rated_voltage=138000.0,
            rated_current=1200.0,
            cooling_mode="ONAF",
        )

        hotspot = model.calculate_hotspot_temperature(25.0, 0.5)

        assert hotspot > 25.0

    def test_hotspot_increases_with_load(self):
        """Test that hotspot increases with load."""
        model = IEEEC57_91(
            rated_mva=25.0,
            rated_voltage=138000.0,
            rated_current=1200.0,
            cooling_mode="ONAF",
        )

        hotspot_low = model.calculate_hotspot_temperature(25.0, 0.5)
        hotspot_high = model.calculate_hotspot_temperature(25.0, 1.0)

        assert hotspot_high > hotspot_low

    def test_transient_hotspot_initial(self):
        """Test transient hotspot at initial time."""
        model = IEEEC57_91(
            rated_mva=25.0,
            rated_voltage=138000.0,
            rated_current=1200.0,
            cooling_mode="ONAF",
        )

        # At time=0, should return initial hotspot
        hotspot = model.calculate_transient_hotspot(
            ambient_temp=25.0, load_factor=1.0, initial_hotspot=80.0, time_minutes=0
        )

        assert hotspot == 80.0

    def test_transient_hotspot_steady_state(self):
        """Test transient hotspot approaches steady state."""
        model = IEEEC57_91(
            rated_mva=25.0,
            rated_voltage=138000.0,
            rated_current=1200.0,
            cooling_mode="ONAF",
        )

        # After long time, should approach steady state
        hotspot = model.calculate_transient_hotspot(
            ambient_temp=25.0,
            load_factor=1.0,
            initial_hotspot=80.0,
            time_minutes=1000,  # Long time
        )

        steady_state = model.calculate_hotspot_temperature(25.0, 1.0)

        # Should be very close to steady state
        assert abs(hotspot - steady_state) < 1.0

    def test_transient_hotspot_middle(self):
        """Test transient hotspot at intermediate time."""
        model = IEEEC57_91(
            rated_mva=25.0,
            rated_voltage=138000.0,
            rated_current=1200.0,
            cooling_mode="ONAF",
        )

        hotspot = model.calculate_transient_hotspot(
            ambient_temp=25.0, load_factor=1.0, initial_hotspot=80.0, time_minutes=10
        )

        # Should be between initial and steady state
        steady_state = model.calculate_hotspot_temperature(25.0, 1.0)

        assert 80.0 < hotspot < steady_state

    def test_get_cooling_mode_info(self):
        """Test getting cooling mode information."""
        model = IEEEC57_91(
            rated_mva=25.0,
            rated_voltage=138000.0,
            rated_current=1200.0,
            cooling_mode="ONAF",
        )

        info = model.get_cooling_mode_info()

        assert info["mode"] == "ONAF"
        assert "description" in info
        assert "top_oil_rise_rated" in info
        assert "hotspot_gradient_rated" in info

    def test_get_available_cooling_modes(self):
        """Test getting all available cooling modes."""
        modes = IEEEC57_91.get_available_cooling_modes()

        assert "ONAN" in modes
        assert "ONAF" in modes
        assert "OFAF" in modes
        assert "ODAF" in modes

    def test_cooling_mode_parameters_onan(self):
        """Test ONAN cooling mode parameters."""
        modes = IEEEC57_91.get_available_cooling_modes()

        assert modes["ONAN"]["top_oil_rise"] == 45
        assert modes["ONAN"]["hotspot_gradient"] == 65

    def test_cooling_mode_parameters_onaf(self):
        """Test ONAF cooling mode parameters."""
        modes = IEEEC57_91.get_available_cooling_modes()

        assert modes["ONAF"]["top_oil_rise"] == 40
        assert modes["ONAF"]["hotspot_gradient"] == 55

    def test_cooling_mode_parameters_ofaf(self):
        """Test OFAF cooling mode parameters."""
        modes = IEEEC57_91.get_available_cooling_modes()

        assert modes["OFAF"]["top_oil_rise"] == 35
        assert modes["OFAF"]["hotspot_gradient"] == 50

    def test_cooling_mode_parameters_odaf(self):
        """Test ODAF cooling mode parameters."""
        modes = IEEEC57_91.get_available_cooling_modes()

        assert modes["ODAF"]["top_oil_rise"] == 30
        assert modes["ODAF"]["hotspot_gradient"] == 45


class TestTransformerParameters:
    """Test suite for TransformerParameters dataclass."""

    def test_transformer_parameters_creation(self):
        """Test creating TransformerParameters."""
        params = TransformerParameters(
            rated_mva=25.0,
            rated_voltage=138000.0,
            rated_current=1200.0,
            cooling_mode="ONAF",
        )

        assert params.rated_mva == 25.0
        assert params.rated_voltage == 138000.0
        assert params.rated_current == 1200.0
        assert params.cooling_mode == "ONAF"

    def test_default_cooling_mode(self):
        """Test default cooling mode is ONAN."""
        params = TransformerParameters(
            rated_mva=25.0, rated_voltage=138000.0, rated_current=1200.0
        )

        assert params.cooling_mode == "ONAN"

    def test_custom_top_oil_rise(self):
        """Test custom top-oil rise parameter."""
        params = TransformerParameters(
            rated_mva=25.0,
            rated_voltage=138000.0,
            rated_current=1200.0,
            cooling_mode="ONAF",
            top_oil_rise=50.0,
        )

        assert params.top_oil_rise == 50.0

    def test_custom_hotspot_gradient(self):
        """Test custom hotspot gradient parameter."""
        params = TransformerParameters(
            rated_mva=25.0,
            rated_voltage=138000.0,
            rated_current=1200.0,
            cooling_mode="ONAF",
            hotspot_gradient=60.0,
        )

        assert params.hotspot_gradient == 60.0


class TestCoolingModes:
    """Test suite for cooling mode constants."""

    def test_cooling_modes_dict_structure(self):
        """Test cooling modes dictionary has correct structure."""
        for mode, params in COOLING_MODES.items():
            assert "top_oil_rise" in params
            assert "hotspot_gradient" in params
            assert "description" in params

    def test_all_standard_modes_present(self):
        """Test all standard cooling modes are present."""
        expected_modes = ["ONAN", "ONAF", "OFAF", "ODAF"]

        for mode in expected_modes:
            assert mode in COOLING_MODES


class TestThermalCalculations:
    """Integration tests for thermal calculations."""

    def test_complete_thermal_calculation(self):
        """Test complete thermal calculation workflow."""
        model = IEEEC57_91(
            rated_mva=25.0,
            rated_voltage=138000.0,
            rated_current=1200.0,
            cooling_mode="ONAF",
        )

        # Given conditions
        ambient_temp = 30.0
        load_percent = 80  # 80% = 20 MVA on 25 MVA

        # Calculate load factor
        load_factor = model.calculate_load_factor(load_percent * 0.25)  # 20/100 = 0.8

        # Calculate temperatures
        top_oil_rise = model.calculate_top_oil_rise(load_factor)
        winding_gradient = model.calculate_winding_gradient(load_factor)
        top_oil_temp = model.calculate_top_oil_temperature(ambient_temp, load_factor)
        hotspot = model.calculate_hotspot_temperature(ambient_temp, load_factor)

        # Verify calculations
        assert top_oil_rise > 0
        assert winding_gradient > 0
        assert top_oil_temp > ambient_temp
        assert hotspot > top_oil_temp

    def test_thermal_rise_with_different_cooling_modes(self):
        """Test that different cooling modes give different rises."""
        ambient = 25.0
        load = 1.0

        results = {}

        for mode in ["ONAN", "ONAF", "OFAF", "ODAF"]:
            model = IEEEC57_91(
                rated_mva=25.0,
                rated_voltage=138000.0,
                rated_current=1200.0,
                cooling_mode=mode,
            )
            hotspot = model.calculate_hotspot_temperature(ambient, load)
            results[mode] = hotspot

        # ONAN should have highest hotspot (least cooling)
        # ODAF should have lowest hotspot (most cooling)
        assert results["ONAN"] > results["ONAF"]
        assert results["OFAF"] > results["ODAF"]
