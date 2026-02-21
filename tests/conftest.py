"""
Shared Test Fixtures for TransformerGuard Tests

This module provides shared pytest fixtures for testing the TransformerGuard modules.
"""

import sys
from pathlib import Path

import pytest

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


# ============================================================================
# DGA Test Fixtures
# ============================================================================


@pytest.fixture
def sample_dga_normal():
    """Sample DGA data for normal transformer operation."""
    return {
        "h2": 45,
        "ch4": 120,
        "c2h2": 0,
        "c2h4": 15,
        "c2h6": 35,
        "co": 180,
        "co2": 1200,
    }


@pytest.fixture
def sample_dga_thermal_fault():
    """Sample DGA data for thermal fault (T2: 300-700°C)."""
    return {
        "h2": 150,
        "ch4": 400,
        "c2h2": 5,
        "c2h4": 250,
        "c2h6": 80,
        "co": 350,
        "co2": 2000,
    }


@pytest.fixture
def sample_dga_arcing():
    """Sample DGA data for arcing (high energy discharge - D2)."""
    return {
        "h2": 500,
        "ch4": 200,
        "c2h2": 150,
        "c2h4": 300,
        "c2h6": 50,
        "co": 400,
        "co2": 2500,
    }


@pytest.fixture
def sample_dga_partial_discharge():
    """Sample DGA data for partial discharge (PD)."""
    return {
        "h2": 300,
        "ch4": 250,
        "c2h2": 2,
        "c2h4": 15,
        "c2h6": 50,
        "co": 100,
        "co2": 800,
    }


@pytest.fixture
def sample_dga_low_energy_discharge():
    """Sample DGA data for low energy discharge (D1)."""
    return {
        "h2": 400,
        "ch4": 100,
        "c2h2": 80,
        "c2h4": 60,
        "c2h6": 30,
        "co": 200,
        "co2": 1500,
    }


@pytest.fixture
def sample_dga_high_temp_thermal():
    """Sample DGA data for high temperature thermal fault (T3 >700°C)."""
    return {
        "h2": 100,
        "ch4": 150,
        "c2h2": 10,
        "c2h4": 500,
        "c2h6": 100,
        "co": 800,
        "co2": 3000,
    }


@pytest.fixture
def sample_dga_mixed_fault():
    """Sample DGA data for mixed discharge/thermal fault (DT)."""
    return {
        "h2": 250,
        "ch4": 300,
        "c2h2": 40,
        "c2h4": 200,
        "c2h6": 60,
        "co": 300,
        "co2": 1800,
    }


@pytest.fixture
def sample_dga_zero_gases():
    """Sample DGA data with all zeros (should result in normal/undetermined)."""
    return {"h2": 0, "ch4": 0, "c2h2": 0, "c2h4": 0, "c2h6": 0, "co": 0, "co2": 0}


# ============================================================================
# Transformer Parameter Fixtures
# ============================================================================


@pytest.fixture
def sample_transformer_params():
    """Sample transformer parameters for thermal calculations."""
    return {
        "rated_mva": 25.0,
        "rated_voltage": 138000.0,
        "rated_current": 1200.0,
        "cooling_mode": "ONAF",
    }


@pytest.fixture
def sample_oil_quality_data():
    """Sample oil quality test data."""
    return {
        "dielectric_strength": 45.0,  # kV
        "moisture_content": 15.0,  # ppm
        "acidity": 0.10,  # mgKOH/g
        "interfacial_tension": 40.0,  # mN/m
    }


@pytest.fixture
def sample_electrical_test_data():
    """Sample electrical test data."""
    return {
        "power_factor": 0.5,  # %
        "capacitance": 1200.0,  # pF
        "insulation_resistance": 5000,  # MΩ
    }


# ============================================================================
# Health Index Fixtures
# ============================================================================


@pytest.fixture
def sample_health_index_healthy():
    """Sample data for a healthy transformer."""
    return {
        "dga_gases": {
            "h2": 30,
            "ch4": 80,
            "c2h2": 0,
            "c2h4": 10,
            "c2h6": 25,
            "co": 150,
            "co2": 1000,
        },
        "dielectric_strength": 50.0,
        "moisture_content": 10.0,
        "acidity": 0.05,
        "interfacial_tension": 45.0,
        "power_factor": 0.3,
        "capacitance": 1100.0,
        "insulation_resistance": 10000.0,
        "age_years": 5.0,
        "average_load_percent": 50.0,
    }


@pytest.fixture
def sample_health_index_critical():
    """Sample data for a critical condition transformer."""
    return {
        "dga_gases": {
            "h2": 800,
            "ch4": 500,
            "c2h2": 200,
            "c2h4": 600,
            "c2h6": 150,
            "co": 600,
            "co2": 4000,
        },
        "dielectric_strength": 20.0,
        "moisture_content": 50.0,
        "acidity": 0.5,
        "interfacial_tension": 15.0,
        "power_factor": 5.0,
        "capacitance": 1800.0,
        "insulation_resistance": 100.0,
        "age_years": 25.0,
        "average_load_percent": 95.0,
    }


# ============================================================================
# Thermal Test Fixtures
# ============================================================================


@pytest.fixture
def sample_thermal_conditions():
    """Sample thermal operating conditions."""
    return {"ambient_temp": 25.0, "load_percent": 80.0, "cooling_type": "ONAF"}


@pytest.fixture
def sample_hot_spot_temps():
    """Sample hot-spot temperature profile."""
    return [95, 100, 105, 110, 115, 120]


# ============================================================================
# RUL Estimation Fixtures
# ============================================================================


@pytest.fixture
def sample_health_index_history():
    """Sample health index history for RUL estimation."""
    return [95, 92, 88, 85, 80, 75, 70, 65]


@pytest.fixture
def sample_fault_types():
    """Sample fault types for RUL estimation."""
    return ["thermal_fault_medium", "overheating"]
