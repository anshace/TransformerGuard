"""
IEEE C57.91-2011 Thermal Model
Main thermal model class implementing IEEE C57.91-2011 standard
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml

# Default constants per IEEE C57.91-2011
IEEE_C57_91_DEFAULTS = {
    "reference_hotspot": 110,  # °C
    "min_life_expectancy_hours": 180000,  # 20.5 years
    "aging_doubling_temp": 6,  # °C
    "oil_exponent": 0.8,
    "winding_exponent": 0.8,
    "top_oil_time_constant": 150,  # minutes
    "winding_time_constant": 5,  # minutes
}


# Cooling mode parameters
COOLING_MODES = {
    "ONAN": {  # Oil Natural, Air Natural
        "top_oil_rise": 45,
        "hotspot_gradient": 65,
        "description": "Oil Natural, Air Natural",
    },
    "ONAF": {  # Oil Natural, Air Forced
        "top_oil_rise": 40,
        "hotspot_gradient": 55,
        "description": "Oil Natural, Air Forced",
    },
    "OFAF": {  # Oil Forced, Air Forced
        "top_oil_rise": 35,
        "hotspot_gradient": 50,
        "description": "Oil Forced, Air Forced",
    },
    "ODAF": {  # Oil Directed, Air Forced
        "top_oil_rise": 30,
        "hotspot_gradient": 45,
        "description": "Oil Directed, Air Forced",
    },
}


@dataclass
class TransformerParameters:
    """Transformer parameters for thermal modeling"""

    rated_mva: float
    rated_voltage: float
    rated_current: float
    cooling_mode: str = "ONAN"
    top_oil_rise: Optional[float] = None
    hotspot_gradient: Optional[float] = None
    oil_exponent: float = 0.8
    winding_exponent: float = 0.8

    def __post_init__(self):
        if self.cooling_mode not in COOLING_MODES:
            raise ValueError(
                f"Invalid cooling mode: {self.cooling_mode}. "
                f"Must be one of {list(COOLING_MODES.keys())}"
            )

        # Set defaults from cooling mode if not provided
        if self.top_oil_rise is None:
            self.top_oil_rise = COOLING_MODES[self.cooling_mode]["top_oil_rise"]
        if self.hotspot_gradient is None:
            self.hotspot_gradient = COOLING_MODES[self.cooling_mode]["hotspot_gradient"]


class IEEEC57_91:
    """
    IEEE C57.91-2011 Thermal Model

    Implements the IEEE C57.91-2011 standard for thermal modeling of oil-filled
    power transformers. This model calculates:
    - Top-oil temperature rise
    - Hot-spot temperature
    - Aging acceleration factors
    - Loss-of-life estimates
    - Loading capability

    Key equations:
    - Δθ_top = Δθ_top_rated × (K^m) where K = load/rated, m = oil exponent
    - Δθ_h = Δθ_h_rated × (K^n) where n = winding exponent
    - θ_h = θ_amb + Δθ_top + Δθ_h

    Attributes:
        params: Transformer parameters
        config: Configuration parameters loaded from YAML
    """

    def __init__(
        self,
        rated_mva: float,
        rated_voltage: float,
        rated_current: float,
        cooling_mode: str = "ONAN",
        config_path: Optional[str] = None,
    ):
        """
        Initialize the IEEE C57.91 thermal model

        Args:
            rated_mva: Rated power in MVA
            rated_voltage: Rated voltage in volts
            rated_current: Rated current in amperes
            cooling_mode: Cooling mode (ONAN, ONAF, OFAF, ODAF)
            config_path: Path to thermal_params.yaml configuration file
        """
        self.params = TransformerParameters(
            rated_mva=rated_mva,
            rated_voltage=rated_voltage,
            rated_current=rated_current,
            cooling_mode=cooling_mode,
        )

        # Load configuration from YAML file
        self.config = self._load_config(config_path)

        # Set model constants
        self._set_constants()

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        default_config = {
            "ambient": {"default": 25, "daily_average": 20, "yearly_average": 20},
            "hotspot": {"constant": 0, "gradient": 65, "exponent": 0.8},
            "loading": {
                "nameplate_mva": 25,
                "rated_current": 1200,
                "rated_voltage": 138000,
            },
            "insulation": {
                "reference_hotspot": 110,
                "temperature_exponent": 0.075,
                "life_years": 20.5,
                "life_hours": 180000,
            },
            "top_oil": {"constant": 30, "gradient": 45, "exponent": 0.8},
            "cooling": COOLING_MODES,
        }

        if config_path is None:
            # Try to find config in standard location
            possible_paths = [
                "config/thermal_params.yaml",
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "config",
                    "thermal_params.yaml",
                ),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    yaml_config = yaml.safe_load(f)
                    # Merge with defaults
                    default_config.update(yaml_config)
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")

        return default_config

    def _set_constants(self):
        """Set model constants from configuration"""
        self.reference_hotspot = self.config.get("insulation", {}).get(
            "reference_hotspot", IEEE_C57_91_DEFAULTS["reference_hotspot"]
        )
        self.min_life_hours = self.config.get("insulation", {}).get(
            "life_hours", IEEE_C57_91_DEFAULTS["min_life_expectancy_hours"]
        )
        self.aging_doubling_temp = IEEE_C57_91_DEFAULTS["aging_doubling_temp"]
        self.oil_exponent = self.params.oil_exponent
        self.winding_exponent = self.params.winding_exponent

        # Time constants
        self.top_oil_time_constant = IEEE_C57_91_DEFAULTS["top_oil_time_constant"]
        self.winding_time_constant = IEEE_C57_91_DEFAULTS["winding_time_constant"]

    def calculate_load_factor(self, load_mva: float) -> float:
        """
        Calculate load factor (K)

        Args:
            load_mva: Current load in MVA

        Returns:
            Load factor K (ratio of load to rated)
        """
        return load_mva / self.params.rated_mva

    def calculate_top_oil_rise(self, load_factor: float) -> float:
        """
        Calculate top-oil temperature rise above ambient

        Equation: Δθ_top = Δθ_top_rated × (K^m)

        Args:
            load_factor: Load factor K (ratio of load to rated)

        Returns:
            Top-oil temperature rise in °C
        """
        K = load_factor
        m = self.oil_exponent
        delta_theta_top_rated = self.params.top_oil_rise

        return delta_theta_top_rated * (K**m)

    def calculate_winding_gradient(self, load_factor: float) -> float:
        """
        Calculate winding hot-spot gradient above top-oil

        Equation: Δθ_h = Δθ_h_rated × (K^n)

        Args:
            load_factor: Load factor K (ratio of load to rated)

        Returns:
            Winding gradient in °C
        """
        K = load_factor
        n = self.winding_exponent
        delta_theta_h_rated = self.params.hotspot_gradient

        return delta_theta_h_rated * (K**n)

    def calculate_top_oil_temperature(
        self, ambient_temp: float, load_factor: float
    ) -> float:
        """
        Calculate absolute top-oil temperature

        Args:
            ambient_temp: Ambient temperature in °C
            load_factor: Load factor K

        Returns:
            Top-oil temperature in °C
        """
        delta_theta_top = self.calculate_top_oil_rise(load_factor)
        return ambient_temp + delta_theta_top

    def calculate_hotspot_temperature(
        self, ambient_temp: float, load_factor: float
    ) -> float:
        """
        Calculate hot-spot temperature

        Equation: θ_h = θ_amb + Δθ_top + Δθ_h

        Args:
            ambient_temp: Ambient temperature in °C
            load_factor: Load factor K

        Returns:
            Hot-spot temperature in °C
        """
        delta_theta_top = self.calculate_top_oil_rise(load_factor)
        delta_theta_h = self.calculate_winding_gradient(load_factor)

        return ambient_temp + delta_theta_top + delta_theta_h

    def calculate_transient_hotspot(
        self,
        ambient_temp: float,
        load_factor: float,
        initial_hotspot: float,
        time_minutes: float,
    ) -> float:
        """
        Calculate transient hot-spot temperature with time response

        Uses exponential approach to steady-state temperature

        Args:
            ambient_temp: Ambient temperature in °C
            load_factor: Load factor K
            initial_hotspot: Initial hot-spot temperature in °C
            time_minutes: Time in minutes

        Returns:
            Transient hot-spot temperature in °C
        """
        # Steady-state hotspot
        steady_state_hotspot = self.calculate_hotspot_temperature(
            ambient_temp, load_factor
        )

        # Time constant for winding (dominates short-term response)
        tau = self.winding_time_constant  # minutes

        # Exponential approach
        if time_minutes <= 0:
            return initial_hotspot

        transient_factor = 1 - math.exp(-time_minutes / tau)

        return (
            initial_hotspot
            + (steady_state_hotspot - initial_hotspot) * transient_factor
        )

    def get_cooling_mode_info(self) -> Dict[str, Any]:
        """
        Get information about the current cooling mode

        Returns:
            Dictionary with cooling mode details
        """
        mode = self.params.cooling_mode
        return {
            "mode": mode,
            "description": COOLING_MODES[mode]["description"],
            "top_oil_rise_rated": self.params.top_oil_rise,
            "hotspot_gradient_rated": self.params.hotspot_gradient,
        }

    @staticmethod
    def get_available_cooling_modes() -> Dict[str, Dict[str, Any]]:
        """Get all available cooling modes and their parameters"""
        return COOLING_MODES.copy()


# Import math for transient calculations
import math
