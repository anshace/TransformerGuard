"""
Hot-Spot Temperature Calculator
Calculate hot-spot temperature based on IEEE C57.91-2011 standard
"""

import math
from dataclasses import dataclass
from typing import Optional

# Hot-spot temperature limits per IEEE C57.91
HOTSPOT_TEMPERATURE_LIMIT = 110  # °C - maximum continuous hot-spot temperature
HOTSPOT_EMERGENCY_LIMIT = 120  # °C - emergency hot-spot temperature limit


@dataclass
class HotSpotResult:
    """
    Result dataclass for hot-spot temperature calculation

    Attributes:
        hotspot_temp: Hot-spot temperature in °C
        top_oil_temp: Top-oil temperature in °C
        winding_gradient: Winding hot-spot gradient in °C
        is_over_limit: Whether hot-spot exceeds 110°C limit
        margin_to_limit: Margin to 110°C limit in °C (negative if over limit)
    """

    hotspot_temp: float
    top_oil_temp: float
    winding_gradient: float
    is_over_limit: bool
    margin_to_limit: float

    def __str__(self) -> str:
        status = "OVER LIMIT" if self.is_over_limit else "Normal"
        return (
            f"HotSpotResult(hotspot={self.hotspot_temp:.1f}°C, "
            f"top_oil={self.top_oil_temp:.1f}°C, "
            f"gradient={self.winding_gradient:.1f}°C, "
            f"status={status}, margin={self.margin_to_limit:.1f}°C)"
        )


class HotSpotCalculator:
    """
    Hot-Spot Temperature Calculator

    Calculates transformer hot-spot temperature based on:
    - Ambient temperature
    - Load current (as percentage of rated)
    - Cooling mode
    - Time constant for transient analysis

    Uses IEEE C57.91-2011 equations:
    - Δθ_top = Δθ_top_rated × (K^m)
    - Δθ_h = Δθ_h_rated × (K^n)
    - θ_h = θ_amb + Δθ_top + Δθ_h

    Where:
    - K = load factor (load / rated)
    - m = oil exponent (~0.8)
    - n = winding exponent (~0.8)
    """

    # Default thermal parameters
    DEFAULT_PARAMS = {
        "ONAN": {"top_oil_rise": 45, "hotspot_gradient": 65},
        "ONAF": {"top_oil_rise": 40, "hotspot_gradient": 55},
        "OFAF": {"top_oil_rise": 35, "hotspot_gradient": 50},
        "ODAF": {"top_oil_rise": 30, "hotspot_gradient": 45},
    }

    def __init__(
        self,
        cooling_mode: str = "ONAN",
        top_oil_rise: Optional[float] = None,
        hotspot_gradient: Optional[float] = None,
        oil_exponent: float = 0.8,
        winding_exponent: float = 0.8,
        top_oil_time_constant: float = 150,
        winding_time_constant: float = 5,
    ):
        """
        Initialize the hot-spot calculator

        Args:
            cooling_mode: Cooling mode (ONAN, ONAF, OFAF, ODAF)
            top_oil_rise: Rated top-oil temperature rise in °C
            hotspot_gradient: Rated winding hot-spot gradient in °C
            oil_exponent: Oil exponent (m), typically 0.8
            winding_exponent: Winding exponent (n), typically 0.8
            top_oil_time_constant: Top-oil thermal time constant in minutes
            winding_time_constant: Winding thermal time constant in minutes
        """
        self.cooling_mode = cooling_mode.upper()

        if self.cooling_mode not in self.DEFAULT_PARAMS:
            raise ValueError(
                f"Invalid cooling mode: {self.cooling_mode}. "
                f"Must be one of {list(self.DEFAULT_PARAMS.keys())}"
            )

        # Use provided values or defaults from cooling mode
        self.top_oil_rise = (
            top_oil_rise or self.DEFAULT_PARAMS[cooling_mode]["top_oil_rise"]
        )
        self.hotspot_gradient = (
            hotspot_gradient or self.DEFAULT_PARAMS[cooling_mode]["hotspot_gradient"]
        )

        self.oil_exponent = oil_exponent
        self.winding_exponent = winding_exponent
        self.top_oil_time_constant = top_oil_time_constant
        self.winding_time_constant = winding_time_constant

        # Temperature limit
        self.temperature_limit = HOTSPOT_TEMPERATURE_LIMIT

    def calculate_load_factor(self, load_percent: float) -> float:
        """
        Calculate load factor from load percentage

        Args:
            load_percent: Load as percentage of rated (e.g., 80 for 80%)

        Returns:
            Load factor K (ratio)
        """
        return load_percent / 100.0

    def calculate_top_oil_rise(self, load_factor: float) -> float:
        """
        Calculate top-oil temperature rise

        Equation: Δθ_top = Δθ_top_rated × (K^m)

        Args:
            load_factor: Load factor K

        Returns:
            Top-oil temperature rise in °C
        """
        return self.top_oil_rise * (load_factor**self.oil_exponent)

    def calculate_winding_gradient(self, load_factor: float) -> float:
        """
        Calculate winding hot-spot gradient

        Equation: Δθ_h = Δθ_h_rated × (K^n)

        Args:
            load_factor: Load factor K

        Returns:
            Winding gradient in °C
        """
        return self.hotspot_gradient * (load_factor**self.winding_exponent)

    def calculate(
        self,
        ambient_temp: float,
        load_percent: float,
        initial_top_oil: Optional[float] = None,
    ) -> HotSpotResult:
        """
        Calculate hot-spot temperature

        Args:
            ambient_temp: Ambient temperature in °C
            load_percent: Load as percentage of rated (e.g., 80 for 80%)
            initial_top_oil: Initial top-oil temperature for transient calculation

        Returns:
            HotSpotResult with calculated temperatures
        """
        # Validate inputs
        if ambient_temp < -50 or ambient_temp > 60:
            raise ValueError(
                f"Ambient temperature {ambient_temp}°C is outside valid range (-50 to 60°C)"
            )

        if load_percent < 0 or load_percent > 300:
            raise ValueError(
                f"Load percent {load_percent}% is outside valid range (0 to 300%)"
            )

        # Calculate load factor
        load_factor = self.calculate_load_factor(load_percent)

        # Calculate temperature rises
        top_oil_rise = self.calculate_top_oil_rise(load_factor)
        winding_gradient = self.calculate_winding_gradient(load_factor)

        # Calculate absolute temperatures
        if initial_top_oil is not None:
            # Use initial top-oil for transient calculation
            top_oil_temp = (
                initial_top_oil + top_oil_rise - (initial_top_oil - ambient_temp)
            )
            top_oil_temp = max(
                ambient_temp + top_oil_rise * 0.1, top_oil_temp
            )  # Approximate transient
        else:
            top_oil_temp = ambient_temp + top_oil_rise

        hotspot_temp = top_oil_temp + winding_gradient

        # Check limits
        is_over_limit = hotspot_temp > self.temperature_limit
        margin_to_limit = self.temperature_limit - hotspot_temp

        return HotSpotResult(
            hotspot_temp=hotspot_temp,
            top_oil_temp=top_oil_temp,
            winding_gradient=winding_gradient,
            is_over_limit=is_over_limit,
            margin_to_limit=margin_to_limit,
        )

    def calculate_transient(
        self,
        ambient_temp: float,
        load_percent: float,
        duration_minutes: float,
        initial_hotspot: Optional[float] = None,
    ) -> HotSpotResult:
        """
        Calculate transient hot-spot temperature

        Accounts for thermal time constants for short-term loading analysis

        Args:
            ambient_temp: Ambient temperature in °C
            load_percent: Load as percentage of rated
            duration_minutes: Duration of load in minutes
            initial_hotspot: Initial hot-spot temperature (defaults to ambient + 20)

        Returns:
            HotSpotResult with transient temperatures
        """
        # Get steady-state result first
        steady_state = self.calculate(ambient_temp, load_percent)

        if initial_hotspot is None:
            # Estimate initial hotspot from ambient
            initial_hotspot = ambient_temp + 20

        # Calculate time-based correction using exponential approach
        # Winding time constant dominates short-term response
        tau = self.winding_time_constant

        if duration_minutes <= 0:
            transient_factor = 0.0
        else:
            transient_factor = 1 - math.exp(-duration_minutes / tau)

        # Interpolate between initial and steady-state
        transient_hotspot = (
            initial_hotspot
            + (steady_state.hotspot_temp - initial_hotspot) * transient_factor
        )

        # Recalculate other temperatures proportionally
        top_oil_ratio = (
            transient_hotspot / steady_state.hotspot_temp
            if steady_state.hotspot_temp > 0
            else 1
        )
        transient_top_oil = steady_state.top_oil_temp * top_oil_ratio
        transient_gradient = transient_hotspot - transient_top_oil

        # Check limits
        is_over_limit = transient_hotspot > self.temperature_limit
        margin_to_limit = self.temperature_limit - transient_hotspot

        return HotSpotResult(
            hotspot_temp=transient_hotspot,
            top_oil_temp=transient_top_oil,
            winding_gradient=transient_gradient,
            is_over_limit=is_over_limit,
            margin_to_limit=margin_to_limit,
        )

    def calculate_time_to_limit(
        self, ambient_temp: float, load_percent: float, initial_hotspot: float
    ) -> Optional[float]:
        """
        Calculate time required to reach hot-spot limit

        Args:
            ambient_temp: Ambient temperature in °C
            load_percent: Load as percentage of rated
            initial_hotspot: Starting hot-spot temperature in °C

        Returns:
            Time in minutes to reach limit, or None if already over limit
        """
        steady_state = self.calculate(ambient_temp, load_percent)

        if steady_state.hotspot_temp <= self.temperature_limit:
            return None  # Will never reach limit

        if initial_hotspot >= self.temperature_limit:
            return 0.0  # Already over limit

        # Calculate time to reach limit using exponential approach
        tau = self.winding_time_constant

        # Solve: T(t) = T_initial + (T_steady - T_initial) * (1 - exp(-t/tau))
        # For T(t) = T_limit:
        # exp(-t/tau) = 1 - (T_limit - T_initial) / (T_steady - T_initial)

        delta_T = steady_state.hotspot_temp - initial_hotspot
        if delta_T <= 0:
            return None

        ratio = (self.temperature_limit - initial_hotspot) / delta_T

        if ratio >= 1:
            return None

        time_minutes = -tau * math.log(1 - ratio)

        return time_minutes

    def get_thermal_params(self) -> dict:
        """Get current thermal parameters"""
        return {
            "cooling_mode": self.cooling_mode,
            "top_oil_rise": self.top_oil_rise,
            "hotspot_gradient": self.hotspot_gradient,
            "oil_exponent": self.oil_exponent,
            "winding_exponent": self.winding_exponent,
            "top_oil_time_constant": self.top_oil_time_constant,
            "winding_time_constant": self.winding_time_constant,
            "temperature_limit": self.temperature_limit,
        }
