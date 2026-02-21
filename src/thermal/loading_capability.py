"""
Dynamic Loading Capability Calculator
Calculate permissible loading limits based on IEEE C57.91-2011 standard
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Loading limits per IEEE C57.91
NORMAL_LOADING_LIMIT = 1.0  # 100% of rated
EMERGENCY_LOADING_LIMIT = 1.2  # 120% of rated
SHORT_TERM_EMERGENCY_LIMIT = 1.5  # 150% of rated

# Temperature limits
HOTSPOT_NORMAL_LIMIT = 110  # °C
HOTSPOT_EMERGENCY_LIMIT = 120  # °C
HOTSPOT_SHORT_TERM_LIMIT = 140  # °C

# Time constants (minutes)
TOP_OIL_TIME_CONSTANT = 150
WINDING_TIME_CONSTANT = 5


@dataclass
class LoadingResult:
    """
    Result dataclass for loading capability calculation

    Attributes:
        normal_loading_mva: Normal loading capability in MVA
        emergency_loading_mva: Emergency loading capability in MVA
        short_term_loading_mva: Short-term emergency loading in MVA
        time_limit_hours: Maximum time for short-term loading in hours
        constraints: Dictionary of constraints and their values
    """

    normal_loading_mva: float
    emergency_loading_mva: float
    short_term_loading_mva: float
    time_limit_hours: float
    constraints: Dict[str, float]

    def __str__(self) -> str:
        return (
            f"LoadingResult(normal={self.normal_loading_mva:.2f}MVA, "
            f"emergency={self.emergency_loading_mva:.2f}MVA, "
            f"short_term={self.short_term_loading_mva:.2f}MVA, "
            f"time_limit={self.time_limit_hours:.1f}h)"
        )


class LoadingCapability:
    """
    Dynamic Loading Capability Calculator

    Calculates permissible loading limits for transformers based on:
    - IEEE C57.91-2011 standard
    - Hot-spot temperature constraints
    - Cooling mode capabilities
    - Ambient temperature effects

    Loading categories:
    - Normal loading: Continuous operation at rated load (hot-spot ≤ 110°C)
    - Long-time emergency: 120% load for extended periods (hot-spot ≤ 120°C)
    - Short-time emergency: Up to 150% for limited duration (hot-spot ≤ 140°C)

    Attributes:
        rated_mva: Transformer rated power in MVA
        cooling_mode: Cooling mode (ONAN, ONAF, OFAF, ODAF)
        top_oil_rise: Rated top-oil temperature rise in °C
        hotspot_gradient: Rated hot-spot gradient in °C
    """

    # Cooling mode parameters
    COOLING_PARAMS = {
        "ONAN": {"top_oil_rise": 45, "hotspot_gradient": 65, "max_normal": 1.0},
        "ONAF": {"top_oil_rise": 40, "hotspot_gradient": 55, "max_normal": 1.15},
        "OFAF": {"top_oil_rise": 35, "hotspot_gradient": 50, "max_normal": 1.25},
        "ODAF": {"top_oil_rise": 30, "hotspot_gradient": 45, "max_normal": 1.35},
    }

    def __init__(
        self,
        rated_mva: float,
        cooling_mode: str = "ONAN",
        top_oil_rise: Optional[float] = None,
        hotspot_gradient: Optional[float] = None,
        oil_exponent: float = 0.8,
        winding_exponent: float = 0.8,
    ):
        """
        Initialize the loading capability calculator

        Args:
            rated_mva: Rated power in MVA
            cooling_mode: Cooling mode (ONAN, ONAF, OFAF, ODAF)
            top_oil_rise: Rated top-oil temperature rise in °C
            hotspot_gradient: Rated hot-spot gradient in °C
            oil_exponent: Oil exponent for thermal calculation
            winding_exponent: Winding exponent for thermal calculation
        """
        self.rated_mva = rated_mva
        self.cooling_mode = cooling_mode.upper()

        if self.cooling_mode not in self.COOLING_PARAMS:
            raise ValueError(f"Invalid cooling mode: {self.cooling_mode}")

        params = self.COOLING_PARAMS[self.cooling_mode]
        self.top_oil_rise = top_oil_rise or params["top_oil_rise"]
        self.hotspot_gradient = hotspot_gradient or params["hotspot_gradient"]
        self.max_normal_load = params["max_normal"]

        self.oil_exponent = oil_exponent
        self.winding_exponent = winding_exponent

        # Temperature limits
        self.hotspot_normal_limit = HOTSPOT_NORMAL_LIMIT
        self.hotspot_emergency_limit = HOTSPOT_EMERGENCY_LIMIT
        self.hotspot_short_term_limit = HOTSPOT_SHORT_TERM_LIMIT

    def _calculate_load_for_hotspot(
        self, ambient_temp: float, hotspot_limit: float
    ) -> float:
        """
        Calculate maximum load factor for given hot-spot limit

        Solves: θ_h = θ_amb + Δθ_top(K) + Δθ_h(K) ≤ hotspot_limit

        Where:
        - Δθ_top(K) = Δθ_top_rated × K^m
        - Δθ_h(K) = Δθ_h_rated × K^n

        Args:
            ambient_temp: Ambient temperature in °C
            hotspot_limit: Maximum hot-spot temperature in °C

        Returns:
            Load factor (ratio of rated)
        """
        # Available temperature budget
        available_temp = hotspot_limit - ambient_temp

        if available_temp <= 0:
            return 0.0

        # Iterative solution for load factor
        # Start with initial guess
        K = 1.0

        for _ in range(50):
            delta_top = self.top_oil_rise * (K**self.oil_exponent)
            delta_h = self.hotspot_gradient * (K**self.winding_exponent)

            current_hotspot = ambient_temp + delta_top + delta_h

            if abs(current_hotspot - hotspot_limit) < 0.1:
                break

            # Adjust K
            if current_hotspot > hotspot_limit:
                K *= 0.95
            else:
                K *= 1.02

        return max(0.0, min(K, 2.0))

    def calculate(self, ambient_temp: float) -> LoadingResult:
        """
        Calculate loading capabilities at given ambient temperature

        Args:
            ambient_temp: Ambient temperature in °C

        Returns:
            LoadingResult with loading capabilities
        """
        # Normal loading (hot-spot ≤ 110°C)
        normal_load_factor = self._calculate_load_for_hotspot(
            ambient_temp, self.hotspot_normal_limit
        )
        normal_mva = normal_load_factor * self.rated_mva

        # Emergency loading (hot-spot ≤ 120°C)
        emergency_load_factor = self._calculate_load_for_hotspot(
            ambient_temp, self.hotspot_emergency_limit
        )
        emergency_mva = emergency_load_factor * self.rated_mva

        # Short-term emergency loading (hot-spot ≤ 140°C)
        short_term_load_factor = self._calculate_load_for_hotspot(
            ambient_temp, self.hotspot_short_term_limit
        )
        short_term_mva = short_term_load_factor * self.rated_mva

        # Time limit for short-term loading
        # Based on winding time constant
        time_limit = self._calculate_time_limit(ambient_temp, short_term_load_factor)

        constraints = {
            "hotspot_normal_limit": self.hotspot_normal_limit,
            "hotspot_emergency_limit": self.hotspot_emergency_limit,
            "hotspot_short_term_limit": self.hotspot_short_term_limit,
            "available_temp_normal": self.hotspot_normal_limit - ambient_temp,
            "available_temp_emergency": self.hotspot_emergency_limit - ambient_temp,
            "available_temp_short_term": self.hotspot_short_term_limit - ambient_temp,
            "top_oil_rise": self.top_oil_rise,
            "hotspot_gradient": self.hotspot_gradient,
        }

        return LoadingResult(
            normal_loading_mva=normal_mva,
            emergency_loading_mva=emergency_mva,
            short_term_loading_mva=short_term_mva,
            time_limit_hours=time_limit,
            constraints=constraints,
        )

    def _calculate_time_limit(self, ambient_temp: float, load_factor: float) -> float:
        """
        Calculate time limit for short-term loading

        Based on reaching hot-spot limit from current temperature

        Args:
            ambient_temp: Ambient temperature in °C
            load_factor: Load factor

        Returns:
            Time limit in hours
        """
        # Calculate steady-state hot-spot
        delta_top = self.top_oil_rise * (load_factor**self.oil_exponent)
        delta_h = self.hotspot_gradient * (load_factor**self.winding_exponent)
        steady_hotspot = ambient_temp + delta_top + delta_h

        # If below limit, no time limit
        if steady_hotspot <= self.hotspot_short_term_limit:
            return 24.0  # Assume 24 hours is safe

        # Initial hotspot at normal load
        normal_delta_top = self.top_oil_rise * (1.0**self.oil_exponent)
        normal_delta_h = self.hotspot_gradient * (1.0**self.winding_exponent)
        initial_hotspot = ambient_temp + normal_delta_top + normal_delta_h

        # Time constant for approach to steady state
        tau = WINDING_TIME_CONSTANT

        # Time to reach limit
        if steady_hotspot <= initial_hotspot:
            return float("inf")

        # Solve: T(t) = T_initial + (T_steady - T_initial) * (1 - exp(-t/tau))
        # For T(t) = T_limit
        ratio = (self.hotspot_short_term_limit - initial_hotspot) / (
            steady_hotspot - initial_hotspot
        )

        if ratio <= 0:
            return 0.0

        time_minutes = -tau * math.log(1 - ratio)

        return time_minutes / 60.0  # Convert to hours

    def calculate_for_duration(
        self, ambient_temp: float, load_mva: float, duration_hours: float
    ) -> Dict[str, any]:
        """
        Check if load is permissible for given duration

        Args:
            ambient_temp: Ambient temperature in °C
            load_mva: Proposed load in MVA
            duration_hours: Duration of load in hours

        Returns:
            Dictionary with check results
        """
        load_factor = load_mva / self.rated_mva

        # Calculate hot-spot for this load
        delta_top = self.top_oil_rise * (load_factor**self.oil_exponent)
        delta_h = self.hotspot_gradient * (load_factor**self.winding_exponent)

        # Check if steady-state is within limits
        steady_hotspot = ambient_temp + delta_top + delta_h

        # Determine applicable limit based on duration
        if duration_hours <= 0.5:
            # Short-term: up to 140°C
            limit = self.hotspot_short_term_limit
            category = "short_term"
        elif duration_hours <= 8:
            # Emergency: up to 120°C
            limit = self.hotspot_emergency_limit
            category = "emergency"
        else:
            # Normal: up to 110°C
            limit = self.hotspot_normal_limit
            category = "normal"

        is_permissible = steady_hotspot <= limit

        result = {
            "load_factor": load_factor,
            "hotspot_temp": steady_hotspot,
            "limit": limit,
            "category": category,
            "is_permissible": is_permissible,
            "margin": limit - steady_hotspot,
        }

        return result

    def get_loading_curve(
        self, ambient_temp: float, max_load_factor: float = 1.5, points: int = 20
    ) -> List[Tuple[float, float]]:
        """
        Generate loading curve (hot-spot vs load factor)

        Args:
            ambient_temp: Ambient temperature in °C
            max_load_factor: Maximum load factor to calculate
            points: Number of points in curve

        Returns:
            List of (load_factor, hotspot_temp) tuples
        """
        curve = []

        for i in range(points + 1):
            load_factor = (i / points) * max_load_factor

            delta_top = self.top_oil_rise * (load_factor**self.oil_exponent)
            delta_h = self.hotspot_gradient * (load_factor**self.winding_exponent)
            hotspot = ambient_temp + delta_top + delta_h

            curve.append((load_factor, hotspot))

        return curve

    def find_max_load(self, ambient_temp: float, duration_hours: float = 24.0) -> float:
        """
        Find maximum permissible load for given duration

        Args:
            ambient_temp: Ambient temperature in °C
            duration_hours: Operating duration in hours

        Returns:
            Maximum load in MVA
        """
        # Determine applicable hot-spot limit
        if duration_hours <= 0.5:
            limit = self.hotspot_short_term_limit
        elif duration_hours <= 8:
            limit = self.hotspot_emergency_limit
        else:
            limit = self.hotspot_normal_limit

        # Calculate load factor
        load_factor = self._calculate_load_for_hotspot(ambient_temp, limit)

        return load_factor * self.rated_mva

    def get_thermal_limits(self) -> Dict[str, float]:
        """Get thermal limits for the transformer"""
        return {
            "hotspot_normal": self.hotspot_normal_limit,
            "hotspot_emergency": self.hotspot_emergency_limit,
            "hotspot_short_term": self.hotspot_short_term_limit,
            "top_oil_rise": self.top_oil_rise,
            "hotspot_gradient": self.hotspot_gradient,
        }

    def get_loading_limits(self, ambient_temp: float) -> Dict[str, float]:
        """
        Get loading limits at given ambient temperature

        Args:
            ambient_temp: Ambient temperature in °C

        Returns:
            Dictionary with loading limits in MVA
        """
        result = self.calculate(ambient_temp)

        return {
            "normal_mva": result.normal_loading_mva,
            "emergency_mva": result.emergency_loading_mva,
            "short_term_mva": result.short_term_loading_mva,
            "normal_percent": (result.normal_loading_mva / self.rated_mva) * 100,
            "emergency_percent": (result.emergency_loading_mva / self.rated_mva) * 100,
            "short_term_percent": (result.short_term_loading_mva / self.rated_mva)
            * 100,
        }
