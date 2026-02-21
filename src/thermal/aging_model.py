"""
Insulation Aging Model
Calculate aging acceleration factor based on IEEE C57.91-2011 standard
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional

# IEEE C57.91 aging constants
REFERENCE_HOTSPOT = 110  # °C - reference hot-spot temperature
MIN_LIFE_HOURS = 180000  # hours - 20.5 years per IEEE C57.12.00
AGING_DOUBLING_TEMP = 6  # °C - aging rate doubles every 6°C above reference
ACTIVATION_ENERGY = 15000  # Kelvin - for Arrhenius equation
REFERENCE_TEMP_K = 383  # Kelvin (110°C + 273)


@dataclass
class AgingResult:
    """
    Result dataclass for insulation aging calculation

    Attributes:
        aging_acceleration_factor: Factor by which aging is accelerated
        equivalent_aging_hours: Equivalent hours of aging at reference temperature
        daily_aging_rate: Daily aging rate in percent
        life_consumed_percent: Percentage of insulation life consumed
    """

    aging_acceleration_factor: float
    equivalent_aging_hours: float
    daily_aging_rate: float
    life_consumed_percent: float

    def __str__(self) -> str:
        return (
            f"AgingResult(F_AA={self.aging_acceleration_factor:.4f}, "
            f"equiv_hours={self.equivalent_aging_hours:.2f}, "
            f"daily_rate={self.daily_aging_rate:.4f}%, "
            f"life_consumed={self.life_consumed_percent:.4f}%)"
        )


class AgingModel:
    """
    Insulation Aging Model

    Calculates aging acceleration factor (F_AA) based on IEEE C57.91-2011:
    - Reference temperature: 110°C
    - Every 6°C above reference doubles aging rate
    - Uses Arrhenius equation: F_AA = exp((E/R)(1/T_ref - 1/T))

    The aging rate of cellulose insulation follows the Arrhenius equation,
    with the rate doubling for every 6°C increase in hot-spot temperature
    above the reference of 110°C.

    Attributes:
        reference_hotspot: Reference hot-spot temperature in °C (default: 110)
        min_life_hours: Minimum insulation life in hours (default: 180000)
    """

    def __init__(
        self,
        reference_hotspot: float = REFERENCE_HOTSPOT,
        min_life_hours: float = MIN_LIFE_HOURS,
        activation_energy: float = ACTIVATION_ENERGY,
        use_table: bool = True,
    ):
        """
        Initialize the aging model

        Args:
            reference_hotspot: Reference hot-spot temperature in °C
            min_life_hours: Minimum insulation life in hours
            activation_energy: Activation energy constant for Arrhenius equation
            use_table: Use lookup table for aging factors (more accurate)
        """
        self.reference_hotspot = reference_hotspot
        self.min_life_hours = min_life_hours
        self.activation_energy = activation_energy
        self.use_table = use_table

        # Build aging factor lookup table
        self._aging_table = self._build_aging_table()

    def _build_aging_table(self) -> Dict[int, float]:
        """
        Build aging factor lookup table based on IEEE C57.91

        Returns dictionary mapping temperature (°C) to aging factor
        where F_AA = 1.0 at 120°C (reference in some standards)
        or F_AA = 0.5 at 110°C (reference in C57.91)
        """
        table = {}

        # Generate table from 60°C to 180°C
        for temp in range(60, 181, 5):
            # Using the doubling relationship: F_AA doubles every 6°C above 110°C
            # At 110°C, F_AA = 0.5 (half the rate at 116°C)
            # At 116°C, F_AA = 1.0
            temp_diff = temp - self.reference_hotspot
            if temp_diff <= 0:
                # Below reference: use Arrhenius for low temperatures
                factor = self._calculate_arrhenius_factor(temp)
            else:
                # Above reference: use exponential doubling
                factor = 0.5 * (2 ** (temp_diff / AGING_DOUBLING_TEMP))

            table[temp] = factor

        return table

    def _calculate_arrhenius_factor(self, temperature_c: float) -> float:
        """
        Calculate aging factor using Arrhenius equation

        F_AA = exp((E/R)(1/T_ref - 1/T))

        Args:
            temperature_c: Temperature in Celsius

        Returns:
            Aging acceleration factor
        """
        T_ref_k = self.reference_hotspot + 273
        T_k = temperature_c + 273

        # Activation energy / gas constant
        E_R = self.activation_energy / (8.314)  # J/(mol·K)

        factor = math.exp(E_R * (1 / T_ref_k - 1 / T_k))

        return factor

    def calculate_aging_factor(self, hotspot_temp: float) -> float:
        """
        Calculate aging acceleration factor (F_AA)

        Args:
            hotspot_temp: Hot-spot temperature in °C

        Returns:
            Aging acceleration factor
        """
        if self.use_table:
            # Use lookup table with interpolation
            return self._interpolate_aging_factor(hotspot_temp)
        else:
            # Use Arrhenius equation directly
            return self._calculate_arrhenius_factor(hotspot_temp)

    def _interpolate_aging_factor(self, temperature: float) -> float:
        """
        Interpolate aging factor from lookup table

        Args:
            temperature: Temperature in °C

        Returns:
            Interpolated aging factor
        """
        if temperature <= 60:
            return self._aging_table[60]

        if temperature >= 180:
            return self._aging_table[180]

        # Find surrounding table entries
        temp_low = int(temperature // 5) * 5
        temp_high = temp_low + 5

        if temp_low == temp_high:
            return self._aging_table.get(temp_low, 1.0)

        # Linear interpolation
        factor_low = self._aging_table.get(temp_low, 1.0)
        factor_high = self._aging_table.get(temp_high, 1.0)

        ratio = (temperature - temp_low) / (temp_high - temp_low)

        return factor_low + (factor_high - factor_low) * ratio

    def calculate(
        self, hotspot_temp: float, duration_hours: float = 1.0
    ) -> AgingResult:
        """
        Calculate aging for given hot-spot temperature and duration

        Args:
            hotspot_temp: Hot-spot temperature in °C
            duration_hours: Duration of exposure in hours

        Returns:
            AgingResult with calculated aging parameters
        """
        # Validate input
        if hotspot_temp < -50 or hotspot_temp > 200:
            raise ValueError(
                f"Hot-spot temperature {hotspot_temp}°C is outside valid range"
            )

        if duration_hours < 0:
            raise ValueError("Duration cannot be negative")

        # Calculate aging acceleration factor
        aging_factor = self.calculate_aging_factor(hotspot_temp)

        # Calculate equivalent aging hours at reference temperature
        equivalent_aging_hours = duration_hours * aging_factor

        # Calculate daily aging rate (assuming continuous exposure)
        daily_hours = 24
        daily_aging_hours = daily_hours * aging_factor
        daily_aging_rate = (daily_aging_hours / self.min_life_hours) * 100

        # Calculate life consumed for this duration
        life_consumed_percent = (equivalent_aging_hours / self.min_life_hours) * 100

        return AgingResult(
            aging_acceleration_factor=aging_factor,
            equivalent_aging_hours=equivalent_aging_hours,
            daily_aging_rate=daily_aging_rate,
            life_consumed_percent=life_consumed_percent,
        )

    def calculate_from_load_profile(
        self, hotspot_temps: list, time_intervals: list
    ) -> AgingResult:
        """
        Calculate aging from load profile with varying temperatures

        Args:
            hotspot_temps: List of hot-spot temperatures in °C
            time_intervals: List of time durations in hours (same length as temps)

        Returns:
            AgingResult for the entire profile
        """
        if len(hotspot_temps) != len(time_intervals):
            raise ValueError("hotspot_temps and time_intervals must have same length")

        total_equivalent_hours = 0.0
        total_duration = 0.0

        for temp, duration in zip(hotspot_temps, time_intervals):
            aging_factor = self.calculate_aging_factor(temp)
            total_equivalent_hours += duration * aging_factor
            total_duration += duration

        # Calculate aging rates
        daily_hours = 24
        if total_duration > 0:
            avg_aging_factor = total_equivalent_hours / total_duration
        else:
            avg_aging_factor = 0

        daily_aging_hours = daily_hours * avg_aging_factor
        daily_aging_rate = (daily_aging_hours / self.min_life_hours) * 100

        life_consumed_percent = (total_equivalent_hours / self.min_life_hours) * 100

        return AgingResult(
            aging_acceleration_factor=avg_aging_factor,
            equivalent_aging_hours=total_equivalent_hours,
            daily_aging_rate=daily_aging_rate,
            life_consumed_percent=life_consumed_percent,
        )

    def estimate_remaining_life(self, hotspot_temp: float) -> float:
        """
        Estimate remaining life at given hot-spot temperature

        Args:
            hotspot_temp: Hot-spot temperature in °C

        Returns:
            Remaining life in hours
        """
        aging_factor = self.calculate_aging_factor(hotspot_temp)

        if aging_factor <= 0:
            return float("inf")

        # Remaining life = Total life / aging factor
        remaining_hours = self.min_life_hours / aging_factor

        return remaining_hours

    def get_aging_rate_table(self) -> Dict[int, float]:
        """
        Get the aging factor lookup table

        Returns:
            Dictionary mapping temperature to aging factor
        """
        return self._aging_table.copy()

    def get_life_expectancy(self, hotspot_temp: float) -> Dict[str, float]:
        """
        Get life expectancy information at given temperature

        Args:
            hotspot_temp: Hot-spot temperature in °C

        Returns:
            Dictionary with life expectancy in various units
        """
        remaining_hours = self.estimate_remaining_life(hotspot_temp)
        aging_factor = self.calculate_aging_factor(hotspot_temp)

        return {
            "remaining_hours": remaining_hours,
            "remaining_years": remaining_hours / 8760,  # hours per year
            "aging_factor": aging_factor,
            "reference_hours": self.min_life_hours,
            "reference_years": self.min_life_hours / 8760,
        }
