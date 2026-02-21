"""
Loss-of-Life Calculator
Calculate cumulative loss-of-life based on IEEE C57.91-2011 standard
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

# Default constants
MIN_LIFE_EXPECTANCY_HOURS = 180000  # 20.5 years per IEEE C57.12.00
MIN_LIFE_EXPECTANCY_YEARS = MIN_LIFE_EXPECTANCY_HOURS / 8760  # 24*365


@dataclass
class LossOfLifeResult:
    """
    Result dataclass for loss-of-life calculation

    Attributes:
        total_loss_of_life_hours: Total loss-of-life in hours
        total_loss_of_life_percent: Total loss-of-life in percentage
        remaining_life_hours: Remaining useful life in hours
        remaining_life_years: Remaining useful life in years
        estimated_end_of_life_date: Estimated end-of-life date (if applicable)
    """

    total_loss_of_life_hours: float
    total_loss_of_life_percent: float
    remaining_life_hours: float
    remaining_life_years: float
    estimated_end_of_life_date: Optional[datetime]

    def __str__(self) -> str:
        date_str = (
            self.estimated_end_of_life_date.strftime("%Y-%m-%d")
            if self.estimated_end_of_life_date
            else "N/A"
        )
        return (
            f"LossOfLifeResult(loss_hours={self.total_loss_of_life_hours:.1f}, "
            f"loss_percent={self.total_loss_of_life_percent:.2f}%, "
            f"remaining_hours={self.remaining_life_hours:.1f}, "
            f"remaining_years={self.remaining_life_years:.2f}, "
            f"eol_date={date_str})"
        )


class LossOfLifeCalculator:
    """
    Loss-of-Life Calculator

    Calculates cumulative loss-of-life for transformers based on:
    - IEEE C57.91-2011 standard
    - Normal life: 180,000 hours (20.5 years)
    - Accumulated loss-of-life over time
    - Remaining useful life estimate

    Loss-of-life is calculated by integrating aging acceleration factors
    over time: L = ∫(F_AA(t) / T_total) dt × 100%

    Attributes:
        min_life_hours: Minimum expected life in hours
        reference_date: Reference date for age calculation
    """

    def __init__(
        self,
        min_life_hours: float = MIN_LIFE_EXPECTANCY_HOURS,
        reference_date: Optional[datetime] = None,
    ):
        """
        Initialize the loss-of-life calculator

        Args:
            min_life_hours: Minimum expected life in hours
            reference_date: Reference date for age calculation (default: current time)
        """
        self.min_life_hours = min_life_hours
        self.reference_date = reference_date or datetime.now()

    def calculate_single_period(
        self, hotspot_temp: float, duration_hours: float, aging_model: "AgingModel"
    ) -> float:
        """
        Calculate loss-of-life for a single period

        Args:
            hotspot_temp: Hot-spot temperature in °C
            duration_hours: Duration of exposure in hours
            aging_model: AgingModel instance for calculating F_AA

        Returns:
            Loss-of-life as percentage
        """
        # Calculate aging factor
        aging_factor = aging_model.calculate_aging_factor(hotspot_temp)

        # Calculate loss-of-life percentage
        loss_of_life_percent = (
            aging_factor * duration_hours / self.min_life_hours
        ) * 100

        return loss_of_life_percent

    def calculate_load_profile(
        self,
        load_profile: List[Tuple[float, float]],
        ambient_temp: float,
        thermal_model: "IEEEC57_91",
        aging_model: "AgingModel",
    ) -> LossOfLifeResult:
        """
        Calculate loss-of-life from load profile

        Args:
            load_profile: List of (load_percent, duration_hours) tuples
            ambient_temp: Ambient temperature in °C
            thermal_model: IEEEC57_91 thermal model
            aging_model: AgingModel instance

        Returns:
            LossOfLifeResult with total and remaining life
        """
        total_loss_percent = 0.0
        total_duration = 0.0

        for load_percent, duration in load_profile:
            # Calculate load factor
            load_factor = load_percent / 100.0

            # Calculate hot-spot temperature
            hotspot_temp = thermal_model.calculate_hotspot_temperature(
                ambient_temp, load_factor
            )

            # Calculate loss-of-life for this period
            loss_percent = self.calculate_single_period(
                hotspot_temp, duration, aging_model
            )

            total_loss_percent += loss_percent
            total_duration += duration

        # Calculate remaining life
        remaining_life_percent = max(0, 100 - total_loss_percent)
        remaining_hours = (remaining_life_percent / 100) * self.min_life_hours
        remaining_years = remaining_hours / 8760

        # Estimate end-of-life date
        average_aging_rate = (
            total_loss_percent / total_duration if total_duration > 0 else 0
        )
        if average_aging_rate > 0:
            years_remaining = remaining_life_percent / average_aging_rate
            eol_date = self.reference_date + timedelta(days=years_remaining * 365)
        else:
            eol_date = None

        return LossOfLifeResult(
            total_loss_of_life_hours=(total_loss_percent / 100) * self.min_life_hours,
            total_loss_of_life_percent=total_loss_percent,
            remaining_life_hours=remaining_hours,
            remaining_life_years=remaining_years,
            estimated_end_of_life_date=eol_date,
        )

    def calculate_annual_loss(
        self,
        average_load_factor: float,
        ambient_temp: float,
        thermal_model: "IEEEC57_91",
        aging_model: "AgingModel",
    ) -> LossOfLifeResult:
        """
        Calculate annual loss-of-life

        Args:
            average_load_factor: Average load factor (0-1)
            ambient_temp: Average ambient temperature in °C
            thermal_model: IEEEC57_91 thermal model
            aging_model: AgingModel instance

        Returns:
            LossOfLifeResult for one year
        """
        annual_hours = 8760
        hotspot_temp = thermal_model.calculate_hotspot_temperature(
            ambient_temp, average_load_factor
        )

        loss_percent = self.calculate_single_period(
            hotspot_temp, annual_hours, aging_model
        )

        return LossOfLifeResult(
            total_loss_of_life_hours=(loss_percent / 100) * self.min_life_hours,
            total_loss_of_life_percent=loss_percent,
            remaining_life_hours=(100 - loss_percent) / 100 * self.min_life_hours,
            remaining_life_years=(100 - loss_percent) / 100 * MIN_LIFE_EXPECTANCY_YEARS,
            estimated_end_of_life_date=self.reference_date
            + timedelta(days=365 * (100 - loss_percent) / loss_percent),
        )

    def calculate_from_age(
        self,
        current_age_hours: float,
        hotspot_temp_history: List[Tuple[float, float]],
        aging_model: "AgingModel",
    ) -> LossOfLifeResult:
        """
        Calculate loss-of-life from transformer age and temperature history

        Args:
            current_age_hours: Total operating hours
            hotspot_temp_history: List of (temperature, duration_hours) tuples
            aging_model: AgingModel instance

        Returns:
            LossOfLifeResult with calculated values
        """
        total_loss_percent = 0.0

        for temp, duration in hotspot_temp_history:
            loss_percent = self.calculate_single_period(temp, duration, aging_model)
            total_loss_percent += loss_percent

        # Calculate remaining life
        remaining_life_percent = max(0, 100 - total_loss_percent)
        remaining_hours = (remaining_life_percent / 100) * self.min_life_hours
        remaining_years = remaining_hours / 8760

        # Estimate end-of-life date
        if total_loss_percent > 0:
            average_aging_rate = total_loss_percent / current_age_hours
            eol_date = self.reference_date + timedelta(
                hours=(remaining_life_percent / average_aging_rate)
            )
        else:
            eol_date = None

        return LossOfLifeResult(
            total_loss_of_life_hours=(total_loss_percent / 100) * self.min_life_hours,
            total_loss_of_life_percent=total_loss_percent,
            remaining_life_hours=remaining_hours,
            remaining_life_years=remaining_years,
            estimated_end_of_life_date=eol_date,
        )

    def calculate_remaining_life_at_temp(
        self,
        current_loss_percent: float,
        future_hotspot_temp: float,
        aging_model: "AgingModel",
    ) -> LossOfLifeResult:
        """
        Calculate remaining life assuming constant future temperature

        Args:
            current_loss_percent: Current loss-of-life percentage (0-100)
            future_hotspot_temp: Future operating hot-spot temperature
            aging_model: AgingModel instance

        Returns:
            LossOfLifeResult with remaining life
        """
        if current_loss_percent < 0 or current_loss_percent > 100:
            raise ValueError("Current loss percentage must be between 0 and 100")

        remaining_life_percent = max(0, 100 - current_loss_percent)
        remaining_hours = (remaining_life_percent / 100) * self.min_life_hours

        # Calculate aging factor for future temperature
        aging_factor = aging_model.calculate_aging_factor(future_hotspot_temp)

        # Time to reach 100% loss at future temperature
        time_to_eol = remaining_hours / aging_factor

        eol_date = self.reference_date + timedelta(hours=time_to_eol)

        return LossOfLifeResult(
            total_loss_of_life_hours=(current_loss_percent / 100) * self.min_life_hours,
            total_loss_of_life_percent=current_loss_percent,
            remaining_life_hours=remaining_hours,
            remaining_life_years=remaining_hours / 8760,
            estimated_end_of_life_date=eol_date,
        )

    def get_aging_acceleration_table(
        self,
        aging_model: "AgingModel",
        min_temp: float = 80,
        max_temp: float = 160,
        step: float = 5,
    ) -> List[Tuple[float, float]]:
        """
        Generate aging acceleration table for temperature range

        Args:
            aging_model: AgingModel instance
            min_temp: Minimum temperature in °C
            max_temp: Maximum temperature in °C
            step: Temperature step in °C

        Returns:
            List of (temperature, aging_acceleration) tuples
        """
        table = []
        temp = min_temp

        while temp <= max_temp:
            factor = aging_model.calculate_aging_factor(temp)
            table.append((temp, factor))
            temp += step

        return table

    def estimate_from_ambient_profile(
        self,
        ambient_profile: List[Tuple[float, float]],
        load_factor: float,
        thermal_model: "IEEEC57_91",
        aging_model: "AgingModel",
    ) -> LossOfLifeResult:
        """
        Estimate loss-of-life from ambient temperature profile

        Args:
            ambient_profile: List of (ambient_temp, duration_hours) tuples
            load_factor: Load factor (0-1)
            thermal_model: IEEEC57_91 thermal model
            aging_model: AgingModel instance

        Returns:
            LossOfLifeResult for the profile
        """
        total_loss_percent = 0.0
        total_duration = 0.0

        for ambient_temp, duration in ambient_profile:
            hotspot_temp = thermal_model.calculate_hotspot_temperature(
                ambient_temp, load_factor
            )
            loss_percent = self.calculate_single_period(
                hotspot_temp, duration, aging_model
            )
            total_loss_percent += loss_percent
            total_duration += duration

        remaining_life_percent = max(0, 100 - total_loss_percent)
        remaining_hours = (remaining_life_percent / 100) * self.min_life_hours
        remaining_years = remaining_hours / 8760

        if total_loss_percent > 0:
            average_aging_rate = total_loss_percent / total_duration
            eol_date = self.reference_date + timedelta(
                days=(remaining_life_percent / average_aging_rate) / 24
            )
        else:
            eol_date = None

        return LossOfLifeResult(
            total_loss_of_life_hours=(total_loss_percent / 100) * self.min_life_hours,
            total_loss_of_life_percent=total_loss_percent,
            remaining_life_hours=remaining_hours,
            remaining_life_years=remaining_years,
            estimated_end_of_life_date=eol_date,
        )
