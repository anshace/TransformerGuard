"""
Gas Trend Forecaster
Forecasts gas concentration trends using linear and exponential methods
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# IEEE C57.104 key gas thresholds
IEEE_THRESHOLDS = {
    "H2": 100,  # ppm - Hydrogen
    "CH4": 120,  # ppm - Methane
    "C2H2": 2,  # ppm - Acetylene
    "C2H4": 50,  # ppm - Ethylene
    "C2H6": 65,  # ppm - Ethane
    "CO": 350,  # ppm - Carbon Monoxide
    "CO2": 2500,  # ppm - Carbon Dioxide
}

# All key gases
KEY_GASES = list(IEEE_THRESHOLDS.keys())

# Total Dissolved Combustible Gas (TDCG) threshold
TDCG_THRESHOLD = 7200  # ppm


@dataclass
class ForecastResult:
    """
    Result of gas trend forecasting

    Attributes:
        gas_name: Name of the gas
        current_value: Current gas concentration (ppm)
        forecast_30_days: Forecasted value at 30 days (ppm)
        forecast_60_days: Forecasted value at 60 days (ppm)
        forecast_90_days: Forecasted value at 90 days (ppm)
        will_exceed_threshold: Whether threshold will be exceeded
        days_to_threshold: Days until threshold exceeded (None if not predicted)
        trend_direction: Trend direction (INCREASING, STABLE, DECREASING)
        confidence: Confidence level (0.0-1.0)
    """

    gas_name: str
    current_value: float
    forecast_30_days: float
    forecast_60_days: float
    forecast_90_days: float
    will_exceed_threshold: bool
    days_to_threshold: Optional[int]
    trend_direction: str
    confidence: float = 0.75

    def __str__(self) -> str:
        return (
            f"ForecastResult({self.gas_name}: {self.current_value:.1f} -> "
            f"{self.forecast_90_days:.1f} ppm, trend={self.trend_direction})"
        )


class GasTrendForecaster:
    """
    Gas Trend Forecaster

    Forecasts gas concentration trends using:
    - Linear extrapolation for short-term
    - Exponential smoothing for medium-term
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        min_history_points: int = 3,
    ):
        """
        Initialize Gas Trend Forecaster

        Args:
            thresholds: Custom thresholds (defaults to IEEE C57.104)
            min_history_points: Minimum points required for forecasting
        """
        self.thresholds = thresholds or IEEE_THRESHOLDS
        self.min_history_points = min_history_points

    def forecast(
        self,
        gas_name: str,
        gas_history: List[float],
        dates: Optional[List[float]] = None,
    ) -> ForecastResult:
        """
        Forecast gas concentration trends

        Args:
            gas_name: Name of the gas
            gas_history: Historical gas concentrations (ppm)
            dates: Optional dates (as days from start) for time series

        Returns:
            ForecastResult with forecasts and trend analysis
        """
        if len(gas_history) < self.min_history_points:
            return self._create_insufficient_data_result(gas_name, gas_history)

        current_value = gas_history[-1]

        # Determine forecast method based on data length
        if len(gas_history) >= 10:
            # Use exponential smoothing for medium-term
            forecast_30, forecast_60, forecast_90, trend, confidence = (
                self._exponential_smoothing_forecast(gas_history, dates)
            )
        else:
            # Use linear extrapolation for short-term
            forecast_30, forecast_60, forecast_90, trend, confidence = (
                self._linear_forecast(gas_history, dates)
            )

        # Get threshold for this gas
        threshold = self.thresholds.get(gas_name, float("inf"))

        # Check if threshold will be exceeded
        will_exceed = (
            forecast_30 > threshold
            or forecast_60 > threshold
            or forecast_90 > threshold
        )

        # Calculate days to threshold
        days_to_threshold = self._calculate_days_to_threshold(
            gas_history, threshold, dates
        )

        return ForecastResult(
            gas_name=gas_name,
            current_value=current_value,
            forecast_30_days=forecast_30,
            forecast_60_days=forecast_60,
            forecast_90_days=forecast_90,
            will_exceed_threshold=will_exceed,
            days_to_threshold=days_to_threshold,
            trend_direction=trend,
            confidence=confidence,
        )

    def forecast_all(
        self,
        gas_histories: Dict[str, List[float]],
        dates: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, ForecastResult]:
        """
        Forecast all gases

        Args:
            gas_histories: Dictionary of gas name to history
            dates: Optional dictionary of gas name to dates

        Returns:
            Dictionary of gas name to ForecastResult
        """
        results = {}

        for gas_name, history in gas_histories.items():
            gas_dates = dates.get(gas_name) if dates else None
            results[gas_name] = self.forecast(gas_name, history, gas_dates)

        return results

    def _linear_forecast(
        self,
        gas_history: List[float],
        dates: Optional[List[float]] = None,
    ) -> Tuple[float, float, float, str, float]:
        """
        Forecast using linear regression

        Args:
            gas_history: Historical values
            dates: Optional dates

        Returns:
            Tuple of (forecast_30, forecast_60, forecast_90, trend, confidence)
        """
        y = np.array(gas_history)
        n = len(y)

        # Create x values (time points)
        if dates:
            x = np.array(dates)
            # Normalize to days from start
            x = x - x[0]
        else:
            x = np.arange(n)

        # Fit linear regression
        try:
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]
            intercept = coeffs[1]

            # Calculate R-squared for confidence
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Determine trend direction
            if slope > 0.01 * np.mean(y):
                trend = "INCREASING"
            elif slope < -0.01 * np.mean(y):
                trend = "DECREASING"
            else:
                trend = "STABLE"

            # Forecast
            # Assume monthly data points, convert to days
            if dates and len(dates) >= 2:
                avg_interval = (dates[-1] - dates[0]) / (len(dates) - 1)
            else:
                avg_interval = 30  # Assume monthly

            last_x = x[-1]
            x_30 = last_x + 30 / avg_interval
            x_60 = last_x + 60 / avg_interval
            x_90 = last_x + 90 / avg_interval

            forecast_30 = max(0, np.polyval(coeffs, x_30))
            forecast_60 = max(0, np.polyval(coeffs, x_60))
            forecast_90 = max(0, np.polyval(coeffs, x_90))

            # Confidence based on R-squared and data length
            confidence = min(0.9, r_squared * 0.8 + 0.2 * (n / 10))

            return forecast_30, forecast_60, forecast_90, trend, confidence

        except Exception:
            # Fallback to simple average
            avg = np.mean(gas_history)
            trend = "STABLE"
            return avg, avg, avg, trend, 0.3

    def _exponential_smoothing_forecast(
        self,
        gas_history: List[float],
        dates: Optional[List[float]] = None,
    ) -> Tuple[float, float, float, str, float]:
        """
        Forecast using exponential smoothing

        Args:
            gas_history: Historical values
            dates: Optional dates

        Returns:
            Tuple of (forecast_30, forecast_60, forecast_90, trend, confidence)
        """
        y = np.array(gas_history)
        n = len(y)

        # Simple exponential smoothing
        alpha = 0.3  # Smoothing factor

        # Calculate smoothed values
        smoothed = [y[0]]
        for i in range(1, len(y)):
            smoothed.append(alpha * y[i] + (1 - alpha) * smoothed[-1])

        # Calculate trend from smoothed series
        if len(smoothed) >= 3:
            recent_trend = (smoothed[-1] - smoothed[-3]) / 2
            avg_value = np.mean(smoothed)

            if recent_trend > 0.01 * avg_value:
                trend = "INCREASING"
            elif recent_trend < -0.01 * avg_value:
                trend = "DECREASING"
            else:
                trend = "STABLE"
        else:
            trend = "STABLE"

        # Calculate forecast based on trend
        last_value = smoothed[-1]
        last_slope = smoothed[-1] - smoothed[-2] if len(smoothed) > 1 else 0

        # For exponential, use damped trend
        damping = 0.9

        forecast_30 = last_value + last_slope * 30 * damping
        forecast_60 = last_value + last_slope * 60 * damping
        forecast_60 = forecast_30 + (forecast_60 - forecast_30) * damping
        forecast_90 = last_value + last_slope * 90 * damping
        forecast_90 = forecast_60 + (forecast_90 - forecast_60) * damping

        # Ensure non-negative
        forecast_30 = max(0, forecast_30)
        forecast_60 = max(0, forecast_60)
        forecast_90 = max(0, forecast_90)

        # Confidence based on data length
        confidence = min(0.85, 0.4 + 0.05 * n)

        return forecast_30, forecast_60, forecast_90, trend, confidence

    def _calculate_days_to_threshold(
        self,
        gas_history: List[float],
        threshold: float,
        dates: Optional[List[float]] = None,
    ) -> Optional[int]:
        """
        Calculate days until threshold is exceeded

        Args:
            gas_history: Historical values
            threshold: Threshold value
            dates: Optional dates

        Returns:
            Days to threshold or None
        """
        if len(gas_history) < 3:
            return None

        current = gas_history[-1]

        if current >= threshold:
            return 0  # Already exceeded

        # Use linear trend to predict when threshold will be exceeded
        y = np.array(gas_history)

        if dates:
            x = np.array(dates)
            x = x - x[0]
        else:
            x = np.arange(len(y))

        try:
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]

            if slope <= 0:
                return None  # Not increasing

            # Solve for when y = threshold
            intercept = coeffs[1]
            days = (threshold - intercept) / slope

            # Convert from data interval to days
            if dates and len(dates) >= 2:
                avg_interval = (dates[-1] - dates[0]) / (len(dates) - 1)
                days = days * avg_interval
            else:
                days = days * 30  # Assume monthly

            if days > 365:
                return None  # More than a year

            return int(days)

        except:
            return None

    def _create_insufficient_data_result(
        self, gas_name: str, gas_history: List[float]
    ) -> ForecastResult:
        """
        Create result for insufficient data

        Args:
            gas_name: Gas name
            gas_history: Available history

        Returns:
            ForecastResult with default values
        """
        current = gas_history[-1] if gas_history else 0

        return ForecastResult(
            gas_name=gas_name,
            current_value=current,
            forecast_30_days=current,
            forecast_60_days=current,
            forecast_90_days=current,
            will_exceed_threshold=False,
            days_to_threshold=None,
            trend_direction="STABLE",
            confidence=0.3,
        )

    def get_critical_gases(
        self, results: Dict[str, ForecastResult]
    ) -> List[ForecastResult]:
        """
        Get gases that will exceed thresholds

        Args:
            results: Dictionary of forecast results

        Returns:
            List of critical forecast results
        """
        critical = []

        for result in results.values():
            if result.will_exceed_threshold:
                critical.append(result)

        # Sort by days to threshold
        critical.sort(key=lambda x: x.days_to_threshold or float("inf"))

        return critical

    def get_increasing_gases(
        self, results: Dict[str, ForecastResult]
    ) -> List[ForecastResult]:
        """
        Get gases with increasing trends

        Args:
            results: Dictionary of forecast results

        Returns:
            List of increasing forecast results
        """
        return [
            result
            for result in results.values()
            if result.trend_direction == "INCREASING"
        ]

    def calculate_tdcg(self, gas_values: Dict[str, float]) -> Tuple[float, bool]:
        """
        Calculate Total Dissolved Combustible Gas (TDCG)

        Args:
            gas_values: Current gas values

        Returns:
            Tuple of (tdcg_value, exceeds_threshold)
        """
        tdcg = sum(gas_values.get(gas, 0) for gas in KEY_GASES)
        exceeds = tdcg > TDCG_THRESHOLD

        return tdcg, exceeds
