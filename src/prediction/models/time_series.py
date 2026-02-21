"""
Time Series Model
Placeholder for ARIMA/Prophet forecasting
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TimeSeriesResult:
    """
    Result from Time Series forecasting

    Attributes:
        forecast: Forecasted values
        lower_bound: Lower confidence bound
        upper_bound: Upper confidence bound
        confidence: Forecast confidence (0.0-1.0)
        seasonal_patterns: Detected seasonal patterns
        trend: Trend direction
        model_ready: Whether model is trained and ready
    """

    forecast: List[float]
    lower_bound: List[float] = field(default_factory=list)
    upper_bound: List[float] = field(default_factory=list)
    confidence: float = 0.0
    seasonal_patterns: Dict[str, float] = field(default_factory=dict)
    trend: str = "STABLE"
    model_ready: bool = False


class TimeSeriesModel:
    """
    Time Series Model for Gas Concentration Forecasting

    This is a placeholder implementation. In production, this would use:
    - ARIMA/SARIMA (statsmodels)
    - Prophet (facebook Prophet)
    - LSTM/GRU (tensorflow/keras)

    Features:
    - Gas concentration forecasting
    - Seasonal pattern detection
    - Confidence intervals
    """

    # Default hyperparameters
    DEFAULT_PARAMS = {
        "order": (1, 1, 1),  # ARIMA order (p, d, q)
        "seasonal_order": (1, 1, 1, 12),  # Seasonal order
        "trend": "c",  # Trend type
        "seasonal": True,
    }

    def __init__(
        self,
        model_type: str = "arima",  # "arima", "prophet", "lstm"
        params: Optional[Dict] = None,
    ):
        """
        Initialize Time Series Model

        Args:
            model_type: Type of model to use
            params: Model hyperparameters
        """
        self.model_type = model_type
        self.params = params or self.DEFAULT_PARAMS.copy()
        self._model = None
        self._is_trained = False
        self._history: List[float] = []

    @property
    def is_available(self) -> bool:
        """Check if time series library is available"""
        try:
            if self.model_type == "arima":
                from statsmodels.tsa.arima.model import ARIMA

                return True
            elif self.model_type == "prophet":
                from prophet import Prophet

                return True
            return False
        except ImportError:
            return False

    def _create_model(self):
        """Create the underlying model"""
        if self.model_type == "arima" and self.is_available:
            from statsmodels.tsa.arima.model import ARIMA

            return ARIMA
        return None

    def train(
        self,
        values: List[float],
        dates: Optional[List] = None,
    ) -> "TimeSeriesModel":
        """
        Train the model

        Args:
            values: Time series values
            dates: Optional dates for the values

        Returns:
            Self for chaining
        """
        self._history = list(values)

        if self.model_type == "arima" and self.is_available:
            self._train_arima(values)
        else:
            self._train_fallback(values)

        self._is_trained = True
        return self

    def _train_arima(self, values: List[float]):
        """Train using ARIMA"""
        try:
            from statsmodels.tsa.arima.model import ARIMA

            order = self.params.get("order", (1, 1, 1))
            self._model = ARIMA(values, order=order)
            self._model = self._model.fit()

        except Exception:
            self._train_fallback(values)

    def _train_fallback(self, values: List[float]):
        """Fallback training"""
        self._model = {
            "values": np.array(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "last_value": values[-1] if values else 0,
            "trend": self._calculate_trend(values),
        }

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend from values"""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        y = np.array(values)

        try:
            coeffs = np.polyfit(x, y, 1)
            return coeffs[0]
        except:
            return 0.0

    def forecast(
        self,
        steps: int = 30,
        confidence: float = 0.95,
    ) -> TimeSeriesResult:
        """
        Make forecasts

        Args:
            steps: Number of steps to forecast
            confidence: Confidence level for intervals

        Returns:
            TimeSeriesResult with forecasts
        """
        if not self._is_trained:
            return TimeSeriesResult(
                forecast=[],
                confidence=0.0,
                model_ready=False,
            )

        if self.model_type == "arima" and hasattr(self._model, "forecast"):
            return self._forecast_arima(steps, confidence)
        else:
            return self._forecast_fallback(steps, confidence)

    def _forecast_arima(
        self,
        steps: int,
        confidence: float,
    ) -> TimeSeriesResult:
        """Forecast using ARIMA"""
        try:
            # Get forecast
            forecast_result = self._model.get_forecast(steps)
            forecast = forecast_result.predicted_mean

            # Get confidence intervals
            conf_int = forecast_result.conf_int(alpha=1 - confidence)

            lower = conf_int.iloc[:, 0].tolist()
            upper = conf_int.iloc[:, 1].tolist()

            # Determine trend
            if len(self._history) >= 5:
                trend = self._calculate_trend(self._history[-5:])
                if trend > 0.01 * np.mean(self._history):
                    trend_str = "INCREASING"
                elif trend < -0.01 * np.mean(self._history):
                    trend_str = "DECREASING"
                else:
                    trend_str = "STABLE"
            else:
                trend_str = "STABLE"

            # Detect seasonal patterns
            seasonal = self._detect_seasonality(self._history)

            return TimeSeriesResult(
                forecast=forecast.tolist(),
                lower_bound=lower,
                upper_bound=upper,
                confidence=confidence,
                seasonal_patterns=seasonal,
                trend=trend_str,
                model_ready=True,
            )

        except Exception:
            return self._forecast_fallback(steps, confidence)

    def _forecast_fallback(
        self,
        steps: int,
        confidence: float,
    ) -> TimeSeriesResult:
        """Fallback forecasting"""
        model_data = self._model

        # Simple linear extrapolation
        trend = model_data.get("trend", 0)
        last_value = model_data.get("last_value", 0)

        forecast = []
        for i in range(1, steps + 1):
            # Damped trend
            damped_trend = trend * (0.9**i)
            value = last_value + damped_trend * i
            forecast.append(max(0, value))  # Ensure non-negative

        # Calculate confidence bounds
        std = model_data.get("std", 1)
        z_score = 1.96  # 95% confidence

        lower = [
            max(0, f - z_score * std * (1 + i * 0.1)) for i, f in enumerate(forecast)
        ]
        upper = [f + z_score * std * (1 + i * 0.1) for f in forecast]

        # Determine trend string
        if trend > 0.01 * model_data.get("mean", 1):
            trend_str = "INCREASING"
        elif trend < -0.01 * model_data.get("mean", 1):
            trend_str = "DECREASING"
        else:
            trend_str = "STABLE"

        return TimeSeriesResult(
            forecast=forecast,
            lower_bound=lower,
            upper_bound=upper,
            confidence=0.7,
            seasonal_patterns={},
            trend=trend_str,
            model_ready=True,
        )

    def _detect_seasonality(self, values: List[float]) -> Dict[str, float]:
        """
        Detect seasonal patterns

        Args:
            values: Time series values

        Returns:
            Dictionary of seasonal patterns
        """
        if len(values) < 12:
            return {}

        try:
            # Simple autocorrelation for seasonality
            from scipy import signal

            # Calculate autocorrelation
            autocorr = np.correlate(values, values, mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]
            autocorr = autocorr / autocorr[0]

            # Find peaks in autocorrelation
            peaks, _ = signal.find_peaks(autocorr[1:], height=0.3)

            if len(peaks) > 0:
                # Return detected seasonal periods
                return {
                    f"period_{peaks[i] + 1}": float(autocorr[peaks[i] + 1])
                    for i in range(min(3, len(peaks)))
                }

        except ImportError:
            pass

        return {}

    def detect_anomalies(
        self,
        threshold: float = 3.0,
    ) -> List[Tuple[int, float, str]]:
        """
        Detect anomalies in the time series

        Args:
            threshold: Z-score threshold for anomaly detection

        Returns:
            List of (index, value, anomaly_type) tuples
        """
        if not self._history or len(self._history) < 3:
            return []

        values = np.array(self._history)
        mean = np.mean(values)
        std = np.std(values)

        if std == 0:
            return []

        anomalies = []

        for i, value in enumerate(values):
            zscore = abs(value - mean) / std

            if zscore > threshold:
                if value > mean:
                    anomalies.append((i, value, "SPIKE_HIGH"))
                else:
                    anomalies.append((i, value, "SPIKE_LOW"))

        return anomalies

    def get_trend_analysis(self) -> Dict:
        """
        Get detailed trend analysis

        Returns:
            Dictionary with trend metrics
        """
        if not self._history:
            return {"trend": "UNKNOWN", "slope": 0.0, "acceleration": 0.0}

        values = self._history

        # Calculate slope
        x = np.arange(len(values))
        y = np.array(values)

        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]

        # Calculate acceleration (second derivative)
        if len(values) >= 3:
            coeffs2 = np.polyfit(x, y, 2)
            acceleration = coeffs2[0]
        else:
            acceleration = 0.0

        # Determine trend
        mean_val = np.mean(values)
        if abs(slope) < 0.01 * mean_val:
            trend = "STABLE"
        elif slope > 0:
            trend = "INCREASING"
        else:
            trend = "DECREASING"

        # Determine acceleration
        if abs(acceleration) < 0.001 * mean_val:
            accel = "CONSTANT"
        elif acceleration > 0:
            accel = "ACCELERATING"
        else:
            accel = "DECELERATING"

        return {
            "trend": trend,
            "slope": float(slope),
            "acceleration": float(acceleration),
            "acceleration_type": accel,
            "mean": float(mean_val),
            "std": float(np.std(values)),
        }

    def update(self, new_value: float) -> "TimeSeriesModel":
        """
        Update model with new observation

        Args:
            new_value: New observation

        Returns:
            Self for chaining
        """
        self._history.append(new_value)

        # Retrain if using ARIMA
        if (
            self.model_type == "arima"
            and self.is_available
            and len(self._history) >= 10
        ):
            try:
                from statsmodels.tsa.arima.model import ARIMA

                order = self.params.get("order", (1, 1, 1))
                self._model = ARIMA(self._history, order=order)
                self._model = self._model.fit()
            except:
                pass

        return self

    @staticmethod
    def prepare_time_series_features(
        gas_history: List[float],
        lags: int = 3,
    ) -> np.ndarray:
        """
        Prepare time series features with lags

        Args:
            gas_history: Historical gas values
            lags: Number of lag features

        Returns:
            Feature array with lagged values
        """
        if len(gas_history) <= lags:
            return np.array([]).reshape(0, lags)

        features = []

        for i in range(lags, len(gas_history)):
            features.append(gas_history[i - lags : i])

        return np.array(features)

    @staticmethod
    def calculate_rolling_stats(
        values: List[float],
        window: int = 3,
    ) -> Dict[str, List[float]]:
        """
        Calculate rolling statistics

        Args:
            values: Time series values
            window: Window size

        Returns:
            Dictionary of rolling statistics
        """
        arr = np.array(values)

        # Rolling mean
        rolling_mean = []
        for i in range(len(arr)):
            start = max(0, i - window + 1)
            rolling_mean.append(np.mean(arr[start : i + 1]))

        # Rolling std
        rolling_std = []
        for i in range(len(arr)):
            start = max(0, i - window + 1)
            rolling_std.append(np.std(arr[start : i + 1]))

        return {
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
        }
