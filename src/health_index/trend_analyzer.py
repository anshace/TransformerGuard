"""
Trend Analyzer
Analyzes health index trends over time and predicts future conditions
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import yaml


@dataclass
class TrendResult:
    """Result of trend analysis"""

    trend_direction: str  # IMPROVING, STABLE, DEGRADING, SIGNIFICANT_CHANGE
    monthly_rate: float  # Points change per month
    yearly_rate: float  # Points change per year
    confidence: float  # Confidence in trend prediction
    prediction_months: Optional[int] = None  # Months until next category
    alerts: List[str] = field(default_factory=list)
    historical_points: List[Dict] = field(default_factory=list)


class TrendAnalyzer:
    """Analyzer for health index trends"""

    # Default thresholds
    DEFAULT_THRESHOLDS = {
        "improvement_threshold": 5,  # Points increase to be IMPROVING
        "degradation_threshold": -5,  # Points decrease to be DEGRADING
        "significant_change": 15,  # Major concern trigger
    }

    # Category thresholds for prediction
    CATEGORY_THRESHOLDS = {
        "EXCELLENT": 85,
        "GOOD": 70,
        "FAIR": 50,
        "POOR": 25,
        "CRITICAL": 0,
    }

    def __init__(self, config_path: Optional[str] = None):
        """Initialize trend analyzer"""
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()

        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path: str):
        """Load configuration from YAML file"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if "trends" in config:
                    self.thresholds = config["trends"]
        except FileNotFoundError:
            pass  # Use defaults

    def _linear_regression(self, x: List[float], y: List[float]) -> tuple:
        """
        Perform linear regression

        Returns:
            (slope, intercept, r_squared)
        """
        if len(x) < 2:
            return 0.0, y[0] if y else 0.0, 0.0

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)

        # Calculate slope and intercept
        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0, sum_y / n, 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n

        # Calculate R-squared
        y_mean = sum_y / n
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y))

        if ss_tot < 1e-10:
            r_squared = 1.0
        else:
            r_squared = 1 - (ss_res / ss_tot)

        return slope, intercept, r_squared

    def _moving_average(self, values: List[float], window: int = 3) -> List[float]:
        """Calculate moving average"""
        if len(values) < window:
            return values

        result = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            result.append(sum(values[start : i + 1]) / (i - start + 1))

        return result

    def analyze(
        self, health_index_history: List[Dict], method: str = "linear"
    ) -> TrendResult:
        """
        Analyze health index trends

        Args:
            health_index_history: List of dicts with 'date' and 'health_index' keys
                                 Date can be datetime or string (ISO format)
            method: 'linear' or 'moving_average'

        Returns:
            TrendResult with analysis
        """
        alerts = []
        historical_points = []

        # Validate input
        if len(health_index_history) < 2:
            return TrendResult(
                trend_direction="STABLE",
                monthly_rate=0.0,
                yearly_rate=0.0,
                confidence=0.0,
                alerts=["Insufficient data for trend analysis"],
            )

        # Parse dates and values
        parsed_data = []
        for entry in health_index_history:
            date = entry.get("date")
            hi = entry.get("health_index")

            if date is None or hi is None:
                continue

            # Parse date
            if isinstance(date, str):
                try:
                    date = datetime.fromisoformat(date.replace("Z", "+00:00"))
                except ValueError:
                    try:
                        date = datetime.strptime(date, "%Y-%m-%d")
                    except ValueError:
                        continue
            elif not isinstance(date, datetime):
                continue

            parsed_data.append((date, hi))

        # Sort by date
        parsed_data.sort(key=lambda x: x[0])

        if len(parsed_data) < 2:
            return TrendResult(
                trend_direction="STABLE",
                monthly_rate=0.0,
                yearly_rate=0.0,
                confidence=0.0,
                alerts=["Insufficient valid data for trend analysis"],
            )

        # Convert dates to months for regression
        first_date = parsed_data[0][0]
        x_months = [(d[0] - first_date).days / 30.44 for d in parsed_data]
        y_values = [d[1] for d in parsed_data]

        # Store historical points for output
        for date, hi in parsed_data:
            historical_points.append(
                {
                    "date": date.isoformat()
                    if isinstance(date, datetime)
                    else str(date),
                    "health_index": hi,
                }
            )

        # Calculate trend using selected method
        if method == "moving_average":
            smoothed = self._moving_average(y_values)
            x_smoothed = list(range(len(smoothed)))
            slope, intercept, r_squared = self._linear_regression(x_smoothed, smoothed)
            # Convert to monthly rate
            monthly_rate = slope
        else:
            # Linear regression
            slope, intercept, r_squared = self._linear_regression(x_months, y_values)
            monthly_rate = slope

        # Yearly rate
        yearly_rate = monthly_rate * 12

        # Determine trend direction
        if monthly_rate >= self.thresholds["improvement_threshold"]:
            trend_direction = "IMPROVING"
        elif monthly_rate <= self.thresholds["degradation_threshold"]:
            trend_direction = "DEGRADING"
        else:
            trend_direction = "STABLE"

        # Check for significant change
        if len(parsed_data) >= 2:
            first_hi = parsed_data[0][1]
            last_hi = parsed_data[-1][1]
            total_change = last_hi - first_hi

            if abs(total_change) >= self.thresholds["significant_change"]:
                trend_direction = "SIGNIFICANT_CHANGE"
                if total_change > 0:
                    alerts.append(
                        f"Significant improvement detected: {total_change:.1f} points"
                    )
                else:
                    alerts.append(
                        f"Significant degradation detected: {total_change:.1f} points"
                    )

        # Calculate confidence based on R-squared and data points
        confidence = min(1.0, r_squared * 0.8 + 0.2 * min(1.0, len(parsed_data) / 10))

        # Predict time to next category
        prediction_months = self._predict_category_change(
            parsed_data, monthly_rate, r_squared
        )

        # Generate alerts based on trend
        if trend_direction == "DEGRADING":
            if monthly_rate <= -2:
                alerts.append(
                    "Rapid degradation detected - immediate attention required"
                )
            else:
                alerts.append("Gradual degradation trend observed - monitor closely")
        elif trend_direction == "IMPROVING":
            if monthly_rate >= 2:
                alerts.append("Significant improvement in condition")
            else:
                alerts.append("Slight improvement trend observed")

        # Check for concerning patterns
        if len(parsed_data) >= 3:
            # Check for accelerating degradation
            recent = y_values[-3:]
            if recent[2] - recent[1] > recent[1] - recent[0]:
                if recent[1] - recent[0] > 0:
                    alerts.append("Degradation rate is accelerating")

        return TrendResult(
            trend_direction=trend_direction,
            monthly_rate=monthly_rate,
            yearly_rate=yearly_rate,
            confidence=confidence,
            prediction_months=prediction_months,
            alerts=alerts,
            historical_points=historical_points,
        )

    def _predict_category_change(
        self, parsed_data: List[tuple], monthly_rate: float, r_squared: float
    ) -> Optional[int]:
        """
        Predict months until next category threshold

        Args:
            parsed_data: List of (date, health_index) tuples
            monthly_rate: Rate of change per month
            r_squared: R-squared value for confidence

        Returns:
            Number of months until next category, or None
        """
        if len(parsed_data) < 2 or abs(monthly_rate) < 0.1:
            return None

        current_hi = parsed_data[-1][1]

        # Determine current category
        current_category = None
        next_threshold = None

        categories = [("EXCELLENT", 85), ("GOOD", 70), ("FAIR", 50), ("POOR", 25)]

        for cat_name, threshold in categories:
            if current_hi >= threshold:
                current_category = cat_name
                next_threshold = threshold
                break

        if next_threshold is None or current_category == "CRITICAL":
            return None

        # Calculate months to next threshold
        points_needed = current_hi - next_threshold

        if monthly_rate < 0:
            # Degrading - going down
            months = int(points_needed / abs(monthly_rate))
            return max(0, months)

        return None

    def predict_future(
        self, health_index_history: List[Dict], months_ahead: int = 12
    ) -> List[Dict]:
        """
        Predict future health index values

        Args:
            health_index_history: List of dicts with 'date' and 'health_index'
            months_ahead: Number of months to predict ahead

        Returns:
            List of predicted values with dates
        """
        if len(health_index_history) < 2:
            return []

        # Parse and sort data
        parsed_data = []
        for entry in health_index_history:
            date = entry.get("date")
            hi = entry.get("health_index")

            if date is None or hi is None:
                continue

            if isinstance(date, str):
                try:
                    date = datetime.fromisoformat(date.replace("Z", "+00:00"))
                except ValueError:
                    try:
                        date = datetime.strptime(date, "%Y-%m-%d")
                    except ValueError:
                        continue

            parsed_data.append((date, hi))

        parsed_data.sort(key=lambda x: x[0])

        if len(parsed_data) < 2:
            return []

        # Get trend
        first_date = parsed_data[0][0]
        x_months = [(d[0] - first_date).days / 30.44 for d in parsed_data]
        y_values = [d[1] for d in parsed_data]

        slope, intercept, _ = self._linear_regression(x_months, y_values)

        # Generate predictions
        predictions = []
        last_date = parsed_data[-1][0]
        last_hi = parsed_data[-1][1]

        for i in range(1, months_ahead + 1):
            future_date = last_date + timedelta(days=30.44 * i)
            future_hi = last_hi + slope * i

            # Clamp to valid range
            future_hi = max(0, min(100, future_hi))

            predictions.append(
                {
                    "date": future_date.isoformat(),
                    "health_index": round(future_hi, 2),
                    "predicted": True,
                }
            )

        return predictions

    def compare_components(
        self, component_history: Dict[str, List[Dict]]
    ) -> Dict[str, TrendResult]:
        """
        Analyze trends for individual components

        Args:
            component_history: Dict of component_name -> list of {date, score}

        Returns:
            Dict of component_name -> TrendResult
        """
        results = {}

        for component_name, history in component_history.items():
            results[component_name] = self.analyze(history)

        return results


def analyze_trends(
    health_index_history: List[Dict],
    method: str = "linear",
    config_path: str = "config/health_index_weights.yaml",
) -> TrendResult:
    """
    Convenience function to analyze health index trends

    Args:
        health_index_history: List of dicts with 'date' and 'health_index'
        method: 'linear' or 'moving_average'
        config_path: Path to configuration file

    Returns:
        TrendResult with analysis
    """
    analyzer = TrendAnalyzer(config_path)
    return analyzer.analyze(health_index_history, method)
