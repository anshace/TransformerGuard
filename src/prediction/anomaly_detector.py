"""
Statistical Anomaly Detection
Detects anomalies in DGA data using statistical methods
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AnomalyResult:
    """
    Result of anomaly detection

    Attributes:
        is_anomaly: Whether an anomaly was detected
        anomaly_score: Anomaly score (0.0-1.0)
        anomaly_type: Type of anomaly (GAS_SPIKE, TREND_CHANGE, CORRELATION_BREAK, NONE)
        affected_gases: List of affected gas names
        severity: Severity level (LOW, MEDIUM, HIGH)
        explanation: Human-readable explanation
    """

    is_anomaly: bool
    anomaly_score: float
    anomaly_type: str
    affected_gases: List[str] = field(default_factory=list)
    severity: str = "LOW"
    explanation: str = ""

    def __str__(self) -> str:
        return (
            f"AnomalyResult(is_anomaly={self.is_anomaly}, "
            f"type={self.anomaly_type}, score={self.anomaly_score:.3f}, "
            f"severity={self.severity})"
        )


class AnomalyDetector:
    """
    Statistical Anomaly Detection for DGA data

    Detects anomalies using:
    - Statistical methods (z-score, IQR)
    - Gas rate-of-change analysis
    - Multi-gas correlation analysis
    """

    # IEEE C57.104 thresholds for anomaly detection
    IEEE_THRESHOLDS = {
        "H2": 100,  # ppm
        "CH4": 120,  # ppm
        "C2H2": 2,  # ppm
        "C2H4": 50,  # ppm
        "C2H6": 65,  # ppm
        "CO": 350,  # ppm
        "CO2": 2500,  # ppm
    }

    # Key gas pairs for correlation analysis
    CORRELATION_PAIRS = [
        ("CH4", "C2H4"),  # Thermal
        ("C2H2", "C2H4"),  # Arcing
        ("H2", "CH4"),  # Thermal
        ("CO", "CO2"),  # General aging
    ]

    def __init__(
        self,
        zscore_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
    ):
        """
        Initialize Anomaly Detector

        Args:
            zscore_threshold: Z-score threshold for anomaly detection (default: 3.0)
            iqr_multiplier: IQR multiplier for outlier detection (default: 1.5)
        """
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier

    def detect_anomaly(
        self,
        current_gas_values: Dict[str, float],
        gas_history: Optional[Dict[str, List[float]]] = None,
        gas_thresholds: Optional[Dict[str, float]] = None,
    ) -> AnomalyResult:
        """
        Detect anomalies in DGA data

        Args:
            current_gas_values: Current gas concentrations (ppm)
            gas_history: Optional historical gas values for trend analysis
            gas_thresholds: Optional custom thresholds (defaults to IEEE)

        Returns:
            AnomalyResult with detection results
        """
        # Use provided thresholds or IEEE defaults
        thresholds = gas_thresholds or self.IEEE_THRESHOLDS

        # Check for threshold violations
        threshold_result = self._check_threshold_violations(
            current_gas_values, thresholds
        )
        if threshold_result.is_anomaly:
            return threshold_result

        # If we have history, check for statistical anomalies
        if gas_history:
            # Check for gas spikes
            spike_result = self._detect_gas_spikes(gas_history, current_gas_values)
            if spike_result.is_anomaly:
                return spike_result

            # Check for trend changes
            trend_result = self._detect_trend_changes(gas_history)
            if trend_result.is_anomaly:
                return trend_result

            # Check for correlation breaks
            corr_result = self._detect_correlation_breaks(gas_history)
            if corr_result.is_anomaly:
                return corr_result

        # No anomaly detected
        return AnomalyResult(
            is_anomaly=False,
            anomaly_score=0.0,
            anomaly_type="NONE",
            affected_gases=[],
            severity="LOW",
            explanation="No anomalies detected in DGA data",
        )

    def _check_threshold_violations(
        self,
        current_gas_values: Dict[str, float],
        thresholds: Dict[str, float],
    ) -> AnomalyResult:
        """
        Check for threshold violations

        Args:
            current_gas_values: Current gas concentrations
            thresholds: Threshold values

        Returns:
            AnomalyResult if threshold violated
        """
        affected_gases = []
        max_exceed_ratio = 0.0

        for gas, value in current_gas_values.items():
            if gas in thresholds:
                threshold = thresholds[gas]
                if value > threshold:
                    affected_gases.append(gas)
                    exceed_ratio = (value - threshold) / threshold
                    max_exceed_ratio = max(max_exceed_ratio, exceed_ratio)

        if affected_gases:
            # Calculate severity based on exceed ratio
            if max_exceed_ratio > 2.0:
                severity = "HIGH"
                score = 0.9
            elif max_exceed_ratio > 1.0:
                severity = "MEDIUM"
                score = 0.7
            else:
                severity = "LOW"
                score = 0.5

            return AnomalyResult(
                is_anomaly=True,
                anomaly_score=score,
                anomaly_type="GAS_SPIKE",
                affected_gases=affected_gases,
                severity=severity,
                explanation=f"Gas concentrations exceed thresholds: {', '.join(affected_gases)}",
            )

        return AnomalyResult(is_anomaly=False, anomaly_score=0.0, anomaly_type="NONE")

    def _detect_gas_spikes(
        self,
        gas_history: Dict[str, List[float]],
        current_gas_values: Dict[str, float],
    ) -> AnomalyResult:
        """
        Detect sudden gas spikes using z-score and IQR methods

        Args:
            gas_history: Historical gas values
            current_gas_values: Current gas concentrations

        Returns:
            AnomalyResult if spike detected
        """
        affected_gases = []
        max_score = 0.0

        for gas, history in gas_history.items():
            if gas not in current_gas_values:
                continue

            if len(history) < 3:
                continue

            current = current_gas_values[gas]

            # Z-score method
            zscore_result = self._zscore_anomaly(history, current)

            # IQR method
            iqr_result = self._iqr_anomaly(history, current)

            # Use the higher score
            if zscore_result > 0 or iqr_result > 0:
                affected_gases.append(gas)
                max_score = max(max_score, zscore_result, iqr_result)

        if affected_gases:
            severity = self._score_to_severity(max_score)
            return AnomalyResult(
                is_anomaly=True,
                anomaly_score=max_score,
                anomaly_type="GAS_SPIKE",
                affected_gases=affected_gases,
                severity=severity,
                explanation=f"Sudden gas spike detected in: {', '.join(affected_gases)}",
            )

        return AnomalyResult(is_anomaly=False, anomaly_score=0.0, anomaly_type="NONE")

    def _zscore_anomaly(self, history: List[float], current: float) -> float:
        """
        Detect anomaly using z-score method

        Args:
            history: Historical values
            current: Current value

        Returns:
            Anomaly score (0.0-1.0)
        """
        if len(history) < 3:
            return 0.0

        arr = np.array(history)
        mean = np.mean(arr)
        std = np.std(arr)

        if std == 0:
            return 0.0

        zscore = abs(current - mean) / std

        # Calculate score based on z-score
        if zscore > self.zscore_threshold:
            # Linear scaling beyond threshold
            score = min(1.0, (zscore - self.zscore_threshold) / 2 + 0.7)
            return score

        return 0.0

    def _iqr_anomaly(self, history: List[float], current: float) -> float:
        """
        Detect anomaly using IQR method

        Args:
            history: Historical values
            current: Current value

        Returns:
            Anomaly score (0.0-1.0)
        """
        if len(history) < 4:
            return 0.0

        arr = np.array(history)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1

        if iqr == 0:
            return 0.0

        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr

        if current < lower_bound or current > upper_bound:
            # Calculate how far outside bounds
            if current < lower_bound:
                distance = lower_bound - current
            else:
                distance = current - upper_bound

            # Scale score based on distance
            score = min(1.0, distance / iqr * 0.5 + 0.5)
            return score

        return 0.0

    def _detect_trend_changes(
        self, gas_history: Dict[str, List[float]]
    ) -> AnomalyResult:
        """
        Detect sudden trend changes

        Args:
            gas_history: Historical gas values

        Returns:
            AnomalyResult if trend change detected
        """
        affected_gases = []
        max_score = 0.0

        for gas, history in gas_history.items():
            if len(history) < 5:
                continue

            # Calculate recent vs historical trend
            recent_trend = self._calculate_trend(history[-3:])
            historical_trend = self._calculate_trend(history[:-3])

            if historical_trend is None or recent_trend is None:
                continue

            # Check for significant trend change
            trend_diff = abs(recent_trend - historical_trend)

            # Significant if recent trend is 3x historical
            if abs(historical_trend) > 0.01 and trend_diff > abs(historical_trend) * 3:
                score = min(1.0, trend_diff / abs(historical_trend) * 0.5)
                affected_gases.append(gas)
                max_score = max(max_score, score)

        if affected_gases:
            severity = self._score_to_severity(max_score)
            return AnomalyResult(
                is_anomaly=True,
                anomaly_score=max_score,
                anomaly_type="TREND_CHANGE",
                affected_gases=affected_gases,
                severity=severity,
                explanation=f"Trend change detected in: {', '.join(affected_gases)}",
            )

        return AnomalyResult(is_anomaly=False, anomaly_score=0.0, anomaly_type="NONE")

    def _calculate_trend(self, values: List[float]) -> Optional[float]:
        """
        Calculate linear trend (slope)

        Args:
            values: List of values

        Returns:
            Slope of trend or None
        """
        if len(values) < 2:
            return None

        try:
            y = np.array(values)
            x = np.arange(len(y))
            coeffs = np.polyfit(x, y, 1)
            return coeffs[0]
        except:
            return None

    def _detect_correlation_breaks(
        self, gas_history: Dict[str, List[float]]
    ) -> AnomalyResult:
        """
        Detect breaks in expected gas correlations

        Args:
            gas_history: Historical gas values

        Returns:
            AnomalyResult if correlation break detected
        """
        affected_gases = []
        max_score = 0.0

        for gas1, gas2 in self.CORRELATION_PAIRS:
            if gas1 not in gas_history or gas2 not in gas_history:
                continue

            history1 = gas_history[gas1]
            history2 = gas_history[gas2]

            if len(history1) < 5 or len(history2) < 5:
                continue

            # Calculate historical correlation
            hist_corr = self._calculate_correlation(history1[:-3], history2[:-3])

            # Calculate recent correlation
            recent_corr = self._calculate_correlation(history1[-3:], history2[-3:])

            # Check for significant correlation break
            if hist_corr is not None and recent_corr is not None:
                corr_diff = abs(recent_corr - hist_corr)

                # If correlation was strong and now weak
                if abs(hist_corr) > 0.7 and corr_diff > 0.5:
                    score = corr_diff
                    affected_gases.extend([gas1, gas2])
                    max_score = max(max_score, score)

        if affected_gases:
            affected_gases = list(set(affected_gases))  # Remove duplicates
            severity = self._score_to_severity(max_score)
            return AnomalyResult(
                is_anomaly=True,
                anomaly_score=max_score,
                anomaly_type="CORRELATION_BREAK",
                affected_gases=affected_gases,
                severity=severity,
                explanation=f"Correlation break detected in gas pairs",
            )

        return AnomalyResult(is_anomaly=False, anomaly_score=0.0, anomaly_type="NONE")

    def _calculate_correlation(
        self, values1: List[float], values2: List[float]
    ) -> Optional[float]:
        """
        Calculate correlation between two gas histories

        Args:
            values1: First gas history
            values2: Second gas history

        Returns:
            Correlation coefficient or None
        """
        if len(values1) < 2 or len(values2) < 2:
            return None

        try:
            corr = np.corrcoef(values1, values2)[0, 1]
            return corr if not np.isnan(corr) else None
        except:
            return None

    def _score_to_severity(self, score: float) -> str:
        """
        Convert anomaly score to severity level

        Args:
            score: Anomaly score (0.0-1.0)

        Returns:
            Severity string
        """
        if score >= 0.8:
            return "HIGH"
        elif score >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"

    def analyze_all_gases(
        self,
        gas_history: Dict[str, List[float]],
    ) -> Dict[str, AnomalyResult]:
        """
        Analyze all gases individually

        Args:
            gas_history: Historical gas values

        Returns:
            Dictionary of gas name to anomaly result
        """
        results = {}

        for gas, history in gas_history.items():
            if len(history) >= 3:
                current = history[-1]
                result = AnomalyResult(
                    is_anomaly=False,
                    anomaly_score=0.0,
                    anomaly_type="NONE",
                )

                # Check z-score
                zscore = self._zscore_anomaly(history[:-1], current)
                if zscore > 0:
                    result.is_anomaly = True
                    result.anomaly_score = max(result.anomaly_score, zscore)
                    result.anomaly_type = "GAS_SPIKE"
                    result.affected_gases = [gas]
                    result.severity = self._score_to_severity(zscore)

                results[gas] = result
            else:
                results[gas] = AnomalyResult(
                    is_anomaly=False,
                    anomaly_score=0.0,
                    anomaly_type="NONE",
                )

        return results
