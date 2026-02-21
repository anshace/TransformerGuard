"""
Loading History Score Calculator
Calculates loading history factor based on average and peak loading over lifetime
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml


@dataclass
class LoadingResult:
    """Result of loading score calculation"""

    score: float  # 0-100
    rating: str  # EXCELLENT, GOOD, FAIR, POOR, CRITICAL
    average_load_percent: float  # Average load as % of rated
    peak_load_percent: float  # Peak load as % of rated
    overload_hours: int  # Hours spent above 100% rated
    issues: List[str] = None
    confidence: float = 1.0

    def __post_init__(self):
        if self.issues is None:
            self.issues = []


class LoadingScore:
    """Calculator for loading history factor in health index"""

    # Default loading to score mapping
    DEFAULT_LOAD_SCORE_MAP = {
        0: 100,
        30: 95,
        50: 90,
        70: 80,
        80: 70,
        90: 60,
        100: 50,
        110: 40,
        120: 30,
    }

    # Overload penalty thresholds
    OVERLOAD_PENALTY_THRESHOLDS = {
        "minor": 100,  # hours
        "moderate": 500,
        "severe": 1000,
    }

    def __init__(self, config_path: Optional[str] = None):
        """Initialize loading score calculator with optional config"""
        self.load_score_map = self.DEFAULT_LOAD_SCORE_MAP.copy()
        self.overload_thresholds = self.OVERLOAD_PENALTY_THRESHOLDS.copy()

        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path: str):
        """Load configuration from YAML file"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if "loading_score" in config:
                    if "average_load_to_score" in config["loading_score"]:
                        self.load_score_map = config["loading_score"][
                            "average_load_to_score"
                        ]
        except FileNotFoundError:
            pass  # Use defaults

    def _interpolate_score(self, load_percent: float) -> float:
        """
        Interpolate score based on load percentage

        Args:
            load_percent: Load as percentage of rated

        Returns:
            Score from 0-100
        """
        # Get sorted load points
        loads = sorted(self.load_score_map.keys())

        # If load is below minimum
        if load_percent <= loads[0]:
            return self.load_score_map[loads[0]]

        # If load is above maximum
        if load_percent >= loads[-1]:
            return self.load_score_map[loads[-1]]

        # Find the two points to interpolate between
        for i in range(len(loads) - 1):
            if loads[i] <= load_percent <= loads[i + 1]:
                # Linear interpolation
                load1, load2 = loads[i], loads[i + 1]
                score1, score2 = self.load_score_map[load1], self.load_score_map[load2]

                ratio = (load_percent - load1) / (load2 - load1)
                return score1 + (score2 - score1) * ratio

        return 50  # Default fallback

    def _calculate_overload_penalty(self, overload_hours: int) -> float:
        """
        Calculate penalty for overload hours

        Args:
            overload_hours: Hours spent above rated load

        Returns:
            Penalty to subtract from score (0-50)
        """
        if overload_hours < self.overload_thresholds["minor"]:
            return 0
        elif overload_hours < self.overload_thresholds["moderate"]:
            return (overload_hours / self.overload_thresholds["minor"]) * 10
        elif overload_hours < self.overload_thresholds["severe"]:
            return (
                10
                + (
                    (overload_hours - self.overload_thresholds["moderate"])
                    / (
                        self.overload_thresholds["severe"]
                        - self.overload_thresholds["moderate"]
                    )
                )
                * 15
            )
        else:
            return 25 + min(
                25, (overload_hours - self.overload_thresholds["severe"]) / 200
            )

    def calculate_score(
        self,
        average_load_percent: float,
        peak_load_percent: Optional[float] = None,
        overload_hours: int = 0,
        load_profile: Optional[List[Dict]] = None,
    ) -> LoadingResult:
        """
        Calculate loading history score (0-100)

        Args:
            average_load_percent: Average load as % of rated capacity
            peak_load_percent: Peak load as % of rated capacity (optional)
            overload_hours: Hours spent above 100% rated load
            load_profile: Optional list of {load_percent, hours} dictionaries

        Returns:
            LoadingResult with score and rating
        """
        issues = []

        # Calculate base score from average load
        base_score = self._interpolate_score(average_load_percent)

        # If detailed load profile provided, use it
        if load_profile:
            weighted_score = 0
            total_hours = 0
            for entry in load_profile:
                load = entry.get("load_percent", 0)
                hours = entry.get("hours", 0)
                segment_score = self._interpolate_score(load)
                weighted_score += segment_score * hours
                total_hours += hours

            if total_hours > 0:
                base_score = weighted_score / total_hours

        # Apply overload penalty
        overload_penalty = self._calculate_overload_penalty(overload_hours)

        # Additional penalty for high peak loads
        peak_penalty = 0
        if peak_load_percent:
            if peak_load_percent > 115:
                peak_penalty = 10
            elif peak_load_percent > 110:
                peak_penalty = 5
            elif peak_load_percent > 105:
                peak_penalty = 2

        # Calculate final score
        score = base_score - overload_penalty - peak_penalty

        # Identify issues
        if average_load_percent > 90:
            issues.append(f"High average loading: {average_load_percent}%")

        if peak_load_percent and peak_load_percent > 110:
            issues.append(f"High peak loading: {peak_load_percent}%")

        if overload_hours > self.overload_thresholds["moderate"]:
            issues.append(f"Significant overload hours: {overload_hours} hrs")

        # Ensure score is in valid range
        score = max(0, min(100, score))

        # Determine rating
        rating = self._get_rating(score)

        return LoadingResult(
            score=score,
            rating=rating,
            average_load_percent=average_load_percent,
            peak_load_percent=peak_load_percent
            if peak_load_percent
            else average_load_percent,
            overload_hours=overload_hours,
            issues=issues,
            confidence=0.90,
        )

    def calculate_from_load_factor(
        self,
        load_factor: float,
        peak_factor: Optional[float] = None,
        total_hours: Optional[float] = None,
    ) -> LoadingResult:
        """
        Calculate loading score from load factor

        Args:
            load_factor: Average load / rated load (0.0 to >1.0)
            peak_factor: Peak load / rated load (optional)
            total_hours: Total operating hours (optional)

        Returns:
            LoadingResult with score and rating
        """
        avg_load = load_factor * 100
        peak_load = (peak_factor * 100) if peak_factor else None

        # Estimate overload hours from peak factor
        overload_hours = 0
        if peak_factor and total_hours and peak_factor > 1.0:
            # Estimate proportion of time above 100%
            overload_factor = peak_factor - 1.0
            overload_hours = int(total_hours * min(overload_factor, 0.3))

        return self.calculate_score(avg_load, peak_load, overload_hours)

    def _get_rating(self, score: float) -> str:
        """Get rating string from score"""
        if score >= 85:
            return "EXCELLENT"
        elif score >= 70:
            return "GOOD"
        elif score >= 50:
            return "FAIR"
        elif score >= 25:
            return "POOR"
        else:
            return "CRITICAL"


def calculate_loading_score(
    average_load_percent: float,
    peak_load_percent: Optional[float] = None,
    overload_hours: int = 0,
    load_profile: Optional[List[Dict]] = None,
    config_path: Optional[str] = "config/health_index_weights.yaml",
) -> LoadingResult:
    """
    Convenience function to calculate loading history score

    Args:
        average_load_percent: Average load as % of rated capacity
        peak_load_percent: Peak load as % of rated capacity (optional)
        overload_hours: Hours spent above 100% rated load
        load_profile: Optional list of {load_percent, hours} dictionaries
        config_path: Path to configuration file

    Returns:
        LoadingResult with score and rating
    """
    calculator = LoadingScore(config_path)
    return calculator.calculate_score(
        average_load_percent, peak_load_percent, overload_hours, load_profile
    )
