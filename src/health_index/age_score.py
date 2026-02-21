"""
Age-Based Score Calculator
Calculates age degradation factor based on expected transformer life
Uses IEEE C57.12.00 standard: 20.5 years (180,000 hours)
"""

from dataclasses import dataclass
from typing import Dict, Optional

import yaml


@dataclass
class AgeResult:
    """Result of age score calculation"""

    score: float  # 0-100
    rating: str  # EXCELLENT, GOOD, FAIR, POOR, CRITICAL
    age_years: float  # Transformer age in years
    expected_life_years: float  # Expected life in years
    life_consumed_percent: float  # Percentage of life consumed
    confidence: float = 1.0


class AgeScore:
    """Calculator for age factor in health index"""

    # IEEE C57.12.00 standard expected life
    DEFAULT_EXPECTED_LIFE_HOURS = 180000  # hours
    DEFAULT_EXPECTED_LIFE_YEARS = 20.5  # years

    # Default age to score mapping
    DEFAULT_AGE_SCORE_MAP = {
        0: 100,
        5: 95,
        10: 85,
        15: 75,
        20: 60,
        25: 45,
        30: 30,
        35: 15,
        40: 5,
    }

    def __init__(self, config_path: Optional[str] = None):
        """Initialize age score calculator with optional config"""
        self.expected_life_years = self.DEFAULT_EXPECTED_LIFE_YEARS
        self.age_score_map = self.DEFAULT_AGE_SCORE_MAP.copy()

        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path: str):
        """Load configuration from YAML file"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if "age_score" in config:
                    self.age_score_map = config["age_score"]
                if "expected_life_years" in config:
                    self.expected_life_years = config["expected_life_years"]
        except FileNotFoundError:
            pass  # Use defaults

    def _interpolate_score(self, age: float) -> float:
        """
        Interpolate score based on age using the mapping

        Args:
            age: Age in years

        Returns:
            Score from 0-100
        """
        # Get sorted age points
        ages = sorted(self.age_score_map.keys())

        # If age is below minimum
        if age <= ages[0]:
            return self.age_score_map[ages[0]]

        # If age is above maximum
        if age >= ages[-1]:
            return self.age_score_map[ages[-1]]

        # Find the two points to interpolate between
        for i in range(len(ages) - 1):
            if ages[i] <= age <= ages[i + 1]:
                # Linear interpolation
                age1, age2 = ages[i], ages[i + 1]
                score1, score2 = self.age_score_map[age1], self.age_score_map[age2]

                ratio = (age - age1) / (age2 - age1)
                return score1 + (score2 - score1) * ratio

        return 50  # Default fallback

    def calculate_score(
        self,
        age_years: float,
        expected_life_years: Optional[float] = None,
        degradation_model: str = "linear",
    ) -> AgeResult:
        """
        Calculate age score (0-100)

        Args:
            age_years: Current age of transformer in years
            expected_life_years: Expected life in years (optional, uses default)
            degradation_model: 'linear' or 'exponential' degradation

        Returns:
            AgeResult with score and rating
        """
        # Use provided expected life or default
        if expected_life_years is None:
            expected_life_years = self.expected_life_years

        # Calculate life consumed
        life_consumed_percent = (age_years / expected_life_years) * 100

        # Calculate score based on degradation model
        if degradation_model == "exponential":
            # Exponential degradation - faster decline in later years
            # Using formula: score = 100 * e^(-k * age) where k is chosen so that
            # score reaches ~10% at expected life
            import math

            k = -math.log(0.1) / expected_life_years
            score = 100 * math.exp(-k * age_years)
        else:
            # Linear degradation (default)
            score = self._interpolate_score(age_years)

        # Ensure score is in valid range
        score = max(0, min(100, score))

        # Determine rating
        rating = self._get_rating(score)

        return AgeResult(
            score=score,
            rating=rating,
            age_years=age_years,
            expected_life_years=expected_life_years,
            life_consumed_percent=life_consumed_percent,
            confidence=0.95,
        )

    def calculate_from_hours(self, operating_hours: float) -> AgeResult:
        """
        Calculate age score from operating hours

        Args:
            operating_hours: Total operating hours

        Returns:
            AgeResult with score and rating
        """
        age_years = operating_hours / (24 * 365.25)
        return self.calculate_score(age_years)

    def calculate_remaining_life(
        self, current_age_years: float, expected_life_years: Optional[float] = None
    ) -> float:
        """
        Calculate remaining useful life in years

        Args:
            current_age_years: Current age in years
            expected_life_years: Expected life in years

        Returns:
            Remaining life in years
        """
        if expected_life_years is None:
            expected_life_years = self.expected_life_years

        remaining = expected_life_years - current_age_years
        return max(0, remaining)

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


def calculate_age_score(
    age_years: float,
    expected_life_years: Optional[float] = None,
    degradation_model: str = "linear",
    config_path: Optional[str] = "config/health_index_weights.yaml",
) -> AgeResult:
    """
    Convenience function to calculate age score

    Args:
        age_years: Current age of transformer in years
        expected_life_years: Expected life in years (optional)
        degradation_model: 'linear' or 'exponential'
        config_path: Path to configuration file

    Returns:
        AgeResult with score and rating
    """
    calculator = AgeScore(config_path)
    return calculator.calculate_score(age_years, expected_life_years, degradation_model)
