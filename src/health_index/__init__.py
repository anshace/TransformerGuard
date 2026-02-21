"""
Health Index Module
Combines DGA, oil quality, electrical, age, and loading into 0-100 score
"""

from .age_score import AgeResult, AgeScore
from .composite_hi import CompositeHealthIndex, HealthIndexResult
from .dga_score import DGAScoreCalculator, DGAScoreResult
from .electrical_score import ElectricalResult, ElectricalScore
from .loading_score import LoadingResult, LoadingScore
from .oil_quality_score import OilQualityResult, OilQualityScore
from .trend_analyzer import TrendAnalyzer, TrendResult

__all__ = [
    "DGAScoreCalculator",
    "DGAScoreResult",
    "OilQualityScore",
    "OilQualityResult",
    "ElectricalScore",
    "ElectricalResult",
    "AgeScore",
    "AgeResult",
    "LoadingScore",
    "LoadingResult",
    "CompositeHealthIndex",
    "HealthIndexResult",
    "TrendAnalyzer",
    "TrendResult",
]
