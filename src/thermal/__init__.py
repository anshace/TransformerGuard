"""
Thermal Model Module
Implements IEEE C57.91-2011 thermal modeling standards
"""

from .aging_model import AgingModel, AgingResult
from .hotspot_calculator import HotSpotCalculator, HotSpotResult
from .ieee_c57_91 import IEEEC57_91
from .loading_capability import LoadingCapability, LoadingResult
from .loss_of_life import LossOfLifeCalculator, LossOfLifeResult

__all__ = [
    "IEEEC57_91",
    "HotSpotCalculator",
    "HotSpotResult",
    "AgingModel",
    "AgingResult",
    "LossOfLifeCalculator",
    "LossOfLifeResult",
    "LoadingCapability",
    "LoadingResult",
]
