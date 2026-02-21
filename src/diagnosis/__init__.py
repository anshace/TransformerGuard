"""
DGA Fault Diagnosis Module

This module implements multiple DGA (Dissolved Gas Analysis) fault diagnosis
methods based on IEEE C57.104 and IEC 60599 standards.

Available Methods:
- Duval Triangle 1: Triangle method for mineral oil transformers
- Duval Pentagon: Extended triangle method with CO/CO2
- Rogers Ratio: Four-ratio method from IEC 60599
- IEC Ratio: IEC standard ratio interpretation
- Doernenburg: Key gas and ratio method
- Key Gas: Individual gas concentration analysis
- Multi-Method: Ensemble of all methods with consensus
- ML Classifier: Machine learning-based fault classification

Author: TransformerGuard Team
"""

from .doernenburg import Doernenburg, DoernenburgResult
from .duval_triangle import DuvalPentagon, DuvalResult, DuvalTriangle1, FaultType
from .iec_ratios import IecRatios, IecResult
from .key_gas import KeyGasMethod, KeyGasResult
from .ml_classifier import DGAFaultClassifier, MLClassifierResult, create_default_classifier
from .multi_method import DiagnosisResult, MultiMethodDiagnosis
from .rogers_ratios import RogersRatios, RogersResult

__all__ = [
    # Duval Triangle
    "DuvalTriangle1",
    "DuvalPentagon",
    "FaultType",
    "DuvalResult",
    # Rogers Ratios
    "RogersRatios",
    "RogersResult",
    # IEC Ratios
    "IecRatios",
    "IecResult",
    # Doernenburg
    "Doernenburg",
    "DoernenburgResult",
    # Key Gas
    "KeyGasMethod",
    "KeyGasResult",
    # ML Classifier
    "DGAFaultClassifier",
    "MLClassifierResult",
    "create_default_classifier",
    # Multi-Method Ensemble
    "MultiMethodDiagnosis",
    "DiagnosisResult",
]
