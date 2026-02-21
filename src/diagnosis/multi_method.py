"""
Multi-Method Ensemble for DGA Fault Diagnosis

This module combines all DGA diagnosis methods into an ensemble that produces
a consensus diagnosis with confidence scores based on IEEE C57.104 and IEC 60599.

The ensemble uses weighted voting or confidence-based consensus to determine
the final fault type, taking into account the results from all individual methods.

Author: TransformerGuard Team
"""

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .doernenburg import Doernenburg, DoernenburgResult
from .duval_triangle import DuvalPentagon, DuvalResult, DuvalTriangle1, FaultType
from .iec_ratios import IecRatios, IecResult
from .key_gas import KeyGasMethod, KeyGasResult
from .rogers_ratios import RogersRatios, RogersResult


@dataclass
class DiagnosisResult:
    """
    Final diagnosis result from the multi-method ensemble.

    Attributes:
        fault_type: Primary detected fault type
        confidence: Overall confidence score (0.0 to 1.0)
        explanation: Comprehensive explanation combining all methods
        method_results: Dictionary of results from each individual method
        supporting_methods: List of methods that support the primary diagnosis
        consensus_score: How many methods agree on the diagnosis
        severity: Severity assessment (NORMAL, LOW, MEDIUM, HIGH, CRITICAL)
    """

    fault_type: FaultType
    confidence: float
    explanation: str
    method_results: Dict[str, Any]
    supporting_methods: List[str] = field(default_factory=list)
    consensus_score: int = 0
    severity: str = "NORMAL"


class MultiMethodDiagnosis:
    """
    Multi-Method Ensemble for DGA Fault Diagnosis.

    This class combines all DGA diagnosis methods:
    - Duval Triangle 1 (for mineral oil transformers)
    - Duval Pentagon (extended version)
    - Rogers Ratio Method
    - IEC Ratio Method
    - Doernenburg Method
    - Key Gas Method

    The ensemble uses confidence-weighted voting to determine the final diagnosis.

    Example:
        >>> ensemble = MultiMethodDiagnosis()
        >>> result = ensemble.diagnose(
        ...     h2=150, ch4=200, c2h2=10, c2h4=50, c2h6=30, co=250, co2=1500
        ... )
        >>> print(result.fault_type)
        FaultType.D2
        >>> print(result.severity)
        CRITICAL
    """

    # Method weights for voting (can be adjusted based on reliability)
    METHOD_WEIGHTS = {
        "DuvalTriangle1": 0.90,
        "DuvalPentagon": 0.85,
        "RogersRatios": 0.80,
        "IecRatios": 0.80,
        "Doernenburg": 0.75,
        "KeyGas": 0.70,
    }

    # Fault type to severity mapping
    SEVERITY_MAP = {
        FaultType.NORMAL: "NORMAL",
        FaultType.PD: "LOW",
        FaultType.T1: "LOW",
        FaultType.T2: "MEDIUM",
        FaultType.D1: "HIGH",
        FaultType.DT: "HIGH",
        FaultType.T3: "CRITICAL",
        FaultType.D2: "CRITICAL",
        FaultType.UNDETERMINED: "UNKNOWN",
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the multi-method ensemble.

        Args:
            config_path: Optional path to configuration file.
        """
        self.config_path = config_path

        # Initialize all methods
        self.duval_triangle = DuvalTriangle1(config_path)
        self.duval_pentagon = DuvalPentagon(config_path)
        self.rogers_ratios = RogersRatios(config_path)
        self.iec_ratios = IecRatios(config_path)
        self.doernenburg = Doernenburg(config_path)
        self.key_gas = KeyGasMethod(config_path)

    def _calculate_weighted_vote(
        self, method_results: Dict[str, Any]
    ) -> tuple[FaultType, int, List[str]]:
        """
        Calculate weighted vote for fault type.

        Args:
            method_results: Dictionary of results from each method

        Returns:
            Tuple of (winning_fault_type, votes, supporting_methods)
        """
        weighted_votes: Dict[FaultType, float] = {}
        method_faults: Dict[str, FaultType] = {}

        for method_name, result in method_results.items():
            if hasattr(result, "fault_type"):
                fault_type = result.fault_type
                confidence = getattr(result, "confidence", 0.5)
                weight = self.METHOD_WEIGHTS.get(method_name, 0.5)

                # Calculate weighted vote
                weighted_score = confidence * weight

                if fault_type not in weighted_votes:
                    weighted_votes[fault_type] = 0
                weighted_votes[fault_type] += weighted_score

                method_faults[method_name] = fault_type

        # Find the winner
        if not weighted_votes:
            return FaultType.UNDETERMINED, 0, []

        # Sort by weighted vote
        sorted_votes = sorted(weighted_votes.items(), key=lambda x: x[1], reverse=True)

        winning_fault = sorted_votes[0][0]

        # Count how many methods agree
        consensus_count = sum(1 for ft in method_faults.values() if ft == winning_fault)

        # Get supporting methods
        supporting = [
            method for method, ft in method_faults.items() if ft == winning_fault
        ]

        return winning_fault, consensus_count, supporting

    def _calculate_overall_confidence(
        self,
        method_results: Dict[str, Any],
        winning_fault: FaultType,
        consensus_count: int,
    ) -> float:
        """
        Calculate overall confidence based on consensus.

        Args:
            method_results: Dictionary of results from each method
            winning_fault: The winning fault type
            consensus_count: Number of methods supporting the winner

        Returns:
            Overall confidence score
        """
        # Average confidence of methods supporting the winner
        supporting_confidences = []
        for method_name, result in method_results.items():
            if hasattr(result, "fault_type") and result.fault_type == winning_fault:
                supporting_confidences.append(result.confidence)

        if not supporting_confidences:
            return 0.3

        avg_confidence = sum(supporting_confidences) / len(supporting_confidences)

        # Adjust based on consensus
        num_methods = len(
            [r for r in method_results.values() if hasattr(r, "fault_type")]
        )
        consensus_ratio = consensus_count / num_methods if num_methods > 0 else 0

        # Final confidence
        overall = avg_confidence * 0.7 + consensus_ratio * 0.3

        return min(overall, 1.0)

    def _generate_explanation(
        self,
        method_results: Dict[str, Any],
        winning_fault: FaultType,
        supporting_methods: List[str],
        consensus_score: int,
    ) -> str:
        """
        Generate comprehensive explanation.

        Args:
            method_results: Results from each method
            winning_fault: The determined fault type
            supporting_methods: Methods supporting the diagnosis
            consensus_score: Number of methods agreeing

        Returns:
            Comprehensive explanation string
        """
        # Get severity
        severity = self.SEVERITY_MAP.get(winning_fault, "UNKNOWN")

        # Build explanation
        explanation = f"=== TransformerGuard Multi-Method DGA Diagnosis ===\n\n"
        explanation += f"Primary Diagnosis: {winning_fault.value}\n"
        explanation += (
            f"Consensus: {consensus_score}/{len(method_results)} methods agree\n"
        )
        explanation += f"Severity: {severity}\n\n"

        explanation += "--- Individual Method Results ---\n"

        for method_name, result in method_results.items():
            if hasattr(result, "fault_type"):
                explanation += f"\n{method_name}:\n"
                explanation += f"  Fault: {result.fault_type.value}\n"
                explanation += f"  Confidence: {result.confidence:.1%}\n"
                explanation += f"  Supporting: {'Yes' if method_name in supporting_methods else 'No'}\n"

        explanation += f"\n--- Supporting Evidence ---\n"

        # Collect supporting evidence
        if winning_fault in [FaultType.PD, FaultType.D1, FaultType.D2]:
            explanation += "- Electrical discharge indicated by:\n"
            for method_name in supporting_methods:
                if method_name in ["DuvalTriangle1", "DuvalPentagon"]:
                    result = method_results.get(method_name)
                    if result and hasattr(result, "gas_percentages"):
                        pct = result.gas_percentages
                        explanation += (
                            f"  {method_name}: C2H2={pct.get('C2H2', 0):.1f}%\n"
                        )

        elif winning_fault in [FaultType.T1, FaultType.T2, FaultType.T3]:
            explanation += "- Thermal fault indicated by:\n"
            for method_name in supporting_methods:
                result = method_results.get(method_name)
                if method_name == "KeyGas" and result:
                    explanation += (
                        f"  {method_name}: Dominant gas indicates overheating\n"
                    )

        explanation += f"\n--- Recommendation ---\n"

        if severity == "NORMAL":
            explanation += "Continue routine monitoring. No action required.\n"
        elif severity == "LOW":
            explanation += "Monitor condition. Schedule next oil test in 3-6 months.\n"
        elif severity == "MEDIUM":
            explanation += (
                "Investigate cause. Consider detailed inspection within 1-3 months.\n"
            )
        elif severity == "HIGH":
            explanation += "Action required. Plan for inspection within 2-4 weeks.\n"
        elif severity == "CRITICAL":
            explanation += (
                "URGENT ACTION REQUIRED. Inspect immediately to prevent failure.\n"
            )

        return explanation

    def _determine_severity(self, fault_type: FaultType) -> str:
        """Determine severity level from fault type."""
        return self.SEVERITY_MAP.get(fault_type, "UNKNOWN")

    def diagnose(
        self,
        h2: float = 0,
        ch4: float = 0,
        c2h2: float = 0,
        c2h4: float = 0,
        c2h6: float = 0,
        co: float = 0,
        co2: float = 0,
        include_pentagon: bool = True,
        **kwargs,
    ) -> DiagnosisResult:
        """
        Run all diagnosis methods and produce ensemble result.

        Args:
            h2: Hydrogen concentration in ppm
            ch4: Methane concentration in ppm
            c2h2: Acetylene concentration in ppm
            c2h4: Ethylene concentration in ppm
            c2h6: Ethane concentration in ppm
            co: Carbon monoxide concentration in ppm
            co2: Carbon dioxide concentration in ppm
            include_pentagon: Whether to include Duval Pentagon in ensemble
            **kwargs: Additional parameters for individual methods

        Returns:
            DiagnosisResult with aggregated diagnosis
        """
        # Run all methods
        method_results = {}

        # Duval Triangle 1
        method_results["DuvalTriangle1"] = self.duval_triangle.diagnose(
            h2=h2, ch4=ch4, c2h2=c2h2, c2h4=c2h4, c2h6=c2h6, co=co, co2=co2
        )

        # Duval Pentagon
        if include_pentagon:
            method_results["DuvalPentagon"] = self.duval_pentagon.diagnose(
                h2=h2, ch4=ch4, c2h2=c2h2, c2h4=c2h4, c2h6=c2h6, co=co, co2=co2
            )

        # Rogers Ratios
        method_results["RogersRatios"] = self.rogers_ratios.diagnose(
            h2=h2, ch4=ch4, c2h2=c2h2, c2h4=c2h4, c2h6=c2h6, co=co, co2=co2
        )

        # IEC Ratios
        method_results["IecRatios"] = self.iec_ratios.diagnose(
            h2=h2, ch4=ch4, c2h2=c2h2, c2h4=c2h4, c2h6=c2h6, co=co, co2=co2
        )

        # Doernenburg
        method_results["Doernenburg"] = self.doernenburg.diagnose(
            h2=h2, ch4=ch4, c2h2=c2h2, c2h4=c2h4, c2h6=c2h6, co=co, co2=co2
        )

        # Key Gas
        method_results["KeyGas"] = self.key_gas.diagnose(
            h2=h2, ch4=ch4, c2h2=c2h2, c2h4=c2h4, c2h6=c2h6, co=co, co2=co2
        )

        # Calculate weighted vote
        winning_fault, consensus_count, supporting_methods = (
            self._calculate_weighted_vote(method_results)
        )

        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            method_results, winning_fault, consensus_count
        )

        # Generate explanation
        explanation = self._generate_explanation(
            method_results, winning_fault, supporting_methods, consensus_count
        )

        # Determine severity
        severity = self._determine_severity(winning_fault)

        return DiagnosisResult(
            fault_type=winning_fault,
            confidence=overall_confidence,
            explanation=explanation,
            method_results={
                name: {
                    "fault_type": result.fault_type.value,
                    "confidence": result.confidence,
                    "explanation": result.explanation,
                }
                for name, result in method_results.items()
            },
            supporting_methods=supporting_methods,
            consensus_score=consensus_count,
            severity=severity,
        )

    def diagnose_partial(
        self,
        h2: float = 0,
        ch4: float = 0,
        c2h2: float = 0,
        c2h4: float = 0,
        c2h6: float = 0,
        co: float = 0,
        co2: float = 0,
        methods: Optional[List[str]] = None,
    ) -> DiagnosisResult:
        """
        Run a subset of diagnosis methods.

        Args:
            h2: Hydrogen concentration in ppm
            ch4: Methane concentration in ppm
            c2h2: Acetylene concentration in ppm
            c2h4: Ethylene concentration in ppm
            c2h6: Ethane concentration in ppm
            co: Carbon monoxide concentration in ppm
            co2: Carbon dioxide concentration in ppm
            methods: List of method names to run. If None, runs all.

        Returns:
            DiagnosisResult with aggregated diagnosis
        """
        if methods is None:
            return self.diagnose(
                h2=h2, ch4=ch4, c2h2=c2h2, c2h4=c2h4, c2h6=c2h6, co=co, co2=co2
            )

        method_results = {}

        available_methods = {
            "DuvalTriangle1": self.duval_triangle,
            "DuvalPentagon": self.duval_pentagon,
            "RogersRatios": self.rogers_ratios,
            "IecRatios": self.iec_ratios,
            "Doernenburg": self.doernenburg,
            "KeyGas": self.key_gas,
        }

        for method_name in methods:
            if method_name in available_methods:
                method_results[method_name] = available_methods[method_name].diagnose(
                    h2=h2, ch4=ch4, c2h2=c2h2, c2h4=c2h4, c2h6=c2h6, co=co, co2=co2
                )

        if not method_results:
            return DiagnosisResult(
                fault_type=FaultType.UNDETERMINED,
                confidence=0.0,
                explanation="No valid methods specified.",
                method_results={},
                supporting_methods=[],
                consensus_score=0,
                severity="UNKNOWN",
            )

        # Calculate weighted vote
        winning_fault, consensus_count, supporting_methods = (
            self._calculate_weighted_vote(method_results)
        )

        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            method_results, winning_fault, consensus_count
        )

        # Generate explanation
        explanation = self._generate_explanation(
            method_results, winning_fault, supporting_methods, consensus_count
        )

        # Determine severity
        severity = self._determine_severity(winning_fault)

        return DiagnosisResult(
            fault_type=winning_fault,
            confidence=overall_confidence,
            explanation=explanation,
            method_results={
                name: {
                    "fault_type": result.fault_type.value,
                    "confidence": result.confidence,
                    "explanation": result.explanation,
                }
                for name, result in method_results.items()
            },
            supporting_methods=supporting_methods,
            consensus_score=consensus_count,
            severity=severity,
        )
