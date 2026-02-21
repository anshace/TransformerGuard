"""
Tests for Multi-Method Ensemble DGA Diagnosis

This module tests the MultiMethodDiagnosis class that combines
multiple DGA diagnosis methods into an ensemble.

Author: TransformerGuard Team
"""

import pytest

from src.diagnosis.multi_method import DiagnosisResult, FaultType, MultiMethodDiagnosis


class TestMultiMethodDiagnosis:
    """Test suite for MultiMethodDiagnosis class."""

    def test_initialization(self):
        """Test that MultiMethodDiagnosis initializes correctly."""
        ensemble = MultiMethodDiagnosis()
        assert ensemble is not None

        # Check all methods are initialized
        assert ensemble.duval_triangle is not None
        assert ensemble.duval_pentagon is not None
        assert ensemble.rogers_ratios is not None
        assert ensemble.iec_ratios is not None
        assert ensemble.doernenburg is not None
        assert ensemble.key_gas is not None

    def test_diagnose_runs_all_methods(self):
        """Test that diagnose runs all individual methods."""
        ensemble = MultiMethodDiagnosis()
        result = ensemble.diagnose(
            h2=50, ch4=100, c2h2=5, c2h4=50, c2h6=30, co=150, co2=1000
        )

        # Check that method_results contains all methods
        assert "DuvalTriangle1" in result.method_results
        assert "DuvalPentagon" in result.method_results
        assert "RogersRatios" in result.method_results
        assert "IecRatios" in result.method_results
        assert "Doernenburg" in result.method_results
        assert "KeyGas" in result.method_results

    def test_diagnose_returns_valid_result(self):
        """Test that diagnose returns a valid DiagnosisResult."""
        ensemble = MultiMethodDiagnosis()
        result = ensemble.diagnose(
            h2=50, ch4=100, c2h2=5, c2h4=50, c2h6=30, co=150, co2=1000
        )

        assert isinstance(result, DiagnosisResult)
        assert result.fault_type is not None
        assert 0 <= result.confidence <= 1

    def test_consensus_score_calculated(self):
        """Test that consensus score is calculated correctly."""
        ensemble = MultiMethodDiagnosis()
        result = ensemble.diagnose(
            h2=50, ch4=100, c2h2=5, c2h4=50, c2h6=30, co=150, co2=1000
        )

        assert result.consensus_score >= 0
        assert result.consensus_score <= 6  # Max 6 methods

    def test_supporting_methods_listed(self):
        """Test that supporting methods are listed."""
        ensemble = MultiMethodDiagnosis()
        result = ensemble.diagnose(
            h2=50, ch4=100, c2h2=5, c2h4=50, c2h6=30, co=150, co2=1000
        )

        assert isinstance(result.supporting_methods, list)

    def test_severity_assigned(self):
        """Test that severity is assigned."""
        ensemble = MultiMethodDiagnosis()
        result = ensemble.diagnose(
            h2=50, ch4=100, c2h2=5, c2h4=50, c2h6=30, co=150, co2=1000
        )

        assert result.severity is not None
        assert result.severity in [
            "NORMAL",
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL",
            "UNKNOWN",
        ]

    def test_severity_normal_for_healthy(self):
        """Test that normal operation gets NORMAL severity."""
        ensemble = MultiMethodDiagnosis()
        # Very low gases = normal
        result = ensemble.diagnose(
            h2=20, ch4=10, c2h2=0, c2h4=5, c2h6=10, co=50, co2=500
        )

        assert result.severity == "NORMAL"

    def test_severity_critical_for_d2(self):
        """Test that D2 (high energy discharge) gets CRITICAL severity."""
        ensemble = MultiMethodDiagnosis()
        # Clear high energy discharge
        result = ensemble.diagnose(
            h2=500, ch4=100, c2h2=200, c2h4=200, c2h6=50, co=300, co2=2000
        )

        assert result.severity == "CRITICAL"

    def test_severity_high_for_d1(self):
        """Test that D1 (low energy discharge) gets HIGH severity."""
        ensemble = MultiMethodDiagnosis()
        result = ensemble.diagnose(
            h2=300, ch4=80, c2h2=100, c2h4=60, c2h6=30, co=200, co2=1500
        )

        assert result.severity == "HIGH"

    def test_severity_medium_for_t2(self):
        """Test that T2 (medium thermal) gets MEDIUM severity."""
        ensemble = MultiMethodDiagnosis()
        result = ensemble.diagnose(
            h2=100, ch4=200, c2h2=2, c2h4=150, c2h6=50, co=250, co2=1500
        )

        assert result.severity == "MEDIUM"

    def test_severity_low_for_t1(self):
        """Test that T1 (low thermal) gets LOW severity."""
        ensemble = MultiMethodDiagnosis()
        result = ensemble.diagnose(
            h2=50, ch4=150, c2h2=1, c2h4=20, c2h6=40, co=100, co2=800
        )

        assert result.severity == "LOW"

    def test_explanation_generated(self):
        """Test that comprehensive explanation is generated."""
        ensemble = MultiMethodDiagnosis()
        result = ensemble.diagnose(
            h2=50, ch4=100, c2h2=5, c2h4=50, c2h6=30, co=150, co2=1000
        )

        assert result.explanation is not None
        assert len(result.explanation) > 0
        assert "Multi-Method DGA Diagnosis" in result.explanation

    def test_method_results_contain_fault_type(self):
        """Test that each method result contains fault type."""
        ensemble = MultiMethodDiagnosis()
        result = ensemble.diagnose(
            h2=50, ch4=100, c2h2=5, c2h4=50, c2h6=30, co=150, co2=1000
        )

        for method_name, method_result in result.method_results.items():
            assert "fault_type" in method_result
            assert "confidence" in method_result

    def test_include_pentagon_parameter(self):
        """Test that include_pentagon parameter works."""
        ensemble = MultiMethodDiagnosis()

        # With pentagon
        result_with = ensemble.diagnose(
            h2=50,
            ch4=100,
            c2h2=5,
            c2h4=50,
            c2h6=30,
            co=150,
            co2=1000,
            include_pentagon=True,
        )

        # Without pentagon
        result_without = ensemble.diagnose(
            h2=50,
            ch4=100,
            c2h2=5,
            c2h4=50,
            c2h6=30,
            co=150,
            co2=1000,
            include_pentagon=False,
        )

        # Pentagon should be in method_results when included
        assert "DuvalPentagon" in result_with.method_results
        assert "DuvalPentagon" not in result_without.method_results

    def test_partial_diagnosis_with_specific_methods(self):
        """Test running diagnosis with specific methods only."""
        ensemble = MultiMethodDiagnosis()

        result = ensemble.diagnose_partial(
            h2=50,
            ch4=100,
            c2h2=5,
            c2h4=50,
            c2h6=30,
            co=150,
            co2=1000,
            methods=["DuvalTriangle1", "RogersRatios"],
        )

        assert len(result.method_results) == 2
        assert "DuvalTriangle1" in result.method_results
        assert "RogersRatios" in result.method_results

    def test_partial_diagnosis_empty_methods(self):
        """Test partial diagnosis with empty methods list."""
        ensemble = MultiMethodDiagnosis()

        result = ensemble.diagnose_partial(
            h2=50, ch4=100, c2h2=5, c2h4=50, c2h6=30, co=150, co2=1000, methods=[]
        )

        # Should return undetermined when no methods specified
        assert result.fault_type == FaultType.UNDETERMINED

    def test_partial_diagnosis_invalid_methods(self):
        """Test partial diagnosis with invalid method names."""
        ensemble = MultiMethodDiagnosis()

        result = ensemble.diagnose_partial(
            h2=50,
            ch4=100,
            c2h2=5,
            c2h4=50,
            c2h6=30,
            co=150,
            co2=1000,
            methods=["InvalidMethod1", "InvalidMethod2"],
        )

        # Should return undetermined when no valid methods
        assert result.fault_type == FaultType.UNDETERMINED

    def test_partial_diagnosis_default(self):
        """Test partial diagnosis with default (None) methods."""
        ensemble = MultiMethodDiagnosis()

        result = ensemble.diagnose_partial(
            h2=50, ch4=100, c2h2=5, c2h4=50, c2h6=30, co=150, co2=1000, methods=None
        )

        # Should run all methods like normal diagnose
        assert len(result.method_results) == 6

    def test_with_thermal_fault_data(self, sample_dga_thermal_fault):
        """Test with typical thermal fault DGA data."""
        ensemble = MultiMethodDiagnosis()
        result = ensemble.diagnose(**sample_dga_thermal_fault)

        # Should detect thermal fault
        assert result.fault_type in [
            FaultType.T1,
            FaultType.T2,
            FaultType.T3,
            FaultType.DT,
        ]

    def test_with_arcing_data(self, sample_dga_arcing):
        """Test with typical arcing DGA data."""
        ensemble = MultiMethodDiagnosis()
        result = ensemble.diagnose(**sample_dga_arcing)

        # Should detect discharge fault
        assert result.fault_type in [FaultType.D1, FaultType.D2, FaultType.DT]

    def test_with_partial_discharge_data(self, sample_dga_partial_discharge):
        """Test with typical partial discharge DGA data."""
        ensemble = MultiMethodDiagnosis()
        result = ensemble.diagnose(**sample_dga_partial_discharge)

        assert result is not None

    def test_with_normal_data(self, sample_dga_normal):
        """Test with normal DGA data."""
        ensemble = MultiMethodDiagnosis()
        result = ensemble.diagnose(**sample_dga_normal)

        # Normal or low severity expected
        assert result.severity in ["NORMAL", "LOW", "MEDIUM"]

    def test_weighted_vote_calculation(self):
        """Test weighted vote calculation."""
        ensemble = MultiMethodDiagnosis()

        # Run a diagnosis and check weighted voting
        result = ensemble.diagnose(
            h2=100, ch4=200, c2h2=10, c2h4=100, c2h6=50, co=200, co2=1500
        )

        # Consensus should be at least 1 (the winning fault)
        assert result.consensus_score >= 1

    def test_confidence_based_on_consensus(self):
        """Test that confidence considers consensus."""
        ensemble = MultiMethodDiagnosis()

        # Run multiple diagnoses
        result1 = ensemble.diagnose(
            h2=50, ch4=100, c2h2=5, c2h4=50, c2h6=30, co=150, co2=1000
        )

        # Confidence should be based on supporting methods
        assert 0 <= result1.confidence <= 1


class TestDiagnosisResult:
    """Test suite for DiagnosisResult dataclass."""

    def test_diagnosis_result_creation(self):
        """Test creating a DiagnosisResult instance."""
        result = DiagnosisResult(
            fault_type=FaultType.NORMAL,
            confidence=0.90,
            explanation="Test explanation",
            method_results={},
        )

        assert result.fault_type == FaultType.NORMAL
        assert result.confidence == 0.90
        assert result.explanation == "Test explanation"
        assert result.method_results == {}

    def test_diagnosis_result_with_optional_fields(self):
        """Test creating result with all optional fields."""
        result = DiagnosisResult(
            fault_type=FaultType.D2,
            confidence=0.85,
            explanation="Test",
            method_results={"DuvalTriangle1": {"fault_type": "D2", "confidence": 0.9}},
            supporting_methods=["DuvalTriangle1", "RogersRatios"],
            consensus_score=2,
            severity="CRITICAL",
        )

        assert result.supporting_methods == ["DuvalTriangle1", "RogersRatios"]
        assert result.consensus_score == 2
        assert result.severity == "CRITICAL"


class TestMultiMethodWeights:
    """Test suite for method weights."""

    def test_method_weights_defined(self):
        """Test that method weights are defined."""
        ensemble = MultiMethodDiagnosis()

        assert hasattr(ensemble, "METHOD_WEIGHTS")
        assert "DuvalTriangle1" in ensemble.METHOD_WEIGHTS
        assert "DuvalPentagon" in ensemble.METHOD_WEIGHTS
        assert "RogersRatios" in ensemble.METHOD_WEIGHTS
        assert "IecRatios" in ensemble.METHOD_WEIGHTS
        assert "Doernenburg" in ensemble.METHOD_WEIGHTS
        assert "KeyGas" in ensemble.METHOD_WEIGHTS

    def test_weights_are_positive(self):
        """Test that all method weights are positive."""
        ensemble = MultiMethodDiagnosis()

        for method, weight in ensemble.METHOD_WEIGHTS.items():
            assert weight > 0

    def test_duval_has_highest_weight(self):
        """Test that Duval methods have highest weights."""
        ensemble = MultiMethodDiagnosis()

        # DuvalTriangle1 should have the highest weight
        assert (
            ensemble.METHOD_WEIGHTS["DuvalTriangle1"]
            >= ensemble.METHOD_WEIGHTS["DuvalPentagon"]
        )
        assert (
            ensemble.METHOD_WEIGHTS["DuvalTriangle1"]
            >= ensemble.METHOD_WEIGHTS["RogersRatios"]
        )


class TestSeverityMap:
    """Test suite for severity mapping."""

    def test_severity_map_defined(self):
        """Test that severity map is defined."""
        ensemble = MultiMethodDiagnosis()

        assert hasattr(ensemble, "SEVERITY_MAP")

        # Check all fault types are mapped
        for fault_type in FaultType:
            assert fault_type in ensemble.SEVERITY_MAP

    def test_normal_severity_is_normal(self):
        """Test that NORMAL fault type has NORMAL severity."""
        ensemble = MultiMethodDiagnosis()

        assert ensemble.SEVERITY_MAP[FaultType.NORMAL] == "NORMAL"

    def test_d2_severity_is_critical(self):
        """Test that D2 fault type has CRITICAL severity."""
        ensemble = MultiMethodDiagnosis()

        assert ensemble.SEVERITY_MAP[FaultType.D2] == "CRITICAL"

    def test_d1_severity_is_high(self):
        """Test that D1 fault type has HIGH severity."""
        ensemble = MultiMethodDiagnosis()

        assert ensemble.SEVERITY_MAP[FaultType.D1] == "HIGH"
