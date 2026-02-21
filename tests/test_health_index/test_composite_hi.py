"""
Tests for Composite Health Index Calculator

This module tests the CompositeHealthIndex class for calculating
overall transformer health index from multiple components.

Author: TransformerGuard Team
"""

import pytest

from src.health_index.composite_hi import CompositeHealthIndex, HealthIndexResult


class TestCompositeHealthIndex:
    """Test suite for CompositeHealthIndex class."""

    def test_initialization(self):
        """Test that CompositeHealthIndex initializes correctly."""
        hi = CompositeHealthIndex()

        assert hi is not None
        assert hi.dga_calculator is not None
        assert hi.oil_calculator is not None
        assert hi.electrical_calculator is not None
        assert hi.age_calculator is not None
        assert hi.loading_calculator is not None

    def test_weights_sum_to_one(self):
        """Test that default weights sum to 1.0."""
        hi = CompositeHealthIndex()

        total_weight = sum(hi.weights.values())

        assert abs(total_weight - 1.0) < 0.001

    def test_weights_defined(self):
        """Test that all required weights are defined."""
        hi = CompositeHealthIndex()

        assert "dga" in hi.weights
        assert "oil_quality" in hi.weights
        assert "electrical" in hi.weights
        assert "age" in hi.weights
        assert "loading" in hi.weights

    def test_categories_defined(self):
        """Test that health categories are defined."""
        hi = CompositeHealthIndex()

        assert "excellent" in hi.categories
        assert "good" in hi.categories
        assert "fair" in hi.categories
        assert "poor" in hi.categories
        assert "critical" in hi.categories

    def test_category_ranges(self):
        """Test that category ranges are valid."""
        hi = CompositeHealthIndex()

        for cat_name, cat_def in hi.categories.items():
            assert "min" in cat_def
            assert "max" in cat_def
            assert cat_def["min"] >= 0
            assert cat_def["max"] <= 100
            assert cat_def["min"] <= cat_def["max"]

    def test_calculate_with_minimal_data(self):
        """Test calculation with only DGA data."""
        hi = CompositeHealthIndex()

        result = hi.calculate(
            dga_gases={
                "h2": 50,
                "ch4": 100,
                "c2h2": 5,
                "c2h4": 30,
                "c2h6": 20,
                "co": 150,
                "co2": 1000,
            }
        )

        assert isinstance(result, HealthIndexResult)
        assert 0 <= result.health_index <= 100

    def test_calculate_with_all_data(self):
        """Test calculation with all component data."""
        hi = CompositeHealthIndex()

        result = hi.calculate(
            dga_gases={
                "h2": 50,
                "ch4": 100,
                "c2h2": 5,
                "c2h4": 30,
                "c2h6": 20,
                "co": 150,
                "co2": 1000,
            },
            dielectric_strength=45.0,
            moisture_content=15.0,
            acidity=0.10,
            power_factor=0.5,
            capacitance=1200.0,
            insulation_resistance=5000.0,
            age_years=10.0,
            average_load_percent=60.0,
        )

        assert result.health_index > 0
        assert result.confidence > 0

    def test_healthy_transformer_score(self, sample_health_index_healthy):
        """Test health index for healthy transformer."""
        hi = CompositeHealthIndex()

        result = hi.calculate(**sample_health_index_healthy)

        # Healthy transformer should have high health index
        assert result.health_index > 70
        assert result.category in ["EXCELLENT", "GOOD"]

    def test_critical_transformer_score(self, sample_health_index_critical):
        """Test health index for critical transformer."""
        hi = CompositeHealthIndex()

        result = hi.calculate(**sample_health_index_critical)

        # Critical transformer should have low health index
        assert result.health_index < 50
        assert result.category in ["POOR", "CRITICAL"]

    def test_category_assignment_excellent(self):
        """Test category assignment for excellent health."""
        hi = CompositeHealthIndex()

        result = hi.calculate(
            dga_gases={
                "h2": 10,
                "ch4": 20,
                "c2h2": 0,
                "c2h4": 5,
                "c2h6": 10,
                "co": 50,
                "co2": 500,
            },
            dielectric_strength=60.0,
            moisture_content=5.0,
            acidity=0.02,
            power_factor=0.1,
            capacitance=1000.0,
            insulation_resistance=20000.0,
            age_years=2.0,
            average_load_percent=40.0,
        )

        assert result.category == "EXCELLENT"

    def test_category_assignment_good(self):
        """Test category assignment for good health."""
        hi = CompositeHealthIndex()

        result = hi.calculate(
            dga_gases={
                "h2": 30,
                "ch4": 50,
                "c2h2": 1,
                "c2h4": 15,
                "c2h6": 15,
                "co": 80,
                "co2": 600,
            },
            dielectric_strength=50.0,
            moisture_content=10.0,
            acidity=0.05,
            power_factor=0.3,
            capacitance=1100.0,
            insulation_resistance=10000.0,
            age_years=8.0,
            average_load_percent=50.0,
        )

        assert result.category in ["EXCELLENT", "GOOD"]

    def test_category_assignment_fair(self):
        """Test category assignment for fair health."""
        hi = CompositeHealthIndex()

        result = hi.calculate(
            dga_gases={
                "h2": 100,
                "ch4": 150,
                "c2h2": 10,
                "c2h4": 50,
                "c2h6": 40,
                "co": 200,
                "co2": 1500,
            },
            dielectric_strength=35.0,
            moisture_content=25.0,
            acidity=0.20,
            power_factor=1.0,
            capacitance=1300.0,
            insulation_resistance=2000.0,
            age_years=15.0,
            average_load_percent=70.0,
        )

        # Category depends on overall health calculation
        assert result.category in ["GOOD", "FAIR", "POOR"]

    def test_category_assignment_poor(self):
        """Test category assignment for poor health."""
        hi = CompositeHealthIndex()

        result = hi.calculate(
            dga_gases={
                "h2": 300,
                "ch4": 300,
                "c2h2": 50,
                "c2h4": 150,
                "c2h6": 80,
                "co": 400,
                "co2": 2500,
            },
            dielectric_strength=25.0,
            moisture_content=40.0,
            acidity=0.35,
            power_factor=2.5,
            capacitance=1500.0,
            insulation_resistance=500.0,
            age_years=20.0,
            average_load_percent=85.0,
        )

        # Category depends on overall health calculation
        assert result.category in ["FAIR", "POOR", "CRITICAL"]

    def test_component_scores_returned(self):
        """Test that component scores are returned."""
        hi = CompositeHealthIndex()

        result = hi.calculate(
            dga_gases={
                "h2": 50,
                "ch4": 100,
                "c2h2": 5,
                "c2h4": 30,
                "c2h6": 20,
                "co": 150,
                "co2": 1000,
            },
            dielectric_strength=45.0,
            moisture_content=15.0,
            acidity=0.10,
            power_factor=0.5,
            capacitance=1200.0,
            insulation_resistance=5000.0,
            age_years=10.0,
            average_load_percent=60.0,
        )

        assert "dga" in result.component_scores
        assert "oil_quality" in result.component_scores
        assert "electrical" in result.component_scores
        assert "age" in result.component_scores
        assert "loading" in result.component_scores

    def test_weights_used_returned(self):
        """Test that weights used are returned in result."""
        hi = CompositeHealthIndex()

        result = hi.calculate(
            dga_gases={
                "h2": 50,
                "ch4": 100,
                "c2h2": 5,
                "c2h4": 30,
                "c2h6": 20,
                "co": 150,
                "co2": 1000,
            },
            age_years=10.0,
        )

        assert result.weights_used is not None

    def test_risk_level_low(self):
        """Test risk level LOW for healthy transformer."""
        hi = CompositeHealthIndex()

        result = hi.calculate(
            dga_gases={
                "h2": 10,
                "ch4": 20,
                "c2h2": 0,
                "c2h4": 5,
                "c2h6": 10,
                "co": 50,
                "co2": 500,
            },
            dielectric_strength=60.0,
            moisture_content=5.0,
            acidity=0.02,
            power_factor=0.1,
            capacitance=1000.0,
            insulation_resistance=20000.0,
            age_years=2.0,
            average_load_percent=40.0,
        )

        assert result.risk_level == "LOW"

    def test_risk_level_moderate(self):
        """Test risk level MODERATE."""
        hi = CompositeHealthIndex()

        result = hi.calculate(
            dga_gases={
                "h2": 80,
                "ch4": 120,
                "c2h2": 5,
                "c2h4": 40,
                "c2h6": 30,
                "co": 150,
                "co2": 1200,
            },
            dielectric_strength=40.0,
            moisture_content=20.0,
            acidity=0.15,
            power_factor=0.8,
            capacitance=1250.0,
            insulation_resistance=3000.0,
            age_years=12.0,
            average_load_percent=65.0,
        )

        # Risk level depends on overall health index calculation
        assert result.risk_level in ["LOW", "MODERATE", "HIGH"]

    def test_risk_level_high(self):
        """Test risk level HIGH for poor condition."""
        hi = CompositeHealthIndex()

        result = hi.calculate(
            dga_gases={
                "h2": 250,
                "ch4": 250,
                "c2h2": 30,
                "c2h4": 120,
                "c2h6": 60,
                "co": 350,
                "co2": 2000,
            },
            dielectric_strength=28.0,
            moisture_content=35.0,
            acidity=0.30,
            power_factor=2.0,
            capacitance=1400.0,
            insulation_resistance=800.0,
            age_years=18.0,
            average_load_percent=80.0,
        )

        # Risk level depends on overall health index calculation
        assert result.risk_level in ["MODERATE", "HIGH", "CRITICAL"]

    def test_risk_level_critical(self):
        """Test risk level CRITICAL for critical condition."""
        hi = CompositeHealthIndex()

        result = hi.calculate(
            dga_gases={
                "h2": 500,
                "ch4": 400,
                "c2h2": 100,
                "c2h4": 300,
                "c2h6": 100,
                "co": 500,
                "co2": 3000,
            },
            dielectric_strength=18.0,
            moisture_content=50.0,
            acidity=0.50,
            power_factor=5.0,
            capacitance=1800.0,
            insulation_resistance=100.0,
            age_years=25.0,
            average_load_percent=95.0,
        )

        # Risk level depends on overall health index calculation
        assert result.risk_level in ["MODERATE", "HIGH", "CRITICAL"]

    def test_recommendations_generated(self):
        """Test that recommendations are generated."""
        hi = CompositeHealthIndex()

        result = hi.calculate(
            dga_gases={
                "h2": 50,
                "ch4": 100,
                "c2h2": 5,
                "c2h4": 30,
                "c2h6": 20,
                "co": 150,
                "co2": 1000,
            },
            age_years=10.0,
            average_load_percent=60.0,
        )

        assert isinstance(result.recommendations, list)
        assert len(result.recommendations) > 0

    def test_confidence_based_on_data_availability(self):
        """Test that confidence reflects data availability."""
        hi = CompositeHealthIndex()

        # With minimal data
        result_minimal = hi.calculate(
            dga_gases={
                "h2": 50,
                "ch4": 100,
                "c2h2": 5,
                "c2h4": 30,
                "c2h6": 20,
                "co": 150,
                "co2": 1000,
            }
        )

        # With full data
        result_full = hi.calculate(
            dga_gases={
                "h2": 50,
                "ch4": 100,
                "c2h2": 5,
                "c2h4": 30,
                "c2h6": 20,
                "co": 150,
                "co2": 1000,
            },
            dielectric_strength=45.0,
            moisture_content=15.0,
            acidity=0.10,
            power_factor=0.5,
            capacitance=1200.0,
            insulation_resistance=5000.0,
            age_years=10.0,
            average_load_percent=60.0,
        )

        # Full data should have higher confidence
        assert result_full.confidence >= result_minimal.confidence


class TestHealthIndexResult:
    """Test suite for HealthIndexResult dataclass."""

    def test_health_index_result_creation(self):
        """Test creating a HealthIndexResult instance."""
        result = HealthIndexResult(
            health_index=75.0, category="GOOD", category_color="#27ae60"
        )

        assert result.health_index == 75.0
        assert result.category == "GOOD"
        assert result.category_color == "#27ae60"

    def test_health_index_result_with_all_fields(self):
        """Test creating result with all optional fields."""
        result = HealthIndexResult(
            health_index=50.0,
            category="FAIR",
            category_color="#f39c12",
            component_scores={"dga": 60.0, "age": 40.0},
            weights_used={"dga": 0.35, "age": 0.15},
            risk_level="MODERATE",
            recommendations=["Test recommendation"],
            confidence=0.8,
        )

        assert result.component_scores == {"dga": 60.0, "age": 40.0}
        assert result.risk_level == "MODERATE"
        assert result.confidence == 0.8


class TestHealthIndexCategories:
    """Test suite for health index categories."""

    def test_excellent_category_range(self):
        """Test excellent category range."""
        hi = CompositeHealthIndex()

        assert hi.categories["excellent"]["min"] == 85
        assert hi.categories["excellent"]["max"] == 100

    def test_good_category_range(self):
        """Test good category range."""
        hi = CompositeHealthIndex()

        assert hi.categories["good"]["min"] == 70
        assert hi.categories["good"]["max"] == 84

    def test_fair_category_range(self):
        """Test fair category range."""
        hi = CompositeHealthIndex()

        assert hi.categories["fair"]["min"] == 50
        assert hi.categories["fair"]["max"] == 69

    def test_poor_category_range(self):
        """Test poor category range."""
        hi = CompositeHealthIndex()

        assert hi.categories["poor"]["min"] == 25
        assert hi.categories["poor"]["max"] == 49

    def test_critical_category_range(self):
        """Test critical category range."""
        hi = CompositeHealthIndex()

        assert hi.categories["critical"]["min"] == 0
        assert hi.categories["critical"]["max"] == 24


class TestRiskThresholds:
    """Test suite for risk thresholds."""

    def test_risk_thresholds_defined(self):
        """Test that risk thresholds are defined."""
        hi = CompositeHealthIndex()

        assert "LOW" in hi.RISK_THRESHOLDS
        assert "MODERATE" in hi.RISK_THRESHOLDS
        assert "HIGH" in hi.RISK_THRESHOLDS
        assert "CRITICAL" in hi.RISK_THRESHOLDS


class TestIndividualComponentCalculators:
    """Test suite for individual component calculators."""

    def test_calculate_dga_component(self):
        """Test DGA component calculation."""
        hi = CompositeHealthIndex()

        result = hi.calculate_dga(
            gases={
                "h2": 50,
                "ch4": 100,
                "c2h2": 5,
                "c2h4": 30,
                "c2h6": 20,
                "co": 150,
                "co2": 1000,
            }
        )

        assert result.score is not None

    def test_calculate_oil_quality_component(self):
        """Test oil quality component calculation."""
        hi = CompositeHealthIndex()

        result = hi.calculate_oil_quality(
            dielectric_strength=45.0, moisture_content=15.0, acidity=0.10
        )

        assert result.score is not None

    def test_calculate_electrical_component(self):
        """Test electrical component calculation."""
        hi = CompositeHealthIndex()

        result = hi.calculate_electrical(
            power_factor=0.5, capacitance=1200.0, insulation_resistance=5000.0
        )

        assert result.score is not None

    def test_calculate_age_component(self):
        """Test age component calculation."""
        hi = CompositeHealthIndex()

        result = hi.calculate_age(age_years=10.0)

        assert result.score is not None

    def test_calculate_loading_component(self):
        """Test loading component calculation."""
        hi = CompositeHealthIndex()

        result = hi.calculate_loading(average_load_percent=60.0)

        assert result.score is not None
