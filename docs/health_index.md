# Health Index Calculation

## Overview

The Health Index is a comprehensive 0-100 score that combines multiple diagnostic factors to provide an overall assessment of transformer condition. It integrates DGA analysis, oil quality, electrical tests, age, and loading history into a single, easy-to-understand metric.

## Component Weights

The composite health index combines five major components with default weights:

| Factor | Weight | Description |
|--------|--------|-------------|
| DGA Score | 35% | Fault diagnosis from dissolved gas analysis |
| Oil Quality | 20% | Dielectric strength, moisture, acidity |
| Electrical Tests | 15% | Power factor, insulation resistance |
| Age | 15% | Time since manufacture |
| Loading History | 15% | Historical loading percentage |

These weights can be customized in `config/health_index_weights.yaml`.

---

## Category Definitions

The health index is categorized into five health states:

| Category | Range | Color | Description |
|----------|-------|-------|-------------|
| EXCELLENT | 85-100 | Green | New or like-new condition |
| GOOD | 70-84 | Light Green | Normal aging, no concerns |
| FAIR | 50-69 | Yellow | Some degradation, monitor closely |
| POOR | 25-49 | Orange | Significant degradation, plan action |
| CRITICAL | 0-24 | Red | Immediate action required |

---

## Component Scoring

### DGA Score (0-100)

The DGA score is based on fault type and Total Dissolved Combustible Gas (TDCG) levels.

#### Scoring Algorithm

1. **Fault Type Score** (0-100):
   - Normal: 100
   - Partial Discharge (PD): 60
   - Low-Energy Discharge (D1): 50
   - High-Energy Discharge (D2): 30
   - Thermal Fault T1: 70
   - Thermal Fault T2: 60
   - Thermal Fault T3: 40
   - Mixed Discharge/Thermal (DT): 45
   - Undetermined: 50

2. **TDCG Level Modifier**:
   - < 720 ppm: × 1.0
   - 721-1920 ppm: × 0.9
   - 1921-4630 ppm: × 0.75
   - 4631-10890 ppm: × 0.5
   - > 10890 ppm: × 0.25

3. **Trend Modifier**:
   - Decreasing: × 1.1
   - Stable: × 1.0
   - Increasing: × 0.85

#### Calculation

```python
from src.health_index import DGAScoreCalculator

dga_data = {
    "fault_type": "T2",
    "tdcg": 1500,
    "trend": "stable"
}

score = DGAScoreCalculator.calculate(dga_data)
print(f"DGA Score: {score.score}")
```

---

### Oil Quality Score (0-100)

Oil quality is assessed based on multiple parameters:

#### Parameters

| Parameter | Method | Good | Fair | Poor |
|-----------|--------|------|------|------|
| Dielectric Strength | ASTM D877 | >30 kV | 25-30 kV | <25 kV |
| Moisture Content | ASTM D1533 | <0.03% | 0.03-0.05% | >0.05% |
| Acidity Number | ASTM D974 | <0.1 mgKOH/g | 0.1-0.2 | >0.2 |
| Interfacial Tension | ASTM D971 | >40 mN/m | 25-40 | <25 |
| Color | ASTM D1500 | <1.0 | 1.0-2.0 | >2.0 |

#### Scoring Algorithm

```python
from src.health_index import OilQualityScore

oil_data = {
    "dielectric_strength": 28,  # kV
    "moisture": 0.04,  # %
    "acidity": 0.15,  # mgKOH/g
    "ift": 35,  # mN/m
    "color": 1.5
}

score = OilQualityScore.calculate(oil_data)
print(f"Oil Quality Score: {score.score}")
```

---

### Electrical Score (0-100)

Electrical tests assess insulation condition:

#### Parameters

| Parameter | Method | Good | Fair | Poor |
|-----------|--------|------|------|------|
| Power Factor 60Hz | ASTM D924 | <0.5% | 0.5-1.0% | >1.0% |
| Insulation Resistance | IEEE 95 | >1000 MΩ | 100-1000 MΩ | <100 MΩ |
| Capacitance | - | ±5% of initial | ±5-10% | >±10% |
| Turns Ratio | - | ±0.5% | ±0.5-1% | >±1% |

#### Scoring Algorithm

```python
from src.health_index import ElectricalScore

electrical_data = {
    "power_factor": 0.6,  # %
    "insulation_resistance": 5000,  # MΩ
    "capacitance_deviation": 3,  # %
    "turns_ratio_deviation": 0.3  # %
}

score = ElectricalScore.calculate(electrical_data)
print(f"Electrical Score: {score.score}")
```

---

### Age Score (0-100)

The age score reflects expected remaining life based on transformer age.

#### Expected Life Curves

| Age (years) | Score | Notes |
|-------------|-------|-------|
| 0-5 | 100-90 | Like new |
| 5-10 | 90-80 | Minor aging |
| 10-15 | 80-70 | Normal aging |
| 15-20 | 70-60 | Accelerated aging |
| 20-25 | 60-40 | Significant aging |
| 25-30 | 40-25 | End of life |
| >30 | <25 | Replace |

#### Calculation

```python
from src.health_index import AgeScore

age_data = {
    "manufacture_date": "2010-06-15",
    "expected_life": 30,  # years
    "maintenance_history": "good"
}

score = AgeScore.calculate(age_data)
print(f"Age Score: {score.score}")
```

---

### Loading Score (0-100)

Loading history affects insulation aging:

#### Parameters

| Loading Profile | Score Impact |
|-----------------|--------------|
| Always <50% | +10 |
| Typically 50-70% | +5 |
| Typically 70-85% | 0 |
| Frequently 85-100% | -10 |
| Occasional overloads | -20 |

#### Calculation

```python
from src.health_index import LoadingScore

loading_data = {
    "avg_load_factor": 0.75,
    "max_load_factor": 1.1,
    "hours_above_100": 100,
    "load_cycles": 1000
}

score = LoadingScore.calculate(loading_data)
print(f"Loading Score: {score.score}")
```

---

## Composite Health Index

### Calculation Formula

```
HI = (DGA × 0.35) + (Oil × 0.20) + (Electrical × 0.15) + (Age × 0.15) + (Loading × 0.15)
```

### Python Usage

```python
from src.health_index import CompositeHealthIndex

# Calculate composite health index
health_index = CompositeHealthIndex(
    transformer_id=1,
    dga_score=85,
    oil_quality_score=75,
    electrical_score=80,
    age_score=70,
    loading_score=82
)

result = health_index.calculate()

print(f"Health Index: {result.health_index}")
print(f"Category: {result.category}")
print(f"Risk Level: {result.risk_level}")
print(f"Confidence: {result.confidence}")
```

### Output Structure

```python
@dataclass
class HealthIndexResult:
    health_index: float           # 0-100
    category: str                # EXCELLENT, GOOD, FAIR, POOR, CRITICAL
    category_color: str          # Color code
    component_scores: Dict[str, float]
    weights_used: Dict[str, float]
    risk_level: str              # LOW, MEDIUM, HIGH, CRITICAL
    recommendations: List[str]
    confidence: float            # 0-1
    calculation_date: datetime
```

---

## Risk Assessment

### Risk Matrix

| HI Range | Risk Level | Action |
|----------|------------|--------|
| 85-100 | LOW | Routine monitoring |
| 70-84 | LOW | Normal maintenance |
| 50-69 | MEDIUM | Increase monitoring frequency |
| 25-49 | HIGH | Plan corrective action |
| 0-24 | CRITICAL | Immediate action required |

---

## Trend Analysis

The health index can be tracked over time to identify degradation patterns:

```python
from src.health_index import TrendAnalyzer

# Get health history
history = [
    {"date": "2024-01-01", "health_index": 85},
    {"date": "2024-04-01", "health_index": 82},
    {"date": "2024-07-01", "health_index": 78},
    {"date": "2024-10-01", "health_index": 75},
]

trend = TrendAnalyzer.analyze(history)
print(f"Trend: {trend.direction}")  # decreasing
print(f"Rate: {trend.rate_per_year}%")  # -4% per year
print(f"Prediction: {trend.prediction}")  # 70 at next assessment
```

### Trend Indicators

| Trend | Description | Rate |
|-------|-------------|------|
| STABLE | No significant change | <3%/year |
| SLOW_DECLINE | Minor degradation | 3-5%/year |
| DECLINE | Moderate degradation | 5-10%/year |
| RAPID_DECLINE | Accelerated degradation | >10%/year |

---

## Configuration

Health index weights can be customized in `config/health_index_weights.yaml`:

```yaml
weights:
  dga: 0.35
  oil_quality: 0.20
  electrical: 0.15
  age: 0.15
  loading: 0.15

categories:
  excellent:
    min: 85
    max: 100
    color: "#2ecc71"
  good:
    min: 70
    max: 84
    color: "#27ae60"
  fair:
    min: 50
    max: 69
    color: "#f39c12"
  poor:
    min: 25
    max: 49
    color: "#e67e22"
  critical:
    min: 0
    max: 24
    color: "#e74c3c"
```

---

## Recommendations

Based on health index, TransformerGuard provides actionable recommendations:

| Category | Recommendations |
|----------|-----------------|
| EXCELLENT | Continue routine maintenance |
| GOOD | Continue scheduled maintenance, monitor trends |
| FAIR | Increase monitoring frequency, consider condition-based maintenance |
| POOR | Plan for replacement/refurbishment, reduce loading if possible |
| CRITICAL | Immediate investigation, possible shutdown for inspection |

---

## References

- IEEE C57.152-2013: IEEE Guide for Diagnostic Field Testing of Fluid-Filled Power Transformers
- CIGRE Technical Brochure 761: Transformer Reliability Survey
- IEC 60422: Mineral insulating oils in electrical equipment - Monitoring and maintenance
