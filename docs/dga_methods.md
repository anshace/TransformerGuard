# DGA Interpretation Methods

TransformerGuard implements multiple DGA interpretation methods based on IEEE C57.104 and IEC 60599 standards. This document provides detailed explanations of each method.

## Overview

Dissolved Gas Analysis (DGA) is one of the most widely accepted power transformer diagnostic techniques. When transformer insulation oil overheats or undergoes electrical discharge, it decomposes into various gases that dissolve in the oil. By analyzing these gases, we can identify potential faults before they cause catastrophic failures.

## Supported Gases

| Gas | Chemical Symbol | Primary Indicator |
|-----|-----------------|-------------------|
| Hydrogen | H2 | Partial discharge, corona |
| Methane | CH4 | Thermal faults |
| Acetylene | C2H2 | Arcing, high-energy discharge |
| Ethylene | C2H4 | High-temperature thermal |
| Ethane | C2H6 | Low-temperature thermal |
| Carbon Monoxide | CO | Paper/insulation degradation |
| Carbon Dioxide | CO2 | Paper/insulation aging |

---

## Duval Triangle Method

The Duval Triangle method uses the relative percentages of CH4, C2H4, and C2H2 to classify faults into 7 zones. This method is widely used due to its simplicity and reliability.

### Triangle Coordinates

The triangle uses percentages calculated as:

```
%CH4 = CH4 / (CH4 + C2H4 + C2H2) × 100
%C2H4 = C2H4 / (CH4 + C2H4 + C2H2) × 100
%C2H2 = C2H2 / (CH4 + C2H4 + C2H2) × 100
```

### Fault Zones

| Zone | Fault Type | Description | Action |
|------|------------|-------------|--------|
| PD | Partial Discharge | Low-energy electrical discharge | Monitor closely |
| D1 | Low Energy Discharge | Spark discharge | Schedule inspection |
| D2 | High Energy Discharge | Arc discharge | Urgent action required |
| T1 | Thermal Fault <300°C | Low-temperature overheating | Monitor trend |
| T2 | Thermal Fault 300-700°C | Medium-temperature overheating | Investigate cause |
| T3 | Thermal Fault >700°C | High-temperature overheating | Immediate action |
| DT | Mixed Discharge/Thermal | Combined fault types | Priority inspection |
| N | Normal | No significant fault | Routine monitoring |

### Duval Triangle Visualization

```
              C2H2 (100%)
                 *
                / \
               /   \
              / D2  \
             /   *   \
            / D1  *   \
           /   *   *   \
          / PD * T1  *  \
         /   *   *   *   \
        / *  *   *   * *  \
       / T2  *   N   * T3 \
      /   *   *   *   *   \
     /     *  DT *       \
    /         *            \
   /           *            \
  **************************** CH4 (100%)
                 
                C2H4 (100%)
```

### Python Example

```python
from src.diagnosis import DuvalTriangle1

dga_values = {
    "h2": 150,
    "ch4": 200,
    "c2h2": 10,
    "c2h4": 50,
    "c2h6": 30,
    "co": 250,
    "co2": 1500
}

result = DuvalTriangle1.analyze(dga_values)
print(f"Fault Type: {result.fault_type}")
print(f"Confidence: {result.confidence}")
```

---

## Rogers Ratio Method

The Rogers Ratio method uses four gas ratios to classify faults. It is based on IEEE C57.104 and provides a systematic approach to fault diagnosis.

### Ratio Calculation

```
R1 = C2H2 / C2H4
R2 = CH4 / H2
R3 = C2H4 / C2H6
R4 = CO2 / CO
```

### Interpretation Codes

| Code | R1 (C2H2/C2H4) | R2 (CH4/H2) | R3 (C2H4/C2H6) | R4 (CO2/CO) | Fault Type |
|------|----------------|-------------|----------------|-------------|------------|
| 0 | <0.1 | <0.1 | <1.0 | <1.0 | Normal |
| 1 | ≥0.1 to <1.0 | ≥0.1 to <1.0 | ≥1.0 to <3.0 | ≥1.0 to <3.0 | Low-temp thermal |
| 2 | ≥1.0 to <3.0 | ≥1.0 to <3.0 | ≥3.0 | ≥3.0 | Thermal T2 |
| 3 | ≥3.0 | ≥3.0 | - | - | Thermal T3 |
| 4 | - | - | - | - | High-energy discharge |
| 5 | - | - | - | - | Low-energy discharge |
| 6 | - | - | - | - | Partial discharge |

### Python Example

```python
from src.diagnosis import RogersRatios

dga_values = {
    "h2": 150,
    "ch4": 200,
    "c2h2": 10,
    "c2h4": 50,
    "c2h6": 30,
    "co": 250,
    "co2": 1500
}

result = RogersRatios.analyze(dga_values)
print(f"Ratios: {result.ratios}")
print(f"Fault Type: {result.fault_type}")
print(f"Codes: {result.codes}")
```

---

## IEC Ratio Method

The IEC Ratio method (IEC 60599) is similar to Rogers but uses different ratio interpretation codes and thresholds.

### Ratio Calculation

```
R1 = C2H2 / C2H4
R2 = CH4 / H2
R3 = C2H4 / C2H6
R4 = CO / CO2
```

### Interpretation Matrix

| Code | R1 | R2 | R3 | R4 | Fault |
|------|----|----|----|----|-------|
| A | >1 | <1 | <1 | >1 | Partial discharge |
| B | <0.1 | <0.1 | <1 | >1 | Low-energy discharge |
| C | <0.1 | 0.1-1.0 | <1 | <1 | High-energy discharge |
| D | <0.1 | >1 | 1-3 | <1 | Thermal fault T1 |
| E | <0.1 | >1 | >3 | <1 | Thermal fault T2 |
| F | <0.1 | >1 | >3 | >1 | Thermal fault T3 |
| 0 | All ratios in normal range | Normal |

### Python Example

```python
from src.diagnosis import IecRatios

dga_values = {
    "h2": 150,
    "ch4": 200,
    "c2h2": 10,
    "c2h4": 50,
    "c2h6": 30,
    "co": 250,
    "co2": 1500
}

result = IecRatios.analyze(dga_values)
print(f"Ratios: {result.ratios}")
print(f"Fault Type: {result.fault_type}")
print(f"Code: {result.code}")
```

---

## Key Gas Method

The Key Gas method identifies faults based on the dominant gas or gas combination. It provides quick identification of major fault types.

### Fault Classification by Dominant Gas

| Dominant Gas | Fault Type | Description |
|--------------|-------------|-------------|
| H2 | Partial Discharge | Corona, low-energy discharge |
| CH4 | Low-Temp Thermal | Overheating <300°C |
| C2H4 | High-Temp Thermal | Overheating >300°C |
| C2H2 | Electrical Discharge | Arcing, sparking |
| CO | Paper Degradation | Cellulose breakdown |
| CO + CH4 | Thermal/Combined | Mixed fault |

### Gas Concentration Thresholds

Based on IEEE C57.104:

| Gas | Level 1 (Monitor) | Level 2 (Action) | Level 3 (Critical) |
|-----|-------------------|------------------|-------------------|
| H2 | 100 ppm | 700 ppm | 1800 ppm |
| CH4 | 120 | 400 | 1000 |
| C2H2 | 35 | 50 | 80 |
| C2H4 | 50 | 100 | 200 |
| C2H6 | 65 | 150 | 360 |
| CO | 350 | 570 | 1400 |
| CO2 | 2500 | 4000 | 10000 |
| TDCG | 2500 | 5000 | 10000 |

### Python Example

```python
from src.diagnosis import KeyGasMethod

dga_values = {
    "h2": 150,
    "ch4": 200,
    "c2h2": 10,
    "c2h4": 50,
    "c2h6": 30,
    "co": 250,
    "co2": 1500
}

result = KeyGasMethod.analyze(dga_values)
print(f"Key Gas: {result.key_gas}")
print(f"Fault Type: {result.fault_type}")
print(f"Dominant Gases: {result.dominant_gases}")
```

---

## Doernenburg Ratio Method

The Doernenburg method uses the same four ratios as Rogers but with stricter requirements for diagnosis.

### Ratio Calculation

Same as Rogers ratios, but requires at least two ratios to exceed threshold values for a valid diagnosis.

### Interpretation

| Ratio | Normal | Fault Indicative |
|-------|--------|------------------|
| C2H2/C2H4 | <0.1 | ≥1.0 |
| CH4/H2 | <0.1 | ≥1.0 |
| C2H4/C2H6 | <1.0 | ≥3.0 |
| CO/CO2 | <0.3 | ≥1.0 |

### Python Example

```python
from src.diagnosis import Doernenburg

dga_values = {
    "h2": 150,
    "ch4": 200,
    "c2h2": 10,
    "c2h4": 50,
    "c2h6": 30,
    "co": 250,
    "co2": 1500
}

result = Doernenburg.analyze(dga_values)
print(f"Ratios: {result.ratios}")
print(f"Fault Type: {result.fault_type}")
```

---

## Multi-Method Ensemble

The Multi-Method Diagnosis combines all methods using weighted voting for consensus diagnosis.

### Method Weights

| Method | Weight | Rationale |
|--------|--------|-----------|
| Duval Triangle | 0.35 | Most widely validated |
| Rogers Ratios | 0.25 | IEEE standard |
| IEC Ratios | 0.20 | IEC standard |
| Key Gas | 0.10 | Quick identification |
| Doernenburg | 0.10 | Conservative backup |

### Consensus Logic

1. Run all individual methods
2. Calculate weighted confidence scores
3. Return fault type with highest weighted score
4. Provide explanation combining all method results

### Python Example

```python
from src.diagnosis import MultiMethodDiagnosis

dga_values = {
    "h2": 150,
    "ch4": 200,
    "c2h2": 10,
    "c2h4": 50,
    "c2h6": 30,
    "co": 250,
    "co2": 1500
}

result = MultiMethodDiagnosis.analyze(dga_values)
print(f"Consensus Fault: {result.fault_type}")
print(f"Confidence: {result.confidence}")
print(f"Method Results: {result.method_results}")
```

### Output Structure

```json
{
  "fault_type": "T2",
  "fault_type_name": "Thermal Fault 300-700°C",
  "fault_confidence": 0.75,
  "tdcg": 2190,
  "explanation": "Medium-temperature thermal fault (300-700°C)...",
  "method_results": {
    "duval": {"fault_type": "T2", "confidence": 0.82},
    "rogers": {"fault_type": "T2", "codes": "0 1 1 1"},
    "iec": {"fault_type": "T2", "codes": "A 0 1 1"},
    "key_gas": {"fault_type": "T1-T2", "dominant_gases": ["CH4", "CO"]},
    "doernenburg": {"fault_type": "T2", "ratios": {...}}
  },
  "gas_percentages": {
    "h2": 6.85,
    "ch4": 9.13,
    "c2h2": 0.46,
    "c2h4": 2.28,
    "c2h6": 1.37,
    "co": 11.42,
    "co2": 68.49
  }
}
```

---

## Total Dissolved Combustible Gas (TDCG)

TDCG is the sum of all combustible gases:

```
TDCG = H2 + CH4 + C2H2 + C2H4 + C2H6 + CO + CO2
```

### TDCG Levels (IEEE C57.104)

| Level | TDCG Range | Action |
|-------|------------|--------|
| Condition 1 | < 720 ppm | Continue normal operation |
| Condition 2 | 721-1920 ppm | Monitor more frequently |
| Condition 3 | 1921-4630 ppm | Schedule inspection |
| Condition 4 | 4631-10890 ppm | Urgent action required |
| Condition 5 | > 10890 ppm | Immediate action |

---

## Recommendations by Fault Type

| Fault Type | Recommendation |
|------------|----------------|
| PD | Monitor DGA monthly; check for moisture |
| D1 | Inspect bushings, tap changers |
| D2 | Immediate inspection; consider offline testing |
| T1 | Check cooling system, reduce load |
| T2 | Investigate overheating source |
| T3 | Critical: Schedule overhaul |
| DT | Comprehensive inspection required |

---

## References

- IEEE C57.104-2019: Guide for Interpretation of Gases Generated in Mineral Oil-Immersed Transformers
- IEC 60599: Mineral oil-impregnated electrical equipment in service - Guide to the interpretation of dissolved and free gases analysis
