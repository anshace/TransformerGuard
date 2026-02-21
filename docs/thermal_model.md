# IEEE C57.91 Thermal Model

## Overview

The thermal model implements IEEE C57.91-2011 (Guide for Loading Mineral-Oil-Immersed Transformers) for calculating:
- Hot-spot temperature
- Top-oil temperature
- Insulation aging acceleration
- Loss-of-life estimation

## Key Concepts

### Temperature Definitions

| Temperature | Symbol | Description |
|-------------|--------|-------------|
| Ambient Temperature | θamb | Surrounding air temperature |
| Top-Oil Temperature | θtop | Temperature of oil at top of tank |
| Hot-Spot Temperature | θh | Maximum winding temperature |
| Hot-Spot Gradient | Δθh | Temperature difference between hot-spot and top-oil |
| Top-Oil Rise | Δθtop | Temperature rise of top-oil over ambient |

### Reference Conditions

Per IEEE C57.91-2011:
- Reference hot-spot temperature: **110°C**
- Normal life expectancy: **180,000 hours** (20.5 years)
- Aging doubling temperature: **6°C** (every 6°C above reference doubles aging rate)

---

## Key Equations

### Hot-Spot Temperature

```
θh = θamb + Δθtop + Δθh
```

Where:
- θamb = Ambient temperature (°C)
- Δθtop = Top-oil temperature rise over ambient (°C)
- Δθh = Hot-spot to top-oil gradient (°C)

### Top-Oil Temperature Rise

```
Δθtop = Δθtop,r × [K^2 × (τtop/τw) + 1] / [K^2 × (τtop/τw) + 1]
```

Simplified steady-state form:
```
Δθtop = Δθtop,r × (K^2 × R + 1) / (R + 1)
```

Where:
- K = Load factor (load/rated load)
- R = Ratio of load losses to no-load losses
- Δθtop,r = Rated top-oil rise (°C)
- τtop = Top-oil time constant (minutes)
- τw = Winding time constant (minutes)

### Aging Acceleration Factor

```
F_AA = exp[15000/383 - 15000/(θh + 273)]
```

This formula is derived from the Arrhenius equation for insulation aging.

Alternative form using reference temperature:
```
F_AA = 2^((θh - 110) / 6)
```

Where:
- θh = Hot-spot temperature (°C)
- 110°C = Reference hot-spot temperature
- 6°C = Temperature doubling factor

### Loss of Life Calculation

```
LOL = Σ(F_AA,i × Δt_i) / 180000 × 100%
```

Where:
- LOL = Percentage of normal life consumed
- F_AA,i = Aging acceleration factor for time period i
- Δt_i = Duration of time period i (hours)
- 180000 = Normal life expectancy in hours

---

## Cooling Modes

TransformerGuard supports the following cooling modes defined in IEEE C57.12.00:

### Cooling Mode Parameters

| Mode | Description | Top-Oil Rise (°C) | Hot-Spot Gradient (°C) | Time Constant (min) |
|------|-------------|-------------------|------------------------|-------------------|
| ONAN | Oil Natural, Air Natural | 45 | 65 | 150 |
| ONAF | Oil Natural, Air Forced | 40 | 55 | 90 |
| OFAF | Oil Forced, Air Forced | 35 | 50 | 45 |
| ODAF | Oil Directed, Air Forced | 30 | 45 | 30 |

### Cooling Mode Selection

```python
from src.thermal import IEEE_C57_91_DEFAULTS, COOLING_MODES

# Select cooling mode parameters
mode = "ONAF"
params = COOLING_MODES[mode]
print(f"Top-oil rise: {params['top_oil_rise']}°C")
print(f"Hot-spot gradient: {params['hotspot_gradient']}°C")
```

---

## Loading Capability

### Emergency Loading

IEEE C57.91 defines emergency loading profiles:

| Emergency Level | Load Factor | Duration | Max Hot-Spot |
|-----------------|-------------|----------|--------------|
| Short-term | 1.5× | 30 min | 140°C |
| Long-term | 1.3× | 4 hours | 130°C |
| Planned | 1.25× | Daily 8h | 120°C |

### Overload Calculation

```python
from src.thermal import LoadingCapability

capability = LoadingCapability(
    rated_mva=25,
    rated_temp_rise=65,
    ambient_temp=35,
    cooling_mode="ONAF"
)

# Calculate permissible load at given conditions
permissible = capability.calculate_load(
    desired_hotspot=120,
    duration_hours=4
)
print(f"Permissible load: {permissible} MVA")
```

---

## Transformer Parameters

### Required Parameters

| Parameter | Symbol | Unit | Typical Values |
|-----------|--------|------|----------------|
| Rated MVA | S_rated | MVA | 10-500 |
| Rated Voltage | V_rated | kV | 11-500 |
| No-load Loss | P_nl | kW | Varies |
| Load Loss | P_ll | kW | Varies |
| Top-Oil Rise | Δθtop,r | °C | 30-65 |
| Hot-Sot Gradient | Δθh,r | °C | 15-80 |
| Oil Time Constant | τtop | min | 30-210 |
| Winding Time Constant | τw | min | 3-10 |

### Winding Exponent

The winding hot-spot rise varies with load factor K according to:

```
Δθh = Δθh,r × K^y
```

Where y is the winding exponent:

| Cooling Mode | Exponent (y) |
|--------------|---------------|
| ONAN | 0.8 |
| ONAF | 0.8 |
| OFAF | 0.9 |
| ODAF | 1.0 |

---

## Python Usage

### Basic Hot-Spot Calculation

```python
from src.thermal import IEEE_C57_91, TransformerParameters

# Create transformer parameters
params = TransformerParameters(
    rated_mva=25,
    rated_voltage=138,
    cooling_mode="ONAF",
    top_oil_rise=40,
    hotspot_gradient=55,
    ambient_temp=30,
    load_factor=0.8
)

# Calculate temperatures
model = IEEE_C57_91(params)
result = model.calculate()

print(f"Top-oil temperature: {result.top_oil_temp:.1f}°C")
print(f"Hot-spot temperature: {result.hotspot_temp:.1f}°C")
print(f"Aging acceleration: {result.aging_factor:.2f}")
print(f"Daily loss of life: {result.daily_lol:.3f}%")
```

### Calculate Loss of Life

```python
from src.thermal import LossOfLife

lol_calculator = LossOfLife(
    normal_life_hours=180000,
    reference_temp=110,
    doubling_temp=6
)

# Calculate loss of life for given temperature profile
profile = [
    {"temp": 95, "hours": 12},
    {"temp": 110, "hours": 6},
    {"temp": 120, "hours": 6}
]

daily_lol = lol_calculator.calculate(profile)
print(f"Daily loss of life: {daily_lol:.3f}%")
```

### Complete Example

```python
from src.thermal import IEEE_C57_91, TransformerParameters

# Define transformer
transformer = TransformerParameters(
    rated_mva=50,
    rated_voltage=230,
    cooling_mode="ONAF",
    no_load_loss=25,
    load_loss=180,
    top_oil_time_constant=90,
    winding_time_constant=5,
    ambient_temp=30,
    load_factor=0.75
)

# Initialize model
model = IEEE_C57_91(transformer)

# Calculate all thermal parameters
results = model.calculate()

print("=== Thermal Analysis Results ===")
print(f"Ambient Temperature: {results.ambient_temp}°C")
print(f"Top-Oil Temperature: {results.top_oil_temp:.1f}°C")
print(f"Hot-Spot Temperature: {results.hotspot_temp:.1f}°C")
print(f"Aging Acceleration: {results.aging_factor:.2f}")
print(f"Hourly Loss of Life: {results.hourly_lol:.4f}%")
print(f"Daily Loss of Life: {results.daily_lol:.3f}%")
print(f"Monthly Loss of Life: {results.monthly_lol:.2f}%")
print(f"Annual Loss of Life: {results.annual_lol:.1f}%")

# Calculate RUL
years_remaining = (100 - results.cumulative_lol) / results.annual_lol
print(f"Remaining Life: {years_remaining:.1f} years")
```

---

## Output Parameters

The thermal model returns:

```python
@dataclass
class ThermalResult:
    ambient_temp: float          # °C
    top_oil_temp: float          # °C
    hotspot_temp: float         # °C
    top_oil_rise: float         # °C
    hotspot_gradient: float     # °C
    aging_factor: float         # Unitless multiplier
    hourly_lol: float           # % per hour
    daily_lol: float            # % per day
    monthly_lol: float          # % per month
    annual_lol: float           # % per year
    cumulative_lol: float       # % of life consumed
    rul_years: float            # Remaining life in years
```

---

## Aging Model Integration

The thermal model integrates with the aging model to provide comprehensive life assessment:

```python
from src.thermal import AgingModel

aging = AgingModel(
    reference_temp=110,
    doubling_temp=6,
    normal_life_hours=180000
)

# Calculate aging for varying temperature
temperature_profile = [90, 100, 110, 120, 130]
aging_factors = [aging.calculate_factor(t) for t in temperature_profile]

print("Temperature vs Aging Factor:")
for temp, factor in zip(temperature_profile, aging_factors):
    print(f"  {temp}°C -> {factor:.2f}x aging rate")
```

### Temperature-Aging Relationship

| Hot-Spot Temp | Aging Factor | Effect |
|---------------|--------------|--------|
| 80°C | 0.12× | Very slow aging |
| 95°C | 0.5× | Below normal |
| 110°C | 1.0× | Normal (reference) |
| 116°C | 2.0× | Double normal |
| 122°C | 4.0× | 4× normal |
| 128°C | 8.0× | 8× normal |
| 134°C | 16× | 16× normal |

---

## Configuration

Thermal parameters are configured in `config/thermal_params.yaml`:

```yaml
ieee_c57_91:
  reference_hotspot: 110  # °C
  min_life_expectancy_hours: 180000
  aging_doubling_temp: 6  # °C
  oil_exponent: 0.8
  winding_exponent: 0.8
  top_oil_time_constant: 150  # minutes
  winding_time_constant: 5  # minutes

cooling_modes:
  ONAN:
    top_oil_rise: 45
    hotspot_gradient: 65
  ONAF:
    top_oil_rise: 40
    hotspot_gradient: 55
  OFAF:
    top_oil_rise: 35
    hotspot_gradient: 50
  ODAF:
    top_oil_rise: 30
    hotspot_gradient: 45
```

---

## References

- IEEE C57.91-2011: IEEE Guide for Loading Mineral-Oil-Immersed Transformers
- IEEE C57.12.00-2015: IEEE Standard General Requirements for Liquid-Immersed Distribution, Power, and Regulating Transformers
