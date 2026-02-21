# ⚡ TransformerGuard — The Minimum Viable Product

## A Focused, Buildable Transformer Health Intelligence System

---

## 🎯 What Exactly Are We Building?

**One product. One problem. One measurable outcome.**

> **TransformerGuard**: An AI-powered transformer health scoring and failure prediction system that ingests DGA (Dissolved Gas Analysis), thermal, and load data to output a 0–100 health score, a remaining useful life estimate, and plain-English maintenance recommendations.

### Why THIS and Nothing Else?

| Reason                            | Detail                                                                                                                                                                                                                                                              |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Proven science**                | Dissolved Gas Analysis (DGA) is one of the most widely accepted power transformer diagnostic techniques currently employed by electricity utilities worldwide.                                                                                                      |
| **Clear fault signal**            | Much like a blood sample can shed light on a person's health, DGA of transformer oil can provide information on past, historical events (faults) that have happened in a transformer as well as active conditions, evidenced by an increasing trend in fault gases. |
| **Known physics**                 | IEEE C57.91 quantifies exactly how temperature shortens insulation life. Every 6 °C rise in the hot-spot temperature above 110 °C roughly cuts insulation life in half.                                                                                             |
| **Existing standards**            | IEEE C57.104 provides guidelines for the interpretation of dissolved gas analysis (DGA) of insulating fluids in power transformers.                                                                                                                                 |
| **Real life expectancy baseline** | The RUL determined in accordance with IEEE Std C57.12.00 represents the industry-proven system with the time length of 180,000 hours (20.5 years), known as "minimum life expectancy".                                                                              |
| **Time to catch faults**          | Outside of the rare occasion of an extremely aggressive fault leading to catastrophic failure, the vast majority of failures are the result of an issue that intensifies over time. This intensification often takes days, weeks, or months.                        |

---

## 🏗 Architecture — Lean, Focused, Deployable

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    TRANSFORMERGUARD ARCHITECTURE                         │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────  DATA SOURCES  ────────────────────────┐      │
│  │                                                                │      │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │      │
│  │  │  DGA Lab      │  │  SCADA /     │  │  Weather     │         │      │
│  │  │  Reports      │  │  Sensor      │  │  (Open-Meteo)│         │      │
│  │  │  (CSV/Excel)  │  │  Logs        │  │  Free API    │         │      │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │      │
│  │         │                 │                  │                 │      │
│  └─────────┼─────────────────┼──────────────────┼─────────────────┘      │
│            │                 │                  │                        │
│            └─────────────────┼──────────────────┘                        │
│                              │                                           │
│                 ┌────────────▼────────────┐                              │
│                 │   INGESTION & STORAGE   │                              │
│                 │                         │                              │
│                 │  • CSV/Excel Parser     │                              │
│                 │  • Data Validator       │                              │
│                 │  • SQLite / PostgreSQL  │                              │
│                 └────────────┬────────────┘                              │
│                              │                                           │
│         ┌────────────────────┼────────────────────┐                      │
│         │                    │                    │                      │
│  ┌──────▼──────┐    ┌───────▼───────┐    ┌───────▼───────┐              │
│  │   MODULE 1  │    │   MODULE 2    │    │   MODULE 3    │              │
│  │             │    │               │    │               │              │
│  │  DGA FAULT  │    │   THERMAL     │    │   HEALTH      │              │
│  │  DIAGNOSIS  │    │   MODEL       │    │   INDEX       │              │
│  │             │    │               │    │   ENGINE      │              │
│  │ • Duval     │    │ • IEEE C57.91 │    │               │              │
│  │   Triangle  │    │   Hot-spot    │    │ • Weighted    │              │
│  │ • Rogers    │    │ • Loss-of-    │    │   multi-param │              │
│  │   Ratios    │    │   life calc   │    │ • 0-100 score │              │
│  │ • IEC 60599 │    │ • Aging accel │    │ • Trend       │              │
│  │ • ML        │    │   factor      │    │   analysis    │              │
│  │   Classifier│    │               │    │               │              │
│  └──────┬──────┘    └───────┬───────┘    └───────┬───────┘              │
│         │                   │                    │                      │
│         └───────────────────┼────────────────────┘                      │
│                             │                                           │
│                ┌────────────▼────────────┐                               │
│                │    MODULE 4             │                               │
│                │    PREDICTION ENGINE    │                               │
│                │                         │                               │
│                │  • Remaining Useful     │                               │
│                │    Life (RUL) estimate  │                               │
│                │  • Failure probability  │                               │
│                │    (30/60/90 day)       │                               │
│                │  • Anomaly detection    │                               │
│                │  • Trend forecasting    │                               │
│                └────────────┬────────────┘                               │
│                             │                                           │
│                ┌────────────▼────────────┐                               │
│                │    MODULE 5             │                               │
│                │    REPORT GENERATOR     │                               │
│                │                         │                               │
│                │  • Rule-based NL        │                               │
│                │    explanations         │                               │
│                │  • PDF report output    │                               │
│                │  • Alert prioritization │                               │
│                │  • Action recom-        │                               │
│                │    mendations           │                               │
│                └────────────┬────────────┘                               │
│                             │                                           │
│                ┌────────────▼────────────┐                               │
│                │    DASHBOARD (FastAPI   │                               │
│                │    + Streamlit/React)   │                               │
│                │                         │                               │
│                │  • Transformer fleet    │                               │
│                │    overview             │                               │
│                │  • Individual health    │                               │
│                │    detail               │                               │
│                │  • Alert timeline       │                               │
│                │  • Trend charts         │                               │
│                └─────────────────────────┘                               │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Critical Architecture Decision: NO LLM in the MVP

Instead of an LLM (which hallucinates and utilities won't trust), use a **template-based explanation engine** grounded in IEEE standards:

```
IF health_score < 30 AND duval_result == "T2_thermal_fault":
    explanation = f"""
    ALERT: Transformer {name} shows significant thermal degradation.
    - DGA Analysis: Duval Triangle indicates T2 thermal fault (300-700°C)
    - Key gas: Ethylene elevated at {c2h4_ppm} ppm (threshold: {threshold} ppm)
    - Hot-spot temperature: {hotspot}°C (IEEE C57.91 limit: 110°C)
    - Estimated RUL: {rul_days} days at current loading

    RECOMMENDED ACTIONS:
    1. Reduce loading to {recommended_load}% within 48 hours
    2. Schedule oil sampling for DGA confirmation
    3. Inspect cooling system (fans, pumps, radiators)
    4. Plan outage for internal inspection within {urgency} days
    """
```

This is **deterministic, traceable, auditable, and standards-compliant** — exactly what utilities require.

---

## 📦 Project Structure

```
TransformerGuard/
├── README.md
├── requirements.txt
├── setup.py
├── docker-compose.yml
├── .env.example
│
├── config/
│   ├── settings.yaml              # App configuration
│   ├── dga_thresholds.yaml        # IEEE C57.104 gas thresholds
│   ├── thermal_params.yaml        # IEEE C57.91 thermal parameters
│   └── health_index_weights.yaml  # HI scoring weights
│
├── data/
│   ├── sample/                    # Sample datasets for demo
│   │   ├── dga_samples.csv
│   │   ├── load_profiles.csv
│   │   └── transformer_fleet.csv
│   ├── raw/                       # User-uploaded raw data
│   ├── processed/                 # Cleaned, validated data
│   └── models/                    # Trained ML model artifacts
│
├── src/
│   ├── __init__.py
│   │
│   ├── ingestion/                 # Data intake
│   │   ├── __init__.py
│   │   ├── csv_parser.py          # Parse DGA lab reports
│   │   ├── excel_parser.py        # Parse utility spreadsheets
│   │   ├── weather_client.py      # Open-Meteo API client
│   │   ├── data_validator.py      # Schema & range validation
│   │   └── synthetic_generator.py # Generate realistic test data
│   │
│   ├── diagnosis/                 # DGA Fault Diagnosis (Module 1)
│   │   ├── __init__.py
│   │   ├── duval_triangle.py      # Duval Triangle Method 1, 4, 5
│   │   ├── duval_pentagon.py      # Duval Pentagon Method
│   │   ├── rogers_ratios.py       # Rogers Ratio Method
│   │   ├── iec_ratios.py          # IEC 60599 Ratio Method
│   │   ├── doernenburg.py         # Doernenburg Ratio Method
│   │   ├── key_gas.py             # Key Gas Method
│   │   ├── multi_method.py        # Ensemble diagnosis (scoring)
│   │   └── ml_classifier.py       # ML-based fault classification
│   │
│   ├── thermal/                   # Thermal Model (Module 2)
│   │   ├── __init__.py
│   │   ├── ieee_c57_91.py         # IEEE C57.91 thermal model
│   │   ├── hotspot_calculator.py  # Hot-spot temperature estimation
│   │   ├── aging_model.py         # Insulation aging acceleration
│   │   ├── loss_of_life.py        # Cumulative loss-of-life calc
│   │   └── loading_capability.py  # Dynamic loading limits
│   │
│   ├── health_index/              # Health Index Engine (Module 3)
│   │   ├── __init__.py
│   │   ├── dga_score.py           # DGA factor score (0-4)
│   │   ├── oil_quality_score.py   # Oil quality score
│   │   ├── electrical_score.py    # Electrical test scores
│   │   ├── age_score.py           # Age-based degradation
│   │   ├── loading_score.py       # Historical loading score
│   │   ├── composite_hi.py        # Weighted composite HI (0-100)
│   │   └── trend_analyzer.py      # HI trend over time
│   │
│   ├── prediction/                # Prediction Engine (Module 4)
│   │   ├── __init__.py
│   │   ├── rul_estimator.py       # Remaining Useful Life
│   │   ├── failure_probability.py # P(failure) at 30/60/90 days
│   │   ├── anomaly_detector.py    # Statistical anomaly detection
│   │   ├── gas_trend_forecast.py  # DGA gas trend forecasting
│   │   └── models/
│   │       ├── gradient_boost.py  # XGBoost/LightGBM model
│   │       ├── random_forest.py   # Random Forest classifier
│   │       └── time_series.py     # ARIMA / Prophet forecasting
│   │
│   ├── reporting/                 # Report Generator (Module 5)
│   │   ├── __init__.py
│   │   ├── template_engine.py     # Rule-based NL explanations
│   │   ├── alert_generator.py     # Priority-ranked alerts
│   │   ├── action_recommender.py  # Maintenance recommendations
│   │   ├── pdf_report.py          # PDF report generation
│   │   └── templates/
│   │       ├── alert_templates.yaml
│   │       ├── recommendation_templates.yaml
│   │       └── report_template.html
│   │
│   ├── api/                       # REST API
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI application
│   │   ├── routes/
│   │   │   ├── transformers.py    # Transformer CRUD
│   │   │   ├── dga.py             # DGA data upload & analysis
│   │   │   ├── health.py          # Health index endpoints
│   │   │   ├── predictions.py     # RUL & failure probability
│   │   │   ├── alerts.py          # Alert management
│   │   │   └── reports.py         # Report generation
│   │   └── schemas/
│   │       ├── transformer.py     # Pydantic models
│   │       ├── dga.py
│   │       └── health.py
│   │
│   └── database/
│       ├── __init__.py
│       ├── models.py              # SQLAlchemy ORM models
│       ├── connection.py          # DB connection manager
│       └── migrations/            # Alembic migrations
│
├── dashboard/                     # Streamlit Dashboard (MVP)
│   ├── app.py                     # Main dashboard
│   ├── pages/
│   │   ├── fleet_overview.py      # Fleet-wide health map
│   │   ├── transformer_detail.py  # Individual deep-dive
│   │   ├── dga_analysis.py        # DGA interpretation tool
│   │   ├── trend_monitor.py       # Trend charts & forecasts
│   │   └── alert_center.py        # Alert management
│   └── components/
│       ├── health_gauge.py        # 0-100 gauge widget
│       ├── duval_plot.py          # Interactive Duval triangle
│       └── gas_chart.py           # Gas concentration charts
│
├── tests/
│   ├── test_diagnosis/
│   │   ├── test_duval_triangle.py
│   │   ├── test_rogers_ratios.py
│   │   └── test_multi_method.py
│   ├── test_thermal/
│   │   ├── test_ieee_c57_91.py
│   │   └── test_aging_model.py
│   ├── test_health_index/
│   │   └── test_composite_hi.py
│   ├── test_prediction/
│   │   └── test_rul_estimator.py
│   └── conftest.py                # Shared test fixtures
│
├── notebooks/
│   ├── 01_dga_exploration.ipynb
│   ├── 02_thermal_model_validation.ipynb
│   ├── 03_health_index_calibration.ipynb
│   ├── 04_ml_model_training.ipynb
│   └── 05_demo_walkthrough.ipynb
│
└── docs/
    ├── getting_started.md
    ├── dga_methods.md
    ├── thermal_model.md
    ├── health_index.md
    ├── api_reference.md
    └── deployment.md
```

---

## 🔬 Module Deep Dives

### Module 1: DGA Fault Diagnosis Engine

This is the core domain logic. Various DGA interpretation techniques have been proposed in the literature, including the Doernenburg Ratio Method (DRM), Roger Ratio Method (RRM), IEC Ratio Method (IRM), Duval Triangle Method (DTM), and Duval Pentagon Method (DPM). While these techniques are well documented and widely used by industry, they may lead to different conclusions for the same oil sample.

**Your differentiator**: The proposed multi-method approach employs a scoring index and random forest machine learning principles to integrate existing interpretation methods into one comprehensive technique. The robustness of the proposed method is assessed using DGA data collected from several transformers under various health conditions. Results indicate that the proposed multi-method, based on the scoring index and random forest, offers greater accuracy and consistency than individual conventional interpretation methods alone.

**Implementation — `src/diagnosis/duval_triangle.py`:**

```python
"""
Duval Triangle Method for DGA Fault Classification.
Implements Duval Triangle 1 (for mineral oil transformers).

Reference: IEC 60599, Duval & DePabla (2001)
Fault types:
  PD  - Partial Discharge
  D1  - Discharge of Low Energy
  D2  - Discharge of High Energy
  T1  - Thermal Fault < 300°C
  T2  - Thermal Fault 300-700°C
  T3  - Thermal Fault > 700°C
  DT  - Mix of Thermal and Electrical
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FaultType(Enum):
    PD = "Partial Discharge"
    D1 = "Discharge of Low Energy"
    D2 = "Discharge of High Energy"
    T1 = "Thermal Fault < 300°C"
    T2 = "Thermal Fault 300-700°C"
    T3 = "Thermal Fault > 700°C"
    DT = "Mixed Thermal and Electrical"
    NORMAL = "Normal Degradation"
    UNDETERMINED = "Undetermined"


@dataclass
class DuvalResult:
    fault_type: FaultType
    confidence: float  # 0.0 to 1.0
    ch4_pct: float
    c2h4_pct: float
    c2h2_pct: float
    explanation: str


class DuvalTriangle1:
    """
    Duval Triangle 1 - Standard method for mineral oil transformers.
    Uses CH4, C2H4, C2H2 gas percentages.
    """

    def diagnose(
        self,
        ch4_ppm: float,
        c2h4_ppm: float,
        c2h2_ppm: float
    ) -> DuvalResult:
        """
        Classify fault type using Duval Triangle 1.

        Args:
            ch4_ppm: Methane concentration in ppm
            c2h4_ppm: Ethylene concentration in ppm
            c2h2_ppm: Acetylene concentration in ppm

        Returns:
            DuvalResult with fault classification
        """
        total = ch4_ppm + c2h4_ppm + c2h2_ppm

        if total == 0:
            return DuvalResult(
                fault_type=FaultType.NORMAL,
                confidence=0.5,
                ch4_pct=0, c2h4_pct=0, c2h2_pct=0,
                explanation="No combustible gases detected. Transformer appears healthy."
            )

        # Calculate percentages
        ch4_pct = (ch4_ppm / total) * 100
        c2h4_pct = (c2h4_ppm / total) * 100
        c2h2_pct = (c2h2_ppm / total) * 100

        # Classify zone based on Duval Triangle 1 boundaries
        fault_type = self._classify_zone(ch4_pct, c2h4_pct, c2h2_pct)

        confidence = self._calculate_confidence(
            ch4_pct, c2h4_pct, c2h2_pct, total
        )

        explanation = self._generate_explanation(
            fault_type, ch4_ppm, c2h4_ppm, c2h2_ppm
        )

        return DuvalResult(
            fault_type=fault_type,
            confidence=confidence,
            ch4_pct=ch4_pct,
            c2h4_pct=c2h4_pct,
            c2h2_pct=c2h2_pct,
            explanation=explanation
        )

    def _classify_zone(
        self, ch4: float, c2h4: float, c2h2: float
    ) -> FaultType:
        """Apply Duval Triangle 1 zone boundaries."""

        # PD zone: high CH4, very low C2H4 and C2H2
        if c2h2 < 2 and c2h4 < 20 and ch4 > 80:
            return FaultType.PD

        # D1 zone: significant C2H2
        if c2h2 > 13 and c2h4 < 23:
            return FaultType.D1

        # D2 zone: high C2H2 with moderate C2H4
        if c2h2 > 13 and c2h4 >= 23 and c2h4 < 40:
            return FaultType.D2

        # T1 zone: dominant CH4, low C2H4, low C2H2
        if c2h2 < 4 and c2h4 < 20 and ch4 > 50:
            return FaultType.T1

        # T2 zone: moderate C2H4, low C2H2
        if c2h2 < 4 and c2h4 >= 20 and c2h4 < 50:
            return FaultType.T2

        # T3 zone: high C2H4, low C2H2
        if c2h2 < 15 and c2h4 >= 50:
            return FaultType.T3

        # DT zone: everything else in the mixed region
        if c2h2 >= 4 and c2h2 <= 13:
            return FaultType.DT

        return FaultType.UNDETERMINED

    def _calculate_confidence(
        self, ch4: float, c2h4: float, c2h2: float, total_ppm: float
    ) -> float:
        """
        Confidence based on:
        1. Distance from zone boundary (deeper = more confident)
        2. Total gas volume (more gas = more signal)
        """
        # Base confidence from gas volume
        if total_ppm < 50:
            volume_conf = 0.3
        elif total_ppm < 200:
            volume_conf = 0.6
        elif total_ppm < 500:
            volume_conf = 0.8
        else:
            volume_conf = 0.95

        return min(volume_conf, 1.0)

    def _generate_explanation(
        self,
        fault_type: FaultType,
        ch4: float, c2h4: float, c2h2: float
    ) -> str:
        """Generate standards-based natural language explanation."""
        explanations = {
            FaultType.PD: (
                f"Partial discharge detected. Dominant gas: CH4 ({ch4:.0f} ppm). "
                f"Low energy electrical activity in gas spaces or voids in insulation."
            ),
            FaultType.D1: (
                f"Low-energy discharge detected. Elevated C2H2 ({c2h2:.0f} ppm). "
                f"Possible arcing or sparking at poor connections or between windings."
            ),
            FaultType.D2: (
                f"High-energy discharge detected. C2H2={c2h2:.0f}ppm, C2H4={c2h4:.0f}ppm. "
                f"Significant arcing — immediate investigation recommended."
            ),
            FaultType.T1: (
                f"Low-temperature thermal fault (<300°C). CH4 dominant ({ch4:.0f} ppm). "
                f"Possible hot spot in core or magnetic circuit."
            ),
            FaultType.T2: (
                f"Medium-temperature thermal fault (300-700°C). C2H4={c2h4:.0f}ppm. "
                f"Overheating in windings, leads, or connections."
            ),
            FaultType.T3: (
                f"High-temperature thermal fault (>700°C). High C2H4 ({c2h4:.0f} ppm). "
                f"Severe overheating — urgent action required."
            ),
            FaultType.DT: (
                f"Mixed thermal and electrical fault. Multiple gas indicators elevated. "
                f"Complex fault — recommend comprehensive inspection."
            ),
        }
        return explanations.get(fault_type, "Unable to determine fault type.")
```

**Implementation — `src/diagnosis/multi_method.py`:**

```python
"""
Multi-Method DGA Ensemble Diagnosis.

Combines Duval Triangle, Rogers Ratio, IEC Ratios, and Key Gas methods
into a single weighted diagnosis using a scoring approach.

Reference: IEEE C57.104-2019
"""

from dataclasses import dataclass
from typing import Dict, List
from .duval_triangle import DuvalTriangle1, FaultType
from .rogers_ratios import RogersRatios
from .iec_ratios import IECRatios
from .key_gas import KeyGasMethod
from .doernenburg import DoernenburgRatios


@dataclass
class GasConcentrations:
    """Standard DGA gas concentrations in ppm."""
    h2: float       # Hydrogen
    ch4: float      # Methane
    c2h6: float     # Ethane
    c2h4: float     # Ethylene
    c2h2: float     # Acetylene
    co: float       # Carbon Monoxide
    co2: float      # Carbon Dioxide


@dataclass
class MultiMethodResult:
    """Combined diagnosis from all methods."""
    primary_fault: FaultType
    consensus_score: float         # 0.0-1.0 agreement among methods
    individual_results: Dict[str, FaultType]
    overall_severity: str          # "Normal", "Caution", "Warning", "Critical"
    gas_status: Dict[str, str]     # Per-gas status vs IEEE thresholds
    explanation: str
    recommended_actions: List[str]


class MultiMethodDiagnosis:
    """
    Ensemble DGA diagnosis combining multiple standard methods.
    """

    # IEEE C57.104-2019 Condition levels (typical values for mineral oil)
    IEEE_THRESHOLDS = {
        "h2":   {"normal": 100, "caution": 200, "warning": 500, "critical": 700},
        "ch4":  {"normal": 75,  "caution": 125, "warning": 400, "critical": 600},
        "c2h6": {"normal": 65,  "caution": 100, "warning": 200, "critical": 400},
        "c2h4": {"normal": 50,  "caution": 100, "warning": 200, "critical": 500},
        "c2h2": {"normal": 2,   "caution": 10,  "warning": 35,  "critical": 50},
        "co":   {"normal": 350, "caution": 570, "warning": 1400,"critical": 1800},
        "co2":  {"normal": 3000,"caution": 5000,"warning": 10000,"critical":15000},
    }

    # Method weights (Duval most reliable per literature)
    METHOD_WEIGHTS = {
        "duval_triangle": 0.30,
        "rogers_ratios": 0.20,
        "iec_ratios": 0.20,
        "key_gas": 0.15,
        "doernenburg": 0.15,
    }

    def __init__(self):
        self.duval = DuvalTriangle1()
        self.rogers = RogersRatios()
        self.iec = IECRatios()
        self.key_gas = KeyGasMethod()
        self.doernenburg = DoernenburgRatios()

    def diagnose(self, gases: GasConcentrations) -> MultiMethodResult:
        """
        Run all DGA methods and combine results.
        """
        # Run each method
        results = {}
        results["duval_triangle"] = self.duval.diagnose(
            gases.ch4, gases.c2h4, gases.c2h2
        ).fault_type
        results["rogers_ratios"] = self.rogers.diagnose(
            gases.h2, gases.ch4, gases.c2h6, gases.c2h4, gases.c2h2
        ).fault_type
        results["iec_ratios"] = self.iec.diagnose(
            gases.h2, gases.ch4, gases.c2h6, gases.c2h4, gases.c2h2
        ).fault_type
        results["key_gas"] = self.key_gas.diagnose(gases).fault_type
        results["doernenburg"] = self.doernenburg.diagnose(gases).fault_type

        # Find consensus
        primary_fault = self._weighted_consensus(results)

        # Calculate agreement
        consensus_score = self._calculate_consensus(results, primary_fault)

        # Check each gas against IEEE thresholds
        gas_status = self._check_ieee_thresholds(gases)

        # Determine overall severity
        severity = self._determine_severity(gas_status, primary_fault)

        # Generate explanation and recommendations
        explanation = self._build_explanation(
            primary_fault, results, gas_status, gases
        )
        actions = self._recommend_actions(severity, primary_fault, gases)

        return MultiMethodResult(
            primary_fault=primary_fault,
            consensus_score=consensus_score,
            individual_results=results,
            overall_severity=severity,
            gas_status=gas_status,
            explanation=explanation,
            recommended_actions=actions,
        )

    def _check_ieee_thresholds(
        self, gases: GasConcentrations
    ) -> Dict[str, str]:
        """Check each gas against IEEE C57.104-2019 condition levels."""
        status = {}
        gas_values = {
            "h2": gases.h2, "ch4": gases.ch4, "c2h6": gases.c2h6,
            "c2h4": gases.c2h4, "c2h2": gases.c2h2,
            "co": gases.co, "co2": gases.co2,
        }
        for gas_name, value in gas_values.items():
            thresholds = self.IEEE_THRESHOLDS[gas_name]
            if value <= thresholds["normal"]:
                status[gas_name] = "Normal"
            elif value <= thresholds["caution"]:
                status[gas_name] = "Caution"
            elif value <= thresholds["warning"]:
                status[gas_name] = "Warning"
            else:
                status[gas_name] = "Critical"
        return status

    def _weighted_consensus(
        self, results: Dict[str, FaultType]
    ) -> FaultType:
        """Find the most-agreed-upon fault type via weighted vote."""
        fault_scores: Dict[FaultType, float] = {}
        for method, fault in results.items():
            weight = self.METHOD_WEIGHTS.get(method, 0.1)
            fault_scores[fault] = fault_scores.get(fault, 0) + weight
        return max(fault_scores, key=fault_scores.get)

    def _calculate_consensus(
        self, results: Dict[str, FaultType], primary: FaultType
    ) -> float:
        """Fraction of methods agreeing with primary diagnosis."""
        agree = sum(
            self.METHOD_WEIGHTS[m]
            for m, f in results.items()
            if f == primary
        )
        return round(agree / sum(self.METHOD_WEIGHTS.values()), 2)

    def _determine_severity(
        self, gas_status: Dict, fault_type: FaultType
    ) -> str:
        """Combine gas status and fault type into severity level."""
        severities = list(gas_status.values())
        if "Critical" in severities or fault_type in (
            FaultType.D2, FaultType.T3
        ):
            return "Critical"
        if "Warning" in severities or fault_type in (
            FaultType.D1, FaultType.T2, FaultType.DT
        ):
            return "Warning"
        if "Caution" in severities or fault_type in (
            FaultType.T1, FaultType.PD
        ):
            return "Caution"
        return "Normal"

    def _recommend_actions(
        self, severity: str, fault: FaultType, gases: GasConcentrations
    ) -> List[str]:
        """Generate prioritized actions based on severity."""
        actions = {
            "Normal": [
                "Continue routine monitoring per standard schedule.",
                "Next DGA sample recommended in 12 months.",
            ],
            "Caution": [
                "Increase DGA sampling frequency to every 3 months.",
                "Review loading history for overload events.",
                "Verify cooling system operational status.",
                "Compare with previous DGA results for trending.",
            ],
            "Warning": [
                "URGENT: Increase DGA sampling to monthly.",
                "Reduce transformer loading to 80% of nameplate if possible.",
                "Perform thermographic survey of external connections.",
                "Schedule inspection during next planned outage.",
                "Review protection relay settings.",
                "Notify asset management of elevated risk.",
            ],
            "Critical": [
                "IMMEDIATE: Consider emergency de-energization plan.",
                "Do NOT increase loading under any circumstances.",
                "Take DGA sample within 24 hours for confirmation.",
                "Perform online gas monitoring if available.",
                "Prepare contingency switching plan.",
                "Alert management and schedule emergency outage.",
                "Arrange internal inspection and oil processing.",
            ],
        }
        return actions.get(severity, actions["Normal"])
```

---

### Module 2: IEEE C57.91 Thermal Model

IEEE C57.169 provides guidance for determining the hottest-spot temperature in distribution and power transformers. It describes the important criteria to be evaluated by any thermal model that can accurately predict the hottest-spot temperature in a transformer.

IEEE C57.91 provides guidance on such methodologies. This guide is undergoing a major revision, including the primary thermal model, and adding the availability of open-source code.

**Implementation — `src/thermal/ieee_c57_91.py`:**

```python
"""
IEEE C57.91 Thermal Model for Oil-Immersed Transformers.

Calculates:
- Top-oil temperature rise
- Hot-spot temperature
- Insulation aging acceleration factor
- Loss of life (per hour, cumulative)
- Remaining useful life estimate

Reference: IEEE Std C57.91-2011 (and upcoming revision)
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TransformerThermalParams:
    """Transformer nameplate and thermal design parameters."""
    name: str
    rated_power_mva: float
    rated_voltage_hv_kv: float
    rated_voltage_lv_kv: float

    # Thermal parameters (from FAT test report)
    delta_theta_oil_rated: float = 55.0   # Top-oil rise at rated load (°C)
    delta_theta_hs_rated: float = 65.0    # Hot-spot rise at rated load (°C)
    hotspot_factor: float = 1.3           # H factor per IEEE C57.91

    # Loss parameters
    ratio_load_loss_no_load: float = 5.0  # R = load loss / no-load loss
    oil_time_constant_hrs: float = 3.0    # τ_oil (hours)
    winding_time_constant_min: float = 5.0  # τ_winding (minutes)

    # Cooling mode
    cooling_type: str = "ONAN"  # ONAN, ONAF, OFAF, ODAF

    # Exponents (depend on cooling mode)
    n: float = 0.8   # Oil exponent
    m: float = 0.8   # Winding exponent

    # Insulation type
    insulation_class: str = "thermally_upgraded"  # or "non_upgraded"


@dataclass
class ThermalState:
    """Current thermal state of transformer."""
    top_oil_temp_c: float
    hotspot_temp_c: float
    ambient_temp_c: float
    load_pu: float                  # Per-unit loading (1.0 = rated)
    aging_acceleration_factor: float
    loss_of_life_pct_per_hour: float
    estimated_rul_years: float
    insulation_condition: str       # "Good", "Moderate", "Degraded", "Critical"


class IEEEC5791ThermalModel:
    """
    IEEE C57.91 thermal model implementation.

    Calculates temperature and aging for oil-immersed transformers.
    """

    # Reference temperatures
    THETA_REF_UPGRADED = 110.0     # °C reference for thermally upgraded paper
    THETA_REF_NON_UPGRADED = 95.0  # °C reference for non-upgraded paper

    # Aging rate constants (Arrhenius equation parameters)
    AGING_CONSTANT_UPGRADED = 15000.0
    AGING_CONSTANT_NON_UPGRADED = 15000.0

    # Standard life expectancy
    STANDARD_LIFE_HOURS = 180_000  # 20.5 years per IEEE C57.12.00

    # Cooling mode exponents
    COOLING_EXPONENTS = {
        "ONAN":  {"n": 0.8, "m": 0.8},
        "ONAF":  {"n": 0.8, "m": 0.8},
        "OFAF":  {"n": 1.0, "m": 1.0},
        "ODAF":  {"n": 1.0, "m": 1.0},
    }

    def __init__(self, params: TransformerThermalParams):
        self.params = params
        exponents = self.COOLING_EXPONENTS.get(
            params.cooling_type, {"n": 0.8, "m": 0.8}
        )
        self.n = exponents["n"]
        self.m = exponents["m"]

        if params.insulation_class == "thermally_upgraded":
            self.theta_ref = self.THETA_REF_UPGRADED
        else:
            self.theta_ref = self.THETA_REF_NON_UPGRADED

    def calculate_steady_state(
        self,
        load_pu: float,
        ambient_temp_c: float,
    ) -> ThermalState:
        """
        Calculate steady-state temperatures and aging.

        Args:
            load_pu: Per-unit loading (1.0 = rated, 1.5 = 150%)
            ambient_temp_c: Ambient temperature in °C
        """
        R = self.params.ratio_load_loss_no_load

        # Top-oil temperature rise over ambient (IEEE C57.91 Eq. 7.1)
        delta_theta_oil = self.params.delta_theta_oil_rated * (
            (load_pu**2 * R + 1) / (R + 1)
        ) ** self.n

        # Top-oil temperature
        top_oil_temp = ambient_temp_c + delta_theta_oil

        # Hot-spot rise over top-oil (IEEE C57.91 Eq. 7.2)
        delta_theta_hs = (
            self.params.delta_theta_hs_rated
            - self.params.delta_theta_oil_rated
        ) * load_pu ** (2 * self.m)

        # Hot-spot temperature
        hotspot_temp = top_oil_temp + delta_theta_hs

        # Aging acceleration factor (IEEE C57.91 Eq. 6.1)
        faa = self._aging_acceleration_factor(hotspot_temp)

        # Loss of life per hour
        lol_pct_per_hour = (faa / self.STANDARD_LIFE_HOURS) * 100

        # Estimated RUL
        rul_years = self._estimate_rul(faa)

        # Condition assessment
        condition = self._assess_condition(hotspot_temp, faa)

        return ThermalState(
            top_oil_temp_c=round(top_oil_temp, 1),
            hotspot_temp_c=round(hotspot_temp, 1),
            ambient_temp_c=ambient_temp_c,
            load_pu=load_pu,
            aging_acceleration_factor=round(faa, 4),
            loss_of_life_pct_per_hour=round(lol_pct_per_hour, 6),
            estimated_rul_years=round(rul_years, 1),
            insulation_condition=condition,
        )

    def _aging_acceleration_factor(self, hotspot_temp_c: float) -> float:
        """
        Aging Acceleration Factor per IEEE C57.91.
        FAA = exp[(A) * (1/θ_ref - 1/θ_hs)]
        where temperatures are in Kelvin.
        """
        theta_hs_k = hotspot_temp_c + 273.15
        theta_ref_k = self.theta_ref + 273.15

        faa = math.exp(
            self.AGING_CONSTANT_UPGRADED * (
                1.0 / theta_ref_k - 1.0 / theta_hs_k
            )
        )
        return faa

    def _estimate_rul(self, current_faa: float) -> float:
        """
        Estimate remaining useful life assuming current conditions persist.
        """
        if current_faa <= 0:
            return 999.0
        remaining_hours = self.STANDARD_LIFE_HOURS / current_faa
        return remaining_hours / 8760  # Convert to years

    def _assess_condition(self, hotspot: float, faa: float) -> str:
        """Assess insulation condition based on hotspot and aging rate."""
        if hotspot > 140 or faa > 32:
            return "Critical"
        elif hotspot > 120 or faa > 4:
            return "Degraded"
        elif hotspot > 110 or faa > 1:
            return "Moderate"
        else:
            return "Good"

    def calculate_loading_profile(
        self,
        hourly_loads_pu: List[float],
        hourly_ambient_temps_c: List[float],
    ) -> dict:
        """
        Calculate 24-hour aging from load and temperature profiles.

        Returns cumulative loss-of-life for the period.
        """
        assert len(hourly_loads_pu) == len(hourly_ambient_temps_c)

        total_faa = 0.0
        max_hotspot = 0.0
        states = []

        for load, ambient in zip(hourly_loads_pu, hourly_ambient_temps_c):
            state = self.calculate_steady_state(load, ambient)
            states.append(state)
            total_faa += state.aging_acceleration_factor
            max_hotspot = max(max_hotspot, state.hotspot_temp_c)

        hours = len(hourly_loads_pu)
        avg_faa = total_faa / hours
        equivalent_aging_hours = total_faa  # 1 hour each
        loss_of_life_pct = (equivalent_aging_hours / self.STANDARD_LIFE_HOURS) * 100

        return {
            "period_hours": hours,
            "average_faa": round(avg_faa, 4),
            "max_hotspot_c": round(max_hotspot, 1),
            "equivalent_aging_hours": round(equivalent_aging_hours, 2),
            "loss_of_life_pct": round(loss_of_life_pct, 6),
            "peak_load_pu": max(hourly_loads_pu),
            "hourly_states": states,
        }

    def max_safe_loading(
        self,
        ambient_temp_c: float,
        max_hotspot_c: float = 120.0,
        max_faa: float = 4.0,
    ) -> float:
        """
        Calculate maximum safe loading for given ambient temperature.
        Binary search for the load that just reaches the limit.
        """
        low, high = 0.0, 2.5  # Search range in per-unit

        for _ in range(50):  # Binary search iterations
            mid = (low + high) / 2
            state = self.calculate_steady_state(mid, ambient_temp_c)

            if (state.hotspot_temp_c > max_hotspot_c or
                    state.aging_acceleration_factor > max_faa):
                high = mid
            else:
                low = mid

        return round(low, 3)
```

---

### Module 3: Composite Health Index

Measuring power transformers HI provides a quantitative evaluation of the overall transformer's health condition, which can be employed as a viable tool for asset management based on a single index. Through incorporating DGA into HI calculations, the complex chemical byproduct substances due to insulation degradation, overheating, and electrical discharges are translated into a standardized score. The relative concentration and generation rates of these gases provide diagnostic information that reflects the severity and type of fault occurring for a single unit or fleet of transformers. This can provide a robust way to evaluate the asset's condition, thus supporting maintenance activities and leading to breakdown prevention and system losses.

**Implementation — `src/health_index/composite_hi.py`:**

```python
"""
Composite Health Index (0-100) for Power Transformers.

Combines multiple condition indicators into a single score.

Components (configurable weights):
- DGA Factor (DGAF):     40%  — Dissolved gas analysis
- Oil Quality Factor:    20%  — Acidity, moisture, BDV, color
- Electrical Tests:      15%  — Power factor, IR, TTR
- Thermal Performance:   15%  — Loading history, thermal aging
- Age & Maintenance:     10%  — Age, maintenance history

Score interpretation:
  85-100: Good        — Normal aging, routine monitoring
  70-84:  Acceptable  — Minor concerns, increase monitoring
  50-69:  Caution     — Significant degradation, plan intervention
  30-49:  Poor        — Major concerns, prioritize maintenance
   0-29:  Critical    — Immediate action required
"""

from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class HealthIndexInput:
    """All inputs needed for composite health index calculation."""
    # DGA data (ppm)
    h2: float = 0
    ch4: float = 0
    c2h6: float = 0
    c2h4: float = 0
    c2h2: float = 0
    co: float = 0
    co2: float = 0

    # Oil quality
    acidity_mg_koh_g: Optional[float] = None
    moisture_ppm: Optional[float] = None
    bdv_kv: Optional[float] = None   # Breakdown voltage
    ift_mn_m: Optional[float] = None  # Interfacial tension
    color: Optional[float] = None

    # Electrical tests
    power_factor_pct: Optional[float] = None
    insulation_resistance_mohm: Optional[float] = None
    turns_ratio_deviation_pct: Optional[float] = None

    # Thermal / Loading
    avg_loading_pct: float = 70
    peak_loading_pct: float = 100
    cumulative_lol_pct: float = 0  # Cumulative loss of life
    max_hotspot_c: float = 90

    # Age and maintenance
    age_years: float = 10
    years_since_last_maintenance: float = 1
    total_maintenance_events: int = 5


@dataclass
class HealthIndexResult:
    """Health Index calculation result."""
    composite_score: float          # 0-100
    condition: str                  # Good/Acceptable/Caution/Poor/Critical
    component_scores: Dict[str, float]
    risk_rank: int                  # 1=highest risk in fleet
    explanation: str
    trending: str                   # "Improving", "Stable", "Declining"


class CompositeHealthIndex:
    """
    Calculates weighted composite health index.
    """

    DEFAULT_WEIGHTS = {
        "dga": 0.40,
        "oil_quality": 0.20,
        "electrical": 0.15,
        "thermal": 0.15,
        "age_maintenance": 0.10,
    }

    def calculate(
        self,
        inputs: HealthIndexInput,
        weights: Optional[Dict[str, float]] = None
    ) -> HealthIndexResult:
        """
        Calculate composite health index from all available inputs.
        """
        w = weights or self.DEFAULT_WEIGHTS

        # Calculate each component score (0-100)
        scores = {}
        scores["dga"] = self._dga_score(inputs)
        scores["oil_quality"] = self._oil_quality_score(inputs)
        scores["electrical"] = self._electrical_score(inputs)
        scores["thermal"] = self._thermal_score(inputs)
        scores["age_maintenance"] = self._age_score(inputs)

        # Weighted composite
        composite = sum(scores[k] * w[k] for k in scores)
        composite = round(max(0, min(100, composite)), 1)

        condition = self._classify_condition(composite)
        explanation = self._generate_explanation(composite, scores, inputs)

        return HealthIndexResult(
            composite_score=composite,
            condition=condition,
            component_scores=scores,
            risk_rank=0,  # Set by fleet manager
            explanation=explanation,
            trending="Stable",  # Set by trend analyzer
        )

    def _dga_score(self, inp: HealthIndexInput) -> float:
        """
        Score DGA results (0=worst, 100=best).
        Based on IEEE C57.104-2019 condition levels.
        """
        # Individual gas scores
        gas_scores = {
            "h2": self._gas_level_score(inp.h2, 100, 200, 500, 700),
            "ch4": self._gas_level_score(inp.ch4, 75, 125, 400, 600),
            "c2h6": self._gas_level_score(inp.c2h6, 65, 100, 200, 400),
            "c2h4": self._gas_level_score(inp.c2h4, 50, 100, 200, 500),
            "c2h2": self._gas_level_score(inp.c2h2, 2, 10, 35, 50),
            "co": self._gas_level_score(inp.co, 350, 570, 1400, 1800),
            "co2": self._gas_level_score(inp.co2, 3000, 5000, 10000, 15000),
        }

        # Acetylene has highest weight (most dangerous)
        weights = {
            "h2": 0.12, "ch4": 0.12, "c2h6": 0.08, "c2h4": 0.18,
            "c2h2": 0.25, "co": 0.13, "co2": 0.12,
        }

        score = sum(gas_scores[g] * weights[g] for g in gas_scores)
        return round(score, 1)

    def _gas_level_score(
        self, value: float,
        normal: float, caution: float,
        warning: float, critical: float
    ) -> float:
        """Convert gas concentration to 0-100 score."""
        if value <= normal:
            return 100
        elif value <= caution:
            return 100 - 25 * (value - normal) / (caution - normal)
        elif value <= warning:
            return 75 - 35 * (value - caution) / (warning - caution)
        elif value <= critical:
            return 40 - 30 * (value - warning) / (critical - warning)
        else:
            return max(0, 10 - (value - critical) / critical * 10)

    def _oil_quality_score(self, inp: HealthIndexInput) -> float:
        """Score oil quality parameters."""
        scores = []

        if inp.acidity_mg_koh_g is not None:
            if inp.acidity_mg_koh_g < 0.1:
                scores.append(100)
            elif inp.acidity_mg_koh_g < 0.2:
                scores.append(75)
            elif inp.acidity_mg_koh_g < 0.4:
                scores.append(40)
            else:
                scores.append(10)

        if inp.moisture_ppm is not None:
            if inp.moisture_ppm < 15:
                scores.append(100)
            elif inp.moisture_ppm < 25:
                scores.append(70)
            elif inp.moisture_ppm < 35:
                scores.append(40)
            else:
                scores.append(10)

        if inp.bdv_kv is not None:
            if inp.bdv_kv > 60:
                scores.append(100)
            elif inp.bdv_kv > 40:
                scores.append(70)
            elif inp.bdv_kv > 25:
                scores.append(35)
            else:
                scores.append(5)

        return round(sum(scores) / len(scores), 1) if scores else 70.0

    def _electrical_score(self, inp: HealthIndexInput) -> float:
        """Score electrical test results."""
        scores = []

        if inp.power_factor_pct is not None:
            if inp.power_factor_pct < 0.5:
                scores.append(100)
            elif inp.power_factor_pct < 1.0:
                scores.append(70)
            elif inp.power_factor_pct < 2.0:
                scores.append(40)
            else:
                scores.append(10)

        if inp.insulation_resistance_mohm is not None:
            if inp.insulation_resistance_mohm > 5000:
                scores.append(100)
            elif inp.insulation_resistance_mohm > 1000:
                scores.append(70)
            elif inp.insulation_resistance_mohm > 500:
                scores.append(40)
            else:
                scores.append(10)

        return round(sum(scores) / len(scores), 1) if scores else 70.0

    def _thermal_score(self, inp: HealthIndexInput) -> float:
        """Score thermal and loading history."""
        score = 100.0

        # Loading penalty
        if inp.peak_loading_pct > 120:
            score -= 30
        elif inp.peak_loading_pct > 100:
            score -= 15

        if inp.avg_loading_pct > 90:
            score -= 20
        elif inp.avg_loading_pct > 80:
            score -= 10

        # Cumulative aging penalty
        if inp.cumulative_lol_pct > 50:
            score -= 40
        elif inp.cumulative_lol_pct > 25:
            score -= 20
        elif inp.cumulative_lol_pct > 10:
            score -= 10

        # Hotspot penalty
        if inp.max_hotspot_c > 120:
            score -= 25
        elif inp.max_hotspot_c > 110:
            score -= 15

        return round(max(0, score), 1)

    def _age_score(self, inp: HealthIndexInput) -> float:
        """Score based on age and maintenance history."""
        # Age penalty (linear degradation)
        if inp.age_years < 10:
            age_score = 100
        elif inp.age_years < 20:
            age_score = 100 - (inp.age_years - 10) * 3
        elif inp.age_years < 30:
            age_score = 70 - (inp.age_years - 20) * 4
        elif inp.age_years < 40:
            age_score = 30 - (inp.age_years - 30) * 2
        else:
            age_score = 10

        # Maintenance bonus/penalty
        if inp.years_since_last_maintenance > 5:
            age_score -= 15
        elif inp.years_since_last_maintenance > 3:
            age_score -= 5

        return round(max(0, age_score), 1)

    def _classify_condition(self, score: float) -> str:
        if score >= 85: return "Good"
        if score >= 70: return "Acceptable"
        if score >= 50: return "Caution"
        if score >= 30: return "Poor"
        return "Critical"

    def _generate_explanation(
        self, composite: float, scores: Dict, inp: HealthIndexInput
    ) -> str:
        """Build a plain-English summary of the health index."""
        worst = min(scores, key=scores.get)
        worst_label = {
            "dga": "dissolved gas analysis",
            "oil_quality": "oil quality",
            "electrical": "electrical tests",
            "thermal": "thermal performance",
            "age_maintenance": "age and maintenance history",
        }

        explanation = (
            f"Health Index: {composite}/100 "
            f"({self._classify_condition(composite)}). "
            f"Primary concern: {worst_label.get(worst, worst)} "
            f"(score: {scores[worst]}/100). "
        )

        if inp.c2h2 > 10:
            explanation += "ALERT: Elevated acetylene indicates possible arcing. "
        if inp.age_years > 25:
            explanation += f"Transformer age ({inp.age_years}y) exceeds typical design life. "

        return explanation
```

---

### Module 4: Prediction Engine

```python
"""
src/prediction/rul_estimator.py

Remaining Useful Life (RUL) Estimator
Combines physics-based (IEEE C57.91 aging) with data-driven (ML) approaches.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RULEstimate:
    """Remaining Useful Life prediction result."""
    rul_years: float
    rul_days: int
    confidence_lower_years: float  # 10th percentile
    confidence_upper_years: float  # 90th percentile
    method: str                    # "physics", "ml", "hybrid"
    failure_probability_30d: float # P(failure in 30 days)
    failure_probability_90d: float # P(failure in 90 days)
    failure_probability_1y: float  # P(failure in 1 year)
    risk_level: str               # Low, Medium, High, Critical
    key_risk_factors: List[str]


class RULEstimator:
    """
    Hybrid RUL estimation combining:
    1. IEEE C57.91 physics-based thermal aging
    2. DGA trend extrapolation
    3. Statistical failure probability
    """

    STANDARD_LIFE_HOURS = 180_000

    def estimate(
        self,
        current_health_index: float,
        cumulative_loss_of_life_pct: float,
        current_aging_factor: float,
        dga_trend_slope: Optional[dict] = None,  # PPM per month change
        age_years: float = 10,
        health_history: Optional[List[Tuple[float, float]]] = None,
    ) -> RULEstimate:
        """
        Estimate remaining useful life using multiple methods.
        """
        # Method 1: Physics-based (IEEE C57.91 aging)
        physics_rul = self._physics_rul(
            cumulative_loss_of_life_pct, current_aging_factor
        )

        # Method 2: DGA trend extrapolation
        dga_rul = self._dga_trend_rul(dga_trend_slope) if dga_trend_slope else None

        # Method 3: Health index trend
        hi_rul = self._hi_trend_rul(health_history) if health_history else None

        # Combine estimates (take minimum as conservative)
        rul_estimates = [r for r in [physics_rul, dga_rul, hi_rul] if r is not None]
        primary_rul = min(rul_estimates) if rul_estimates else physics_rul

        # Calculate failure probabilities
        fp_30d = self._failure_probability(primary_rul, 30)
        fp_90d = self._failure_probability(primary_rul, 90)
        fp_1y = self._failure_probability(primary_rul, 365)

        # Confidence interval (±30% for physics, wider for data-driven)
        conf_lower = max(0, primary_rul * 0.5)
        conf_upper = primary_rul * 1.5

        # Risk factors
        risk_factors = self._identify_risk_factors(
            current_health_index, current_aging_factor,
            dga_trend_slope, age_years
        )

        risk_level = self._assess_risk(primary_rul, fp_90d)

        return RULEstimate(
            rul_years=round(primary_rul, 1),
            rul_days=int(primary_rul * 365),
            confidence_lower_years=round(conf_lower, 1),
            confidence_upper_years=round(conf_upper, 1),
            method="hybrid" if len(rul_estimates) > 1 else "physics",
            failure_probability_30d=round(fp_30d, 4),
            failure_probability_90d=round(fp_90d, 4),
            failure_probability_1y=round(fp_1y, 4),
            risk_level=risk_level,
            key_risk_factors=risk_factors,
        )

    def _physics_rul(
        self, cumulative_lol_pct: float, current_faa: float
    ) -> float:
        """RUL based on IEEE C57.91 insulation aging."""
        remaining_life_pct = max(0, 100 - cumulative_lol_pct)
        remaining_hours = (remaining_life_pct / 100) * self.STANDARD_LIFE_HOURS

        if current_faa > 0:
            # Adjust for current aging rate
            adjusted_hours = remaining_hours / current_faa
        else:
            adjusted_hours = remaining_hours

        return adjusted_hours / 8760  # Hours to years

    def _dga_trend_rul(self, trend_slope: dict) -> Optional[float]:
        """
        RUL based on DGA gas trending.
        Extrapolate when gases will reach critical thresholds.
        """
        critical_thresholds = {
            "h2": 700, "ch4": 600, "c2h4": 500, "c2h2": 50,
        }

        min_time_to_critical = float("inf")

        for gas, slope in trend_slope.items():
            if gas in critical_thresholds and slope > 0:
                current_val = trend_slope.get(f"{gas}_current", 0)
                remaining = critical_thresholds[gas] - current_val
                if remaining > 0:
                    months_to_critical = remaining / slope
                    years = months_to_critical / 12
                    min_time_to_critical = min(min_time_to_critical, years)

        if min_time_to_critical == float("inf"):
            return None
        return max(0, min_time_to_critical)

    def _hi_trend_rul(
        self, history: List[Tuple[float, float]]
    ) -> Optional[float]:
        """
        RUL based on health index decline trend.
        Uses linear extrapolation to HI = 30 (Poor threshold).
        """
        if len(history) < 3:
            return None

        times = [h[0] for h in history]  # Years
        scores = [h[1] for h in history]

        # Simple linear regression
        n = len(times)
        sum_x = sum(times)
        sum_y = sum(scores)
        sum_xy = sum(t * s for t, s in zip(times, scores))
        sum_x2 = sum(t ** 2 for t in times)

        denom = n * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-10:
            return None

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n

        if slope >= 0:  # HI not declining
            return None

        # When does HI reach 30?
        critical_hi = 30
        time_to_critical = (critical_hi - intercept) / slope
        current_time = times[-1]

        return max(0, time_to_critical - current_time)

    def _failure_probability(
        self, rul_years: float, horizon_days: int
    ) -> float:
        """
        Estimate probability of failure within horizon.
        Uses exponential CDF as approximation.
        """
        if rul_years <= 0:
            return 1.0

        lambda_rate = 1.0 / (rul_years * 365)  # Failure rate per day
        prob = 1 - np.exp(-lambda_rate * horizon_days)
        return min(1.0, max(0.0, prob))

    def _identify_risk_factors(
        self, hi: float, faa: float,
        dga_trend: Optional[dict], age: float
    ) -> List[str]:
        """Identify top risk factors for this transformer."""
        factors = []

        if hi < 50:
            factors.append(f"Low health index ({hi}/100)")
        if faa > 4:
            factors.append(f"Accelerated aging (FAA={faa:.1f}x normal)")
        if age > 25:
            factors.append(f"Advanced age ({age} years)")

        if dga_trend:
            for gas, slope in dga_trend.items():
                if not gas.endswith("_current") and slope > 5:
                    factors.append(f"Rapidly increasing {gas.upper()} (+{slope:.0f} ppm/month)")

        return factors[:5]  # Top 5

    def _assess_risk(self, rul_years: float, fp_90d: float) -> str:
        if rul_years < 0.5 or fp_90d > 0.3:
            return "Critical"
        if rul_years < 2 or fp_90d > 0.1:
            return "High"
        if rul_years < 5 or fp_90d > 0.05:
            return "Medium"
        return "Low"
```

---

## 📊 Data Strategy

### What You Can Work With Today

| Dataset                                                | Source                       | What It Gives You                    |
| ------------------------------------------------------ | ---------------------------- | ------------------------------------ |
| **IEC TC 10 DGA Database**                             | Published in IEEE/IEC papers | ~3000+ DGA samples with known faults |
| **Kaggle Electricity Transformer Dataset (ETDataset)** | Kaggle                       | Load and temperature time series     |
| **Open-Meteo API**                                     | open-meteo.com (free)        | Hourly weather data for any location |
| **Synthetic DGA data**                                 | Generate yourself            | Controlled scenarios for testing     |
| **IEEE C57.104-2019 tables**                           | Standard document            | Threshold values for gas levels      |

Very diversified and massive DGA datasets of in-service transformers were collected from various utilities for study. One dataset consists of 3147 instances with four classes: No fault, Thermal fault, low energy discharge, and high energy discharge.

### Synthetic Data Generator (critical for MVP)

```python
"""
src/ingestion/synthetic_generator.py

Generates realistic synthetic transformer data for development and demo.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


class SyntheticTransformerGenerator:
    """
    Generate realistic synthetic DGA and sensor data.
    Based on IEEE C57.104 typical gas concentrations.
    """

    # Gas profiles for different conditions
    GAS_PROFILES = {
        "healthy": {
            "h2": (30, 15), "ch4": (20, 10), "c2h6": (15, 8),
            "c2h4": (10, 5), "c2h2": (0.5, 0.3),
            "co": (200, 80), "co2": (2000, 500),
        },
        "aging": {
            "h2": (80, 25), "ch4": (50, 20), "c2h6": (40, 15),
            "c2h4": (30, 12), "c2h2": (1, 0.5),
            "co": (400, 100), "co2": (4000, 800),
        },
        "thermal_fault_developing": {
            "h2": (150, 40), "ch4": (100, 30), "c2h6": (80, 25),
            "c2h4": (120, 35), "c2h2": (3, 1.5),
            "co": (600, 150), "co2": (6000, 1000),
        },
        "arcing": {
            "h2": (500, 100), "ch4": (80, 30), "c2h6": (30, 15),
            "c2h4": (100, 40), "c2h2": (40, 15),
            "co": (500, 120), "co2": (5000, 800),
        },
        "partial_discharge": {
            "h2": (300, 80), "ch4": (50, 20), "c2h6": (10, 5),
            "c2h4": (15, 8), "c2h2": (5, 3),
            "co": (300, 100), "co2": (3000, 600),
        },
    }

    def generate_fleet(
        self,
        n_transformers: int = 50,
        seed: int = 42
    ) -> pd.DataFrame:
        """Generate a fleet of transformers with varied conditions."""
        np.random.seed(seed)

        transformers = []
        conditions = list(self.GAS_PROFILES.keys())
        condition_weights = [0.40, 0.30, 0.15, 0.08, 0.07]

        for i in range(n_transformers):
            condition = np.random.choice(conditions, p=condition_weights)
            age = np.random.uniform(2, 45)
            rating = np.random.choice([5, 10, 20, 40, 63, 100])

            transformers.append({
                "id": f"TX-{i+1:04d}",
                "name": f"Transformer {i+1}",
                "rated_mva": rating,
                "voltage_kv": np.random.choice([33, 66, 110, 132, 220]),
                "age_years": round(age, 1),
                "condition_category": condition,
                "location": f"Substation-{np.random.randint(1, 20):02d}",
                "cooling_type": np.random.choice(
                    ["ONAN", "ONAF", "OFAF"], p=[0.5, 0.35, 0.15]
                ),
            })

        return pd.DataFrame(transformers)

    def generate_dga_history(
        self,
        transformer_id: str,
        condition: str,
        n_samples: int = 20,
        years_span: float = 10,
        degradation_rate: float = 0.05,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate time-series DGA history for one transformer.

        Includes realistic gas trending (gradual increase for
        deteriorating transformers).
        """
        if seed:
            np.random.seed(seed)

        profile = self.GAS_PROFILES[condition]
        records = []

        start_date = datetime.now() - timedelta(days=years_span * 365)
        interval_days = (years_span * 365) / n_samples

        for i in range(n_samples):
            sample_date = start_date + timedelta(days=i * interval_days)

            # Time-based degradation factor
            time_factor = 1 + degradation_rate * (i / n_samples)

            record = {
                "transformer_id": transformer_id,
                "sample_date": sample_date.strftime("%Y-%m-%d"),
                "sample_number": i + 1,
            }

            for gas, (mean, std) in profile.items():
                value = np.random.normal(mean * time_factor, std)
                record[gas] = max(0, round(value, 1))

            # Add metadata
            record["oil_temperature_c"] = round(
                np.random.normal(65, 8), 1
            )
            record["ambient_temperature_c"] = round(
                np.random.normal(25, 8), 1
            )
            record["load_pct"] = round(
                np.random.normal(70, 15), 1
            )
            record["lab"] = np.random.choice(
                ["LabA", "LabB", "LabC"]
            )

            records.append(record)

        return pd.DataFrame(records)
```

---

## 🖥 Dashboard — Streamlit MVP

Use **Streamlit** (not React) for the MVP — it ships 10x faster:

```python
"""
dashboard/app.py

TransformerGuard Dashboard — Fleet Health Overview
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
sys.path.append("..")

from src.health_index.composite_hi import CompositeHealthIndex, HealthIndexInput
from src.diagnosis.multi_method import MultiMethodDiagnosis, GasConcentrations
from src.thermal.ieee_c57_91 import IEEEC5791ThermalModel, TransformerThermalParams

st.set_page_config(
    page_title="TransformerGuard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ TransformerGuard — Transformer Fleet Health Intelligence")


# --- SIDEBAR: Upload or use sample data ---
st.sidebar.header("📁 Data Source")
data_source = st.sidebar.radio(
    "Choose data source:",
    ["Sample Fleet (Demo)", "Upload CSV"]
)

if data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader(
        "Upload DGA data (CSV)", type=["csv"]
    )
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        st.info("Please upload a CSV file with DGA data.")
        st.stop()
else:
    # Load sample data
    from src.ingestion.synthetic_generator import SyntheticTransformerGenerator
    gen = SyntheticTransformerGenerator()
    fleet_df = gen.generate_fleet(50)
    df = fleet_df


# --- FLEET OVERVIEW ---
st.header("🏭 Fleet Overview")

col1, col2, col3, col4 = st.columns(4)

hi_engine = CompositeHealthIndex()

# Calculate health index for each transformer
health_scores = []
for _, row in df.iterrows():
    gen = SyntheticTransformerGenerator()
    dga = gen.generate_dga_history(
        row["id"], row["condition_category"],
        n_samples=1, seed=hash(row["id"]) % 10000
    ).iloc[0]

    inp = HealthIndexInput(
        h2=dga["h2"], ch4=dga["ch4"], c2h6=dga["c2h6"],
        c2h4=dga["c2h4"], c2h2=dga["c2h2"],
        co=dga["co"], co2=dga["co2"],
        age_years=row["age_years"],
        avg_loading_pct=dga.get("load_pct", 70),
    )
    result = hi_engine.calculate(inp)
    health_scores.append({
        "id": row["id"],
        "name": row["name"],
        "health_score": result.composite_score,
        "condition": result.condition,
        "rated_mva": row["rated_mva"],
        "age": row["age_years"],
        "location": row["location"],
    })

health_df = pd.DataFrame(health_scores)

# Summary metrics
critical = len(health_df[health_df["condition"] == "Critical"])
poor = len(health_df[health_df["condition"] == "Poor"])
caution = len(health_df[health_df["condition"] == "Caution"])
good = len(health_df[health_df["condition"].isin(["Good", "Acceptable"])])

col1.metric("🔴 Critical", critical)
col2.metric("🟠 Poor", poor)
col3.metric("🟡 Caution", caution)
col4.metric("🟢 Healthy", good)


# --- FLEET HEATMAP ---
st.subheader("Fleet Health Heatmap")
fig = px.treemap(
    health_df,
    path=["location", "name"],
    values="rated_mva",
    color="health_score",
    color_continuous_scale=["red", "orange", "yellow", "green"],
    range_color=[0, 100],
    title="Transformer Fleet — Size = Rating (MVA), Color = Health Score"
)
st.plotly_chart(fig, use_container_width=True)


# --- RISK TABLE ---
st.subheader("🚨 Priority Alert Table")
priority_df = health_df.sort_values("health_score").head(10)
st.dataframe(
    priority_df.style.background_gradient(
        subset=["health_score"],
        cmap="RdYlGn",
        vmin=0, vmax=100
    ),
    use_container_width=True
)


# --- DGA ANALYSIS TOOL ---
st.header("🔬 DGA Analysis Tool")
st.write("Enter DGA gas concentrations for instant diagnosis:")

col_a, col_b = st.columns(2)

with col_a:
    h2 = st.number_input("H2 (ppm)", value=50.0, min_value=0.0)
    ch4 = st.number_input("CH4 (ppm)", value=30.0, min_value=0.0)
    c2h6 = st.number_input("C2H6 (ppm)", value=20.0, min_value=0.0)
    c2h4 = st.number_input("C2H4 (ppm)", value=15.0, min_value=0.0)

with col_b:
    c2h2 = st.number_input("C2H2 (ppm)", value=1.0, min_value=0.0)
    co = st.number_input("CO (ppm)", value=300.0, min_value=0.0)
    co2 = st.number_input("CO2 (ppm)", value=2500.0, min_value=0.0)

if st.button("🔍 Analyze DGA", type="primary"):
    gases = GasConcentrations(
        h2=h2, ch4=ch4, c2h6=c2h6,
        c2h4=c2h4, c2h2=c2h2, co=co, co2=co2
    )
    diagnosis = MultiMethodDiagnosis()
    result = diagnosis.diagnose(gases)

    # Display results
    severity_color = {
        "Normal": "green", "Caution": "orange",
        "Warning": "red", "Critical": "darkred"
    }

    st.subheader(f"Diagnosis: {result.primary_fault.value}")
    st.markdown(
        f"**Severity:** :{severity_color[result.overall_severity]}"
        f"[{result.overall_severity}]"
    )
    st.write(f"**Consensus:** {result.consensus_score*100:.0f}% method agreement")
    st.write(f"**Explanation:** {result.explanation}")

    st.subheader("Recommended Actions")
    for i, action in enumerate(result.recommended_actions, 1):
        st.write(f"{i}. {action}")

    # Individual method results
    with st.expander("View Individual Method Results"):
        for method, fault in result.individual_results.items():
            st.write(f"**{method}:** {fault.value}")

    # Gas status table
    with st.expander("IEEE C57.104 Gas Status"):
        gas_df = pd.DataFrame([
            {"Gas": k.upper(), "Status": v}
            for k, v in result.gas_status.items()
        ])
        st.dataframe(gas_df, use_container_width=True)
```

---

## 🗓 Build Timeline — 16 Weeks to Demo-Ready MVP

| Week      | Sprint       | Deliverable                                                     | Code                        |
| --------- | ------------ | --------------------------------------------------------------- | --------------------------- |
| **1-2**   | Setup        | Project structure, CI/CD, DB schema, synthetic data generator   | `ingestion/`, `config/`     |
| **3-4**   | DGA Core     | Duval Triangle, Rogers Ratios, IEC Ratios, Key Gas, Doernenburg | `diagnosis/`                |
| **5-6**   | Multi-Method | Ensemble diagnosis, IEEE threshold checking, scoring            | `diagnosis/multi_method.py` |
| **7-8**   | Thermal      | IEEE C57.91 model, hot-spot calc, aging factor, loss-of-life    | `thermal/`                  |
| **9-10**  | Health Index | Component scores, composite HI, trend analysis                  | `health_index/`             |
| **11-12** | Prediction   | RUL estimator, failure probability, anomaly detection           | `prediction/`               |
| **13-14** | Reporting    | Template engine, PDF reports, alert generation                  | `reporting/`                |
| **15-16** | Dashboard    | Streamlit UI, fleet view, DGA tool, integration testing         | `dashboard/`                |

### Team for This Build

| Role                      | Count         | Responsibility                                |
| ------------------------- | ------------- | --------------------------------------------- |
| **Lead Dev (you)**        | 1             | Architecture, core modules, integration       |
| **Power Systems Advisor** | 1 (part-time) | Validate DGA logic, thermal model, thresholds |

This can be built by **1-2 people**. The power systems advisor can be a consultant, a professor, or a retired utility engineer working 5-10 hours/week to validate your science.

---

## 💰 Actual Costs for This MVP

| Item                      | Cost                             | Notes                         |
| ------------------------- | -------------------------------- | ----------------------------- |
| **Your time (16 weeks)**  | $0 if founder / $40-60K if hired | The real cost                 |
| **Domain consultant**     | $2K-5K                           | Part-time review              |
| **Compute**               | $0                               | Runs on any modern laptop     |
| **Tools**                 | $0                               | All open-source               |
| **IEEE C57.104 Standard** | ~$100                            | One-time purchase (essential) |
| **IEEE C57.91 Standard**  | ~$100                            | One-time purchase (essential) |
| **Domain (website)**      | $12/yr                           | Optional                      |
| **Total**                 | **$200-$5,200**                  | + your living expenses        |

---

## 🎯 How to Prove Value (The Demo Pitch)

When you have the MVP, your demo script is:

> _"Here are 50 transformers. Your team currently checks them all on the same schedule. TransformerGuard analyzed the DGA data you already have and found: 3 transformers are Critical — they need attention NOW. 7 are in decline and will need intervention within 18 months. 40 are fine and don't need your time yet. Here's the report for your worst transformer — with the exact IEEE standard reference explaining why, and exactly what action to take."_

**That's a $500K conversation.** No LLM needed. No cascade simulator needed. No digital twin needed. Just solid DGA science + clear health scoring + actionable recommendations.

---

## 🚀 After the MVP Proves Value

Only then do you expand:

```
MVP (Months 1-4)                    → Transformer Health Scoring
                                         ↓ (proven, paying pilot)
Phase 2 (Months 5-8)               → Fleet-wide risk ranking + loading optimization
                                         ↓ (2-3 utility customers)
Phase 3 (Months 9-14)              → Grid topology integration + cascade risk
                                         ↓ (Series A funding)
Phase 4 (Months 15-20)             → Full AEIP platform (digital twin + AI reasoning)
```

**Build the speedboat. Win the race. Then build the aircraft carrier.**
