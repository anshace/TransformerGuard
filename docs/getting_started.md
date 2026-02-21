# Getting Started with TransformerGuard

## Overview

TransformerGuard is an AI-powered transformer health scoring and failure prediction system that:
- Ingests DGA (Dissolved Gas Analysis), thermal, and load data
- Outputs a 0-100 health score
- Estimates remaining useful life (RUL)
- Provides plain-English maintenance recommendations

## Prerequisites

- Python 3.10+
- pixi (package manager) or pip
- SQLite (default) or PostgreSQL (for production)

## Installation

### Using pixi (recommended)

```bash
pixi install
```

### Using pip

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Initialize the database:**

```bash
python -c "from src.database import init_db; init_db()"
```

2. **Start the API server:**

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

3. **Start the dashboard:**

```bash
cd dashboard && streamlit run app.py
```

4. **Access the application:**
   - API Documentation: http://localhost:8000/docs
   - Dashboard: http://localhost:8501

## Loading Sample Data

```bash
python scripts/load_sample_data.py
```

## Running Tests

```bash
pytest tests/
```

## Project Structure

```
TransformerGuard/
├── config/                      # Configuration files
│   ├── settings.yaml           # Application settings
│   ├── dga_thresholds.yaml    # IEEE C57.104 gas thresholds
│   ├── thermal_params.yaml    # IEEE C57.91 thermal parameters
│   └── health_index_weights.yaml  # Health index scoring weights
├── src/                        # Source code
│   ├── api/                    # FastAPI application
│   │   ├── main.py            # Application entry point
│   │   ├── routes/            # API endpoints
│   │   │   ├── transformers.py
│   │   │   ├── dga.py
│   │   │   ├── health.py
│   │   │   ├── predictions.py
│   │   │   ├── alerts.py
│   │   │   └── reports.py
│   │   └── schemas/           # Pydantic models
│   ├── database/              # Database models and connection
│   │   ├── models.py          # SQLAlchemy models
│   │   └── connection.py      # Database connection
│   ├── diagnosis/             # DGA interpretation methods
│   │   ├── duval_triangle.py  # Duval Triangle method
│   │   ├── rogers_ratios.py  # Rogers Ratio method
│   │   ├── iec_ratios.py      # IEC Ratio method
│   │   ├── key_gas.py         # Key Gas method
│   │   ├── doernenburg.py     # Doernenburg method
│   │   └── multi_method.py    # Ensemble method
│   ├── thermal/               # Thermal modeling
│   │   ├── ieee_c57_91.py    # IEEE C57.91 model
│   │   ├── hotspot_calculator.py
│   │   ├── aging_model.py
│   │   └── loss_of_life.py
│   ├── health_index/          # Health index calculation
│   │   ├── composite_hi.py   # Composite health index
│   │   ├── dga_score.py      # DGA component score
│   │   ├── oil_quality_score.py
│   │   ├── electrical_score.py
│   │   ├── age_score.py
│   │   └── loading_score.py
│   ├── prediction/            # RUL and failure prediction
│   │   ├── rul_estimator.py  # Remaining useful life
│   │   ├── failure_probability.py
│   │   ├── anomaly_detector.py
│   │   └── gas_trend_forecast.py
│   └── reporting/             # Reports and recommendations
│       ├── action_recommender.py
│       ├── alert_generator.py
│       └── pdf_report.py
├── dashboard/                  # Streamlit dashboard
│   ├── app.py
│   ├── pages/
│   │   ├── fleet_overview.py
│   │   ├── transformer_detail.py
│   │   ├── dga_analysis.py
│   │   ├── trend_monitor.py
│   │   └── alert_center.py
│   └── components/
├── data/                       # Data directory
│   ├── sample/                # Sample data files
│   ├── processed/
│   └── models/
├── docs/                       # Documentation
├── tests/                      # Test suite
├── notebooks/                  # Jupyter notebooks
└── reporting/                  # Report outputs
```

## Configuration

Configuration files are located in the `config/` directory:

| File | Description |
|------|-------------|
| `settings.yaml` | Application settings (database, API, logging) |
| `dga_thresholds.yaml` | IEEE C57.104 gas thresholds |
| `thermal_params.yaml` | IEEE C57.91 thermal parameters |
| `health_index_weights.yaml` | Health index scoring weights |

### Sample Configuration (settings.yaml)

```yaml
app:
  name: "TransformerGuard"
  version: "1.0.0"
  debug: false

database:
  type: "sqlite"
  path: "data/transformerguard.db"

api:
  host: "0.0.0.0"
  port: 8000

dashboard:
  title: "TransformerGuard Dashboard"
  theme: "light"
  refresh_interval: 300
```

## Next Steps

- See [API Reference](api_reference.md) for detailed API documentation
- See [DGA Methods](dga_methods.md) for information on DGA interpretation
- See [Thermal Model](thermal_model.md) for IEEE C57.91 thermal modeling
- See [Health Index](health_index.md) for health scoring methodology
- See [Deployment](deployment.md) for production deployment instructions
