# TransformerGuard

A Focused, Buildable Transformer Health Intelligence System

## Quick Start

```bash
# Install dependencies
pixi install

# Initialize database
pixi run python -c "from src.database import init_db; init_db()"

# Load sample data
pixi run python scripts/load_sample_data.py

# Run dashboard
pixi run streamlit run dashboard/app.py
```
