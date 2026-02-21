# Database Migrations

This directory is intended for Alembic database migrations for the TransformerGuard application.

## Overview

Alembic is a database migration tool that handles schema changes over time. It works with SQLAlchemy to manage database versions.

## Setup

### Install Alembic

```bash
pip install alembic
```

### Initialize Alembic (First Time)

If this is the first time setting up migrations:

```bash
# Navigate to project root
cd /home/ansh/Documents/TransformerGuard

# Initialize Alembic configuration
alembic init src/database/migrations/alembic
```

### Configure Alembic

Edit `alembic.ini` to set the SQLAlchemy URL:

```ini
sqlalchemy.url = sqlite:///data/transformerguard.db
```

Or use environment variables:

```ini
sqlalchemy.url = driver://user:pass@localhost/dbname
```

## Common Commands

### Create a New Migration

Auto-generate a migration based on model changes:

```bash
alembic revision --autogenerate -m "Add new column"
```

Create an empty migration:

```bash
alembic revision -m "Create new table"
```

### Apply Migrations

Apply all pending migrations:

```bash
alembic upgrade head
```

Apply a specific migration:

```bash
alembic upgrade <revision_id>
```

### Rollback Migrations

Undo the last migration:

```bash
alembic downgrade -1
```

Undo all migrations:

```bash
alembic downgrade base
```

### Check Status

Show current migration:

```bash
alembic current
```

Show migration history:

```bash
alembic history --verbose
```

Show pending migrations:

```bash
alembic check
```

## Migration Script Structure

A typical migration file looks like:

```python
"""Add new column

Revision ID: abc123
Revises: 
Create Date: 2024-01-01 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = 'abc123'
down_revision = 'previous_revision'
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.add_column('table_name', sa.Column('new_column', sa.String()))

def downgrade() -> None:
    op.drop_column('table_name', 'new_column')
```

## Using the Database Without Migrations

For development or testing, you can create tables directly from models:

```python
from src.database import init_db, DatabaseConnection

# Initialize database connection
db = init_db()

# Create all tables
db.create_tables()

# Or for testing - recreate tables
db.recreate_tables()
```

## Best Practices

1. **Always create migrations** when changing models, even for small changes
2. **Test migrations** in a development environment first
3. **Keep migrations small** - one change per migration when possible
4. **Include rollback** logic in every migration
5. **Use meaningful messages** for migration names

## Configuration

The Alembic configuration is stored in:
- `alembic.ini` - Main configuration file
- `src/database/migrations/env.py` - Environment configuration
- `src/database/migrations/script.py.mako` - Migration script template

## Support

For more information, see the [Alembic documentation](https://alembic.sqlalchemy.org/).
