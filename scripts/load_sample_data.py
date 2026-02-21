#!/usr/bin/env python
"""
Script to load sample data into the TransformerGuard database.
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import get_session, init_db
from src.database.models import DGARecord, Transformer


def load_transformers(csv_path: str) -> list[Transformer]:
    """Load transformer data from CSV file."""
    session = get_session()
    transformers = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            transformer = Transformer(
                id=int(row["id"]),
                name=row["name"],
                serial_number=row["serial_number"],
                manufacturer=row["manufacturer"],
                manufacture_date=datetime.strptime(
                    row["manufacture_date"], "%Y-%m-%d"
                ).date()
                if row["manufacture_date"]
                else None,
                installation_date=datetime.strptime(
                    row["installation_date"], "%Y-%m-%d"
                ).date()
                if row["installation_date"]
                else None,
                rated_mva=float(row["rated_mva"]) if row["rated_mva"] else None,
                rated_voltage_kv=float(row["rated_voltage_kv"])
                if row["rated_voltage_kv"]
                else None,
                cooling_type=row["cooling_type"],
                location=row["location"],
                substation=row["substation"],
            )
            transformers.append(transformer)
            session.merge(transformer)

    session.commit()
    print(f"Loaded {len(transformers)} transformers")
    return transformers


def load_dga_records(csv_path: str) -> list[DGARecord]:
    """Load DGA records from CSV file."""
    session = get_session()
    records = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = DGARecord(
                transformer_id=int(row["transformer_id"]),
                sample_date=datetime.strptime(row["sample_date"], "%Y-%m-%d"),
                h2=float(row["h2"]) if row["h2"] else None,
                ch4=float(row["ch4"]) if row["ch4"] else None,
                c2h2=float(row["c2h2"]) if row["c2h2"] else None,
                c2h4=float(row["c2h4"]) if row["c2h4"] else None,
                c2h6=float(row["c2h6"]) if row["c2h6"] else None,
                co=float(row["co"]) if row["co"] else None,
                co2=float(row["co2"]) if row["co2"] else None,
                o2=float(row["o2"]) if row["o2"] else None,
                n2=float(row["n2"]) if row["n2"] else None,
                lab_name=row["lab_name"],
                sample_type=row["sample_type"],
            )
            records.append(record)
            session.add(record)

    session.commit()
    print(f"Loaded {len(records)} DGA records")
    return records


def main():
    """Main function to load all sample data."""
    # Initialize database first
    print("Initializing database...")
    db_conn = init_db()
    db_conn.create_tables()

    data_dir = Path(__file__).parent.parent / "data" / "sample"

    # Load transformers first
    print("Loading transformers...")
    load_transformers(data_dir / "transformer_fleet.csv")

    # Load DGA records
    print("Loading DGA records...")
    load_dga_records(data_dir / "dga_samples.csv")

    print("Sample data loaded successfully!")


if __name__ == "__main__":
    main()
