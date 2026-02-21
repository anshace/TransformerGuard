"""
Synthetic Data Generator for Transformer DGA and Sensor Data

This module generates realistic synthetic DGA (Dissolved Gas Analysis) and
sensor data for testing and development purposes. Gas profiles are based on
IEEE C57.104 typical gas concentrations for various fault conditions.

Author: TransformerGuard Team
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)


class TransformerCondition(Enum):
    """
    Enumeration of transformer health conditions.
    
    Based on IEEE C57.104 fault type classifications:
    - HEALTHY: Normal aging, no significant faults
    - AGING: Accelerated aging, elevated gases but no specific fault
    - THERMAL_FAULT_LOW: Thermal fault < 300°C (T1)
    - THERMAL_FAULT_MEDIUM: Thermal fault 300-700°C (T2)
    - THERMAL_FAULT_HIGH: Thermal fault > 700°C (T3)
    - PARTIAL_DISCHARGE: Low energy electrical discharge (PD)
    - ARCING: High energy discharge (D2)
    - SPARKING: Low energy discharge (D1)
    """
    
    HEALTHY = "healthy"
    AGING = "aging"
    THERMAL_FAULT_LOW = "thermal_fault_low"
    THERMAL_FAULT_MEDIUM = "thermal_fault_medium"
    THERMAL_FAULT_HIGH = "thermal_fault_high"
    PARTIAL_DISCHARGE = "partial_discharge"
    ARCING = "arcing"
    SPARKING = "sparking"


# Gas profiles for different conditions
# Format: (mean_ppm, std_ppm) - based on IEEE C57.104 typical values
GAS_PROFILES: Dict[TransformerCondition, Dict[str, Tuple[float, float]]] = {
    TransformerCondition.HEALTHY: {
        "h2": (30, 15),
        "ch4": (20, 10),
        "c2h6": (15, 8),
        "c2h4": (10, 5),
        "c2h2": (0.5, 0.3),
        "co": (200, 80),
        "co2": (2000, 500),
    },
    TransformerCondition.AGING: {
        "h2": (80, 30),
        "ch4": (100, 40),
        "c2h6": (60, 25),
        "c2h4": (50, 20),
        "c2h2": (2, 1),
        "co": (500, 150),
        "co2": (5000, 1500),
    },
    TransformerCondition.THERMAL_FAULT_LOW: {
        "h2": (100, 40),
        "ch4": (250, 80),
        "c2h6": (150, 50),
        "c2h4": (100, 40),
        "c2h2": (3, 2),
        "co": (400, 150),
        "co2": (4000, 1200),
    },
    TransformerCondition.THERMAL_FAULT_MEDIUM: {
        "h2": (150, 60),
        "ch4": (400, 120),
        "c2h6": (200, 70),
        "c2h4": (400, 120),
        "c2h2": (15, 8),
        "co": (500, 200),
        "co2": (5000, 1500),
    },
    TransformerCondition.THERMAL_FAULT_HIGH: {
        "h2": (200, 80),
        "ch4": (500, 150),
        "c2h6": (300, 100),
        "c2h4": (800, 250),
        "c2h2": (50, 20),
        "co": (600, 250),
        "co2": (6000, 2000),
    },
    TransformerCondition.PARTIAL_DISCHARGE: {
        "h2": (500, 200),
        "ch4": (100, 50),
        "c2h6": (50, 25),
        "c2h4": (30, 15),
        "c2h2": (5, 3),
        "co": (300, 100),
        "co2": (3000, 1000),
    },
    TransformerCondition.ARCING: {
        "h2": (800, 300),
        "ch4": (300, 120),
        "c2h6": (100, 50),
        "c2h4": (500, 200),
        "c2h2": (200, 80),
        "co": (400, 150),
        "co2": (4000, 1200),
    },
    TransformerCondition.SPARKING: {
        "h2": (400, 150),
        "ch4": (150, 60),
        "c2h6": (60, 25),
        "c2h4": (200, 80),
        "c2h2": (80, 30),
        "co": (350, 120),
        "co2": (3500, 1000),
    },
}


@dataclass
class TransformerConfig:
    """
    Configuration for a synthetic transformer.
    
    Attributes:
        transformer_id: Unique identifier
        name: Human-readable name
        rated_mva: Rated capacity in MVA
        rated_kv: Rated voltage in kV
        manufacture_year: Year of manufacture
        location: Location name
        latitude: GPS latitude
        longitude: GPS longitude
        cooling_type: Cooling type (ONAN, ONAF, etc.)
        oil_type: Insulation oil type
    """
    
    transformer_id: str
    name: str
    rated_mva: float
    rated_kv: float
    manufacture_year: int
    location: str = "Unknown"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    cooling_type: str = "ONAN"
    oil_type: str = "Mineral"


class SyntheticTransformerGenerator:
    """
    Generate realistic synthetic DGA and sensor data.
    
    Based on IEEE C57.104 typical gas concentrations for various
    transformer fault conditions. Supports generating complete fleets
    of transformers with realistic degradation patterns.
    
    Features:
    - Multiple fault condition profiles
    - Realistic gas concentration distributions
    - Time-series degradation modeling
    - Load and temperature profiles
    - Fleet generation with diverse conditions
    
    Example:
        >>> generator = SyntheticTransformerGenerator(seed=42)
        >>> fleet = generator.generate_fleet(n_transformers=50)
        >>> print(fleet.columns.tolist())
        ['transformer_id', 'name', 'rated_mva', ...]
        
        >>> dga_history = generator.generate_dga_history(
        ...     transformer_id="TR-001",
        ...     condition=TransformerCondition.THERMAL_FAULT_MEDIUM,
        ...     n_samples=24,
        ...     years_span=2
        ... )
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the generator with optional random seed.
        
        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        logger.info(f"Initialized synthetic generator with seed={seed}")
    
    def generate_fleet(
        self,
        n_transformers: int = 50,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate a fleet of synthetic transformers.
        
        Args:
            n_transformers: Number of transformers to generate.
            seed: Random seed (overrides instance seed if provided).
            
        Returns:
            DataFrame with transformer configurations.
        """
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        
        logger.info(f"Generating fleet of {n_transformers} transformers")
        
        transformers = []
        for i in range(n_transformers):
            # Generate transformer ID
            transformer_id = f"TR-{i + 1:04d}"
            
            # Generate name based on type
            types = ["MAIN", "SUB", "DIST", "IND", "GEN"]
            type_idx = rng.integers(0, len(types))
            name = f"TR-{types[type_idx]}-{i + 1:03d}"
            
            # Generate ratings
            rated_mva = rng.choice([5, 10, 15, 25, 40, 50, 75, 100, 150, 250])
            rated_kv = rng.choice([11, 33, 66, 132, 220, 400])
            
            # Generate manufacture year (1950-2020)
            manufacture_year = int(rng.integers(1950, 2021))
            
            # Generate location
            locations = [
                "North Substation", "South Substation", "East Substation",
                "West Substation", "Central Station", "Industrial Park",
                "Downtown Station", "Rural Station", "Coastal Station",
            ]
            location = rng.choice(locations)
            
            # Generate coordinates (roughly US-based)
            latitude = round(rng.uniform(25.0, 48.0), 4)
            longitude = round(rng.uniform(-125.0, -70.0), 4)
            
            # Generate cooling type
            cooling_types = ["ONAN", "ONAF", "OFAF", "ODAF"]
            cooling_type = rng.choice(cooling_types)
            
            # Generate oil type
            oil_types = ["Mineral", "Synthetic", "Natural Ester", "Silicone"]
            oil_type = rng.choice(oil_types, p=[0.7, 0.1, 0.15, 0.05])
            
            transformers.append(TransformerConfig(
                transformer_id=transformer_id,
                name=name,
                rated_mva=rated_mva,
                rated_kv=rated_kv,
                manufacture_year=manufacture_year,
                location=location,
                latitude=latitude,
                longitude=longitude,
                cooling_type=cooling_type,
                oil_type=oil_type,
            ))
        
        # Convert to DataFrame
        df = pd.DataFrame([t.__dict__ for t in transformers])
        
        logger.info(f"Generated {len(df)} transformer configurations")
        return df
    
    def generate_dga_history(
        self,
        transformer_id: str,
        condition: Union[TransformerCondition, str],
        n_samples: int = 24,
        years_span: float = 2.0,
        degradation_rate: float = 0.1,
        start_date: Optional[datetime] = None,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate DGA history for a single transformer.
        
        Args:
            transformer_id: Transformer identifier.
            condition: Health condition (TransformerCondition or string).
            n_samples: Number of DGA samples to generate.
            years_span: Time span in years.
            degradation_rate: Rate of gas increase per year (0.0 to 1.0).
            start_date: Starting date for the history.
            seed: Random seed.
            
        Returns:
            DataFrame with DGA history.
        """
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        
        # Parse condition
        if isinstance(condition, str):
            condition = TransformerCondition(condition.lower())
        
        # Get gas profile
        profile = GAS_PROFILES[condition]
        
        # Generate dates
        if start_date is None:
            start_date = datetime.now() - timedelta(days=int(years_span * 365))
        
        end_date = start_date + timedelta(days=int(years_span * 365))
        dates = pd.date_range(start=start_date, end=end_date, periods=n_samples)
        
        # Generate gas concentrations with degradation
        records = []
        for i, date in enumerate(dates):
            # Calculate degradation factor
            degradation_factor = 1 + (degradation_rate * years_span * i / n_samples)
            
            # Generate gas values
            record = {
                "transformer_id": transformer_id,
                "sample_date": date,
            }
            
            for gas, (mean, std) in profile.items():
                # Apply degradation
                current_mean = mean * degradation_factor
                current_std = std * degradation_factor
                
                # Generate value with log-normal distribution (can't be negative)
                value = rng.lognormal(
                    mean=np.log(current_mean),
                    sigma=current_std / current_mean
                )
                record[gas] = round(max(0, value), 2)
            
            # Add temperature and load data
            record["oil_temp"] = round(rng.normal(65, 15), 1)
            record["ambient_temp"] = round(rng.normal(25, 10), 1)
            record["load_pct"] = round(rng.normal(70, 20), 1)
            
            # Ensure load is within bounds
            record["load_pct"] = max(0, min(150, record["load_pct"]))
            
            records.append(record)
        
        df = pd.DataFrame(records)
        logger.info(
            f"Generated {len(df)} DGA samples for {transformer_id} "
            f"with condition {condition.value}"
        )
        
        return df
    
    def generate_fleet_dga_history(
        self,
        fleet_df: pd.DataFrame,
        samples_per_transformer: int = 12,
        years_span: float = 2.0,
        condition_distribution: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate DGA history for an entire fleet.
        
        Args:
            fleet_df: DataFrame with transformer configurations.
            samples_per_transformer: Number of samples per transformer.
            years_span: Time span in years.
            condition_distribution: Distribution of conditions.
                                   If None, uses realistic distribution.
            seed: Random seed.
            
        Returns:
            DataFrame with DGA history for all transformers.
        """
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        
        # Default condition distribution (realistic)
        if condition_distribution is None:
            condition_distribution = {
                "healthy": 0.50,
                "aging": 0.25,
                "thermal_fault_low": 0.08,
                "thermal_fault_medium": 0.05,
                "thermal_fault_high": 0.02,
                "partial_discharge": 0.05,
                "arcing": 0.03,
                "sparking": 0.02,
            }
        
        conditions = list(condition_distribution.keys())
        probabilities = list(condition_distribution.values())
        
        all_records = []
        
        for _, transformer in fleet_df.iterrows():
            # Assign condition
            condition = rng.choice(conditions, p=probabilities)
            
            # Generate degradation rate based on condition
            if condition == "healthy":
                degradation_rate = 0.02
            elif condition == "aging":
                degradation_rate = 0.05
            else:
                degradation_rate = 0.15
            
            # Generate history
            history = self.generate_dga_history(
                transformer_id=transformer["transformer_id"],
                condition=condition,
                n_samples=samples_per_transformer,
                years_span=years_span,
                degradation_rate=degradation_rate,
                seed=rng.integers(0, 1000000),
            )
            
            all_records.append(history)
        
        df = pd.concat(all_records, ignore_index=True)
        df = df.sort_values(["transformer_id", "sample_date"]).reset_index(drop=True)
        
        logger.info(
            f"Generated DGA history for {len(fleet_df)} transformers "
            f"({len(df)} total samples)"
        )
        
        return df
    
    def generate_load_profile(
        self,
        transformer_id: str,
        start_date: datetime,
        end_date: datetime,
        rated_mva: float,
        resolution_hours: int = 1,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic load profile for a transformer.
        
        Args:
            transformer_id: Transformer identifier.
            start_date: Start date.
            end_date: End date.
            rated_mva: Rated capacity in MVA.
            resolution_hours: Time resolution in hours.
            seed: Random seed.
            
        Returns:
            DataFrame with load profile.
        """
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        
        # Generate timestamps
        n_periods = int((end_date - start_date).total_seconds() / 3600 / resolution_hours)
        timestamps = pd.date_range(start=start_date, periods=n_periods, freq=f"{resolution_hours}h")
        
        # Generate load pattern
        records = []
        for ts in timestamps:
            # Base load varies by hour of day
            hour = ts.hour
            
            # Daily pattern: higher during day, lower at night
            if 6 <= hour <= 10:  # Morning ramp
                base_load = 0.6 + 0.2 * (hour - 6) / 4
            elif 10 <= hour <= 18:  # Daytime
                base_load = 0.8
            elif 18 <= hour <= 22:  # Evening
                base_load = 0.7
            else:  # Night
                base_load = 0.4
            
            # Add weekly pattern (lower on weekends)
            if ts.dayofweek >= 5:  # Weekend
                base_load *= 0.7
            
            # Add seasonal variation
            month = ts.month
            if month in [6, 7, 8]:  # Summer
                base_load *= 1.2
            elif month in [12, 1, 2]:  # Winter
                base_load *= 1.1
            
            # Add random variation
            load_variation = rng.normal(0, 0.1)
            load_pct = base_load + load_variation
            
            # Ensure within bounds
            load_pct = max(0.1, min(1.3, load_pct))
            
            # Calculate actual load in MVA
            load_mva = load_pct * rated_mva
            
            records.append({
                "transformer_id": transformer_id,
                "timestamp": ts,
                "load_pct": round(load_pct * 100, 1),
                "load_mva": round(load_mva, 2),
            })
        
        df = pd.DataFrame(records)
        return df
    
    def generate_temperature_profile(
        self,
        transformer_id: str,
        start_date: datetime,
        end_date: datetime,
        base_ambient: float = 25.0,
        resolution_hours: int = 1,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic temperature profile.
        
        Args:
            transformer_id: Transformer identifier.
            start_date: Start date.
            end_date: End date.
            base_ambient: Base ambient temperature in °C.
            resolution_hours: Time resolution in hours.
            seed: Random seed.
            
        Returns:
            DataFrame with temperature profile.
        """
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        
        # Generate timestamps
        n_periods = int((end_date - start_date).total_seconds() / 3600 / resolution_hours)
        timestamps = pd.date_range(start=start_date, periods=n_periods, freq=f"{resolution_hours}h")
        
        records = []
        for ts in timestamps:
            # Ambient temperature varies by hour
            hour = ts.hour
            
            # Daily temperature pattern
            if 6 <= hour <= 14:  # Warming up
                ambient = base_ambient - 5 + 10 * (hour - 6) / 8
            elif 14 <= hour <= 20:  # Cooling down
                ambient = base_ambient + 5 - 8 * (hour - 14) / 6
            else:  # Night
                ambient = base_ambient - 8
            
            # Add seasonal variation
            month = ts.month
            if month in [6, 7, 8]:  # Summer
                ambient += 10
            elif month in [12, 1, 2]:  # Winter
                ambient -= 15
            
            # Add random variation
            ambient += rng.normal(0, 3)
            
            # Calculate oil temperature (typically 20-40°C above ambient)
            oil_temp = ambient + 30 + rng.normal(0, 5)
            
            # Calculate hot spot temperature (typically 10-20°C above oil)
            hotspot_temp = oil_temp + 15 + rng.normal(0, 3)
            
            records.append({
                "transformer_id": transformer_id,
                "timestamp": ts,
                "ambient_temp_c": round(ambient, 1),
                "oil_temp_c": round(oil_temp, 1),
                "hotspot_temp_c": round(hotspot_temp, 1),
            })
        
        df = pd.DataFrame(records)
        return df
    
    def generate_sensor_data(
        self,
        transformer_id: str,
        start_date: datetime,
        end_date: datetime,
        include_vibration: bool = False,
        include_moisture: bool = True,
        resolution_minutes: int = 15,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic sensor data for a transformer.
        
        Args:
            transformer_id: Transformer identifier.
            start_date: Start date.
            end_date: End date.
            include_vibration: Whether to include vibration data.
            include_moisture: Whether to include moisture data.
            resolution_minutes: Time resolution in minutes.
            seed: Random seed.
            
        Returns:
            DataFrame with sensor data.
        """
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        
        # Generate timestamps
        n_periods = int((end_date - start_date).total_seconds() / 60 / resolution_minutes)
        timestamps = pd.date_range(start=start_date, periods=n_periods, freq=f"{resolution_minutes}min")
        
        records = []
        for ts in timestamps:
            record = {
                "transformer_id": transformer_id,
                "timestamp": ts,
            }
            
            # Oil temperature
            record["oil_temp_c"] = round(rng.normal(65, 10), 1)
            
            # Winding temperature (typically 5-15°C above oil)
            record["winding_temp_c"] = round(record["oil_temp_c"] + rng.normal(10, 3), 1)
            
            # Tank pressure (slightly above atmospheric)
            record["tank_pressure_kpa"] = round(rng.normal(102, 2), 2)
            
            # Oil level (percentage)
            record["oil_level_pct"] = round(rng.normal(95, 3), 1)
            record["oil_level_pct"] = max(50, min(100, record["oil_level_pct"]))
            
            # Load current
            record["load_current_a"] = round(rng.normal(500, 100), 1)
            
            if include_moisture:
                # Moisture in oil (ppm)
                record["moisture_ppm"] = round(rng.normal(15, 5), 1)
                record["moisture_ppm"] = max(0, record["moisture_ppm"])
            
            if include_vibration:
                # Vibration amplitude (mm/s)
                record["vibration_mm_s"] = round(rng.normal(2, 0.5), 2)
                record["vibration_mm_s"] = max(0, record["vibration_mm_s"])
            
            records.append(record)
        
        df = pd.DataFrame(records)
        return df
    
    def get_condition_summary(
        self,
        dga_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Summarize the condition distribution in generated DGA data.
        
        Args:
            dga_df: DataFrame with DGA data.
            
        Returns:
            DataFrame with condition summary statistics.
        """
        # Calculate key ratios and totals
        df = dga_df.copy()
        
        df["tcg"] = df["h2"] + df["ch4"] + df["c2h6"] + df["c2h4"] + df["c2h2"]
        df["tdg"] = df["tcg"] + df["co"] + df["co2"]
        
        # Get latest sample for each transformer
        latest = df.sort_values("sample_date").groupby("transformer_id").last().reset_index()
        
        # Classify condition based on gas levels
        def classify_condition(row):
            h2, ch4, c2h6, c2h4, c2h2 = row["h2"], row["ch4"], row["c2h6"], row["c2h4"], row["c2h2"]
            
            # Simple rule-based classification
            if c2h2 > 50:
                return "arcing"
            elif c2h2 > 10:
                return "sparking"
            elif h2 > 300 and c2h2 < 10:
                return "partial_discharge"
            elif c2h4 > 400 and c2h2 > 20:
                return "thermal_fault_high"
            elif c2h4 > 200:
                return "thermal_fault_medium"
            elif c2h4 > 50 or ch4 > 150:
                return "thermal_fault_low"
            elif row["tcg"] > 200:
                return "aging"
            else:
                return "healthy"
        
        latest["condition"] = latest.apply(classify_condition, axis=1)
        
        # Summary statistics
        summary = latest.groupby("condition").agg({
            "transformer_id": "count",
            "tcg": ["mean", "std", "max"],
            "h2": "mean",
            "c2h2": "mean",
        }).round(2)
        
        summary.columns = ["_".join(col).strip() for col in summary.columns.values]
        summary = summary.rename(columns={"transformer_id_count": "count"})
        
        return summary.reset_index()