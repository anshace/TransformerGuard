"""
Weather API Client for Historical Temperature Data

This module provides a client for the Open-Meteo historical weather API
to retrieve temperature data for thermal modeling of transformers.

Open-Meteo is a free, open-source weather API that doesn't require an API key.

Author: TransformerGuard Team
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)

# Try to import httpx, fall back to requests
try:
    import httpx
    USE_HTTPX = True
except ImportError:
    import requests
    USE_HTTPX = False


@dataclass
class WeatherData:
    """
    Data class representing historical weather data.
    
    Attributes:
        latitude: Location latitude
        longitude: Location longitude
        start_date: Start date of the data
        end_date: End date of the data
        hourly_temperature: List of hourly temperature values in °C
        hourly_timestamps: List of timestamps for each temperature reading
        hourly_humidity: List of hourly relative humidity values in % (optional)
        hourly_pressure: List of hourly pressure values in hPa (optional)
        metadata: Additional metadata about the request
    """
    
    latitude: float
    longitude: float
    start_date: str
    end_date: str
    hourly_temperature: List[float]
    hourly_timestamps: List[datetime]
    hourly_humidity: Optional[List[float]] = None
    hourly_pressure: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert weather data to a pandas DataFrame.
        
        Returns:
            DataFrame with timestamp index and temperature column.
        """
        df = pd.DataFrame({
            "timestamp": self.hourly_timestamps,
            "temperature_c": self.hourly_temperature,
        })
        
        if self.hourly_humidity:
            df["humidity_pct"] = self.hourly_humidity
        if self.hourly_pressure:
            df["pressure_hpa"] = self.hourly_pressure
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        
        return df
    
    @property
    def average_temperature(self) -> float:
        """Calculate average temperature over the period."""
        if not self.hourly_temperature:
            return 0.0
        return sum(self.hourly_temperature) / len(self.hourly_temperature)
    
    @property
    def max_temperature(self) -> float:
        """Get maximum temperature in the period."""
        if not self.hourly_temperature:
            return 0.0
        return max(self.hourly_temperature)
    
    @property
    def min_temperature(self) -> float:
        """Get minimum temperature in the period."""
        if not self.hourly_temperature:
            return 0.0
        return min(self.hourly_temperature)
    
    @property
    def daily_average_temperatures(self) -> Dict[str, float]:
        """
        Calculate daily average temperatures.
        
        Returns:
            Dictionary mapping date strings to average temperatures.
        """
        df = self.to_dataframe()
        daily_avg = df["temperature_c"].resample("D").mean()
        return {
            date.strftime("%Y-%m-%d"): temp 
            for date, temp in daily_avg.items()
        }


class WeatherClient:
    """
    Client for Open-Meteo Historical Weather API.
    
    Provides access to historical weather data for thermal modeling
    of transformers. Uses caching to minimize API calls.
    
    Open-Meteo API Documentation:
    https://open-meteo.com/en/docs/historical-weather-api
    
    Features:
    - Free API (no key required)
    - Hourly historical temperature data
    - Local file-based caching
    - Automatic retry on failures
    - Rate limiting support
    
    Example:
        >>> client = WeatherClient()
        >>> weather = client.get_historical_weather(
        ...     lat=40.7128,
        ...     lon=-74.0060,
        ...     start_date="2024-01-01",
        ...     end_date="2024-01-31"
        ... )
        >>> print(f"Average temp: {weather.average_temperature:.1f}°C")
        Average temp: 2.3°C
    """
    
    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    CACHE_DIR = Path(".cache/weather")
    CACHE_EXPIRY_HOURS = 24  # Cache expires after 24 hours
    
    def __init__(
        self,
        cache_enabled: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the weather client.
        
        Args:
            cache_enabled: Whether to enable caching.
            cache_dir: Directory for cache storage. If None, uses default.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts.
            retry_delay: Delay between retries in seconds.
        """
        self.cache_enabled = cache_enabled
        self.cache_dir = Path(cache_dir) if cache_dir else self.CACHE_DIR
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Weather cache directory: {self.cache_dir}")
    
    def get_historical_weather(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        include_humidity: bool = False,
        include_pressure: bool = False,
    ) -> WeatherData:
        """
        Get historical weather data for a location and time period.
        
        Args:
            lat: Latitude coordinate (-90 to 90).
            lon: Longitude coordinate (-180 to 180).
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            include_humidity: Whether to include relative humidity data.
            include_pressure: Whether to include pressure data.
            
        Returns:
            WeatherData object with hourly temperature data.
            
        Raises:
            ValueError: If parameters are invalid.
            RuntimeError: If API request fails after retries.
        """
        # Validate parameters
        self._validate_coordinates(lat, lon)
        self._validate_dates(start_date, end_date)
        
        logger.info(
            f"Fetching weather data for ({lat}, {lon}) "
            f"from {start_date} to {end_date}"
        )
        
        # Check cache first
        cache_key = self._get_cache_key(lat, lon, start_date, end_date)
        if self.cache_enabled:
            cached_data = self._get_from_cache(cache_key)
            if cached_data:
                logger.info("Using cached weather data")
                return self._dict_to_weather_data(cached_data)
        
        # Build API request
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m",
            "timezone": "auto",
        }
        
        if include_humidity:
            params["hourly"] += ",relative_humidity_2m"
        if include_pressure:
            params["hourly"] += ",pressure_msl"
        
        # Make API request with retries
        response_data = self._make_request(params)
        
        # Parse response
        weather_data = self._parse_response(
            response_data, 
            lat, 
            lon, 
            start_date, 
            end_date,
            include_humidity,
            include_pressure,
        )
        
        # Cache the result
        if self.cache_enabled:
            self._save_to_cache(cache_key, weather_data.to_dict())
        
        return weather_data
    
    def _validate_coordinates(self, lat: float, lon: float) -> None:
        """Validate latitude and longitude values."""
        if not -90 <= lat <= 90:
            raise ValueError(f"Invalid latitude: {lat}. Must be between -90 and 90.")
        if not -180 <= lon <= 180:
            raise ValueError(f"Invalid longitude: {lon}. Must be between -180 and 180.")
    
    def _validate_dates(self, start_date: str, end_date: str) -> None:
        """Validate date format and range."""
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")
        
        if start > end:
            raise ValueError(f"Start date ({start_date}) must be before end date ({end_date})")
        
        # Open-Meteo has data from ~1940 onwards
        min_date = datetime(1940, 1, 1)
        if start < min_date:
            raise ValueError(f"Start date too early. Data available from {min_date.strftime('%Y-%m-%d')}")
    
    def _get_cache_key(
        self, 
        lat: float, 
        lon: float, 
        start_date: str, 
        end_date: str
    ) -> str:
        """Generate a unique cache key for the request."""
        key_string = f"{lat:.4f}_{lon:.4f}_{start_date}_{end_date}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve data from cache if available and not expired.
        
        Args:
            cache_key: Unique cache key.
            
        Returns:
            Cached data dictionary or None if not found/expired.
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        # Check cache age
        cache_age = time.time() - cache_file.stat().st_mtime
        if cache_age > self.CACHE_EXPIRY_HOURS * 3600:
            logger.debug(f"Cache expired for key {cache_key}")
            return None
        
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read cache file: {e}")
            return None
    
    def _save_to_cache(
        self, 
        cache_key: str, 
        data: Dict[str, Any]
    ) -> None:
        """
        Save data to cache.
        
        Args:
            cache_key: Unique cache key.
            data: Data dictionary to cache.
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, "w") as f:
                json.dump(data, f, default=str)
            logger.debug(f"Saved weather data to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache file: {e}")
    
    def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make API request with retry logic.
        
        Args:
            params: Request parameters.
            
        Returns:
            JSON response data.
            
        Raises:
            RuntimeError: If request fails after all retries.
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                if USE_HTTPX:
                    return self._make_httpx_request(params)
                else:
                    return self._make_requests_request(params)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"API request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        
        raise RuntimeError(f"API request failed after {self.max_retries} retries: {last_error}")
    
    def _make_httpx_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make request using httpx client."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            return response.json()
    
    def _make_requests_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make request using requests library."""
        response = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def _parse_response(
        self,
        response_data: Dict[str, Any],
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        include_humidity: bool,
        include_pressure: bool,
    ) -> WeatherData:
        """
        Parse API response into WeatherData object.
        
        Args:
            response_data: Raw API response.
            lat: Latitude.
            lon: Longitude.
            start_date: Start date.
            end_date: End date.
            include_humidity: Whether humidity was requested.
            include_pressure: Whether pressure was requested.
            
        Returns:
            WeatherData object.
        """
        hourly = response_data.get("hourly", {})
        
        # Extract timestamps
        timestamps = []
        for ts in hourly.get("time", []):
            timestamps.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
        
        # Extract temperatures
        temperatures = hourly.get("temperature_2m", [])
        
        # Extract optional data
        humidity = None
        pressure = None
        
        if include_humidity:
            humidity = hourly.get("relative_humidity_2m")
        
        if include_pressure:
            pressure = hourly.get("pressure_msl")
        
        # Build metadata
        metadata = {
            "api_source": "Open-Meteo",
            "units": {
                "temperature": "°C",
                "humidity": "%",
                "pressure": "hPa",
            },
            "timezone": response_data.get("timezone"),
            "elevation": response_data.get("elevation"),
            "generation_time_ms": response_data.get("generationtime_ms"),
        }
        
        return WeatherData(
            latitude=lat,
            longitude=lon,
            start_date=start_date,
            end_date=end_date,
            hourly_temperature=temperatures,
            hourly_timestamps=timestamps,
            hourly_humidity=humidity,
            hourly_pressure=pressure,
            metadata=metadata,
        )
    
    def _dict_to_weather_data(self, data: Dict[str, Any]) -> WeatherData:
        """Convert dictionary to WeatherData object."""
        # Parse timestamps
        timestamps = []
        for ts in data.get("hourly_timestamps", []):
            if isinstance(ts, str):
                timestamps.append(datetime.fromisoformat(ts))
            elif isinstance(ts, datetime):
                timestamps.append(ts)
        
        return WeatherData(
            latitude=data["latitude"],
            longitude=data["longitude"],
            start_date=data["start_date"],
            end_date=data["end_date"],
            hourly_temperature=data["hourly_temperature"],
            hourly_timestamps=timestamps,
            hourly_humidity=data.get("hourly_humidity"),
            hourly_pressure=data.get("hourly_pressure"),
            metadata=data.get("metadata", {}),
        )
    
    def get_temperature_for_date(
        self,
        lat: float,
        lon: float,
        date: str,
    ) -> Dict[str, float]:
        """
        Get temperature statistics for a specific date.
        
        Args:
            lat: Latitude coordinate.
            lon: Longitude coordinate.
            date: Date in YYYY-MM-DD format.
            
        Returns:
            Dictionary with min, max, avg temperatures.
        """
        weather = self.get_historical_weather(lat, lon, date, date)
        
        return {
            "date": date,
            "min_temp": weather.min_temperature,
            "max_temp": weather.max_temperature,
            "avg_temp": weather.average_temperature,
        }
    
    def get_monthly_summary(
        self,
        lat: float,
        lon: float,
        year: int,
        month: int,
    ) -> Dict[str, Any]:
        """
        Get monthly temperature summary.
        
        Args:
            lat: Latitude coordinate.
            lon: Longitude coordinate.
            year: Year (e.g., 2024).
            month: Month (1-12).
            
        Returns:
            Dictionary with monthly statistics.
        """
        # Calculate date range for the month
        start_date = f"{year:04d}-{month:02d}-01"
        if month == 12:
            end_date = f"{year:04d}-12-31"
        else:
            end_date = f"{year:04d}-{month:02d}-{_days_in_month(year, month):02d}"
        
        weather = self.get_historical_weather(lat, lon, start_date, end_date)
        
        return {
            "year": year,
            "month": month,
            "start_date": start_date,
            "end_date": end_date,
            "min_temp": weather.min_temperature,
            "max_temp": weather.max_temperature,
            "avg_temp": weather.average_temperature,
            "daily_averages": weather.daily_average_temperatures,
        }
    
    def clear_cache(self) -> int:
        """
        Clear all cached weather data.
        
        Returns:
            Number of cache files deleted.
        """
        if not self.cache_dir.exists():
            return 0
        
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")
        
        logger.info(f"Cleared {count} cache files")
        return count


def _days_in_month(year: int, month: int) -> int:
    """Get the number of days in a month."""
    if month == 2:
        # Check for leap year
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            return 29
        return 28
    elif month in (4, 6, 9, 11):
        return 30
    else:
        return 31