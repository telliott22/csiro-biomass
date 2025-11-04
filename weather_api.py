"""
Weather Feature Engineering for Pasture Biomass Prediction

Fetches historical weather data from Open-Meteo API with caching.
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import time
from typing import Dict, Tuple, Optional


class WeatherAPIClient:
    """Client for fetching historical weather data with caching."""

    # Representative locations for Australian cattle farming regions
    LOCATIONS = {
        'Tas': {'lat': -40.85, 'lon': 145.12, 'name': 'Smithton'},      # NW Tasmania dairy
        'Vic': {'lat': -38.38, 'lon': 142.48, 'name': 'Warrnambool'},   # SW Victoria dairy
        'NSW': {'lat': -30.98, 'lon': 150.26, 'name': 'Gunnedah'},      # Northern NSW grazing
        'WA': {'lat': -35.03, 'lon': 117.88, 'name': 'Albany'}          # SW WA high rainfall
    }

    def __init__(self, cache_dir='weather_cache'):
        """
        Initialize weather API client.

        Args:
            cache_dir: Directory to store cached API responses
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        self.session = requests.Session()

    def _get_cache_key(self, state: str, start_date: str, end_date: str) -> str:
        """Generate cache key for a request."""
        key_str = f"{state}_{start_date}_{end_date}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path for cached response."""
        return self.cache_dir / f"{cache_key}.json"

    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load data from cache if available."""
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                return json.load(f)
        return None

    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save data to cache."""
        cache_path = self._get_cache_path(cache_key)
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def fetch_weather_data(self, state: str, start_date: str, end_date: str,
                          days_before: int = 30) -> pd.DataFrame:
        """
        Fetch historical weather data for a state and date range.

        Args:
            state: Australian state code (Tas, Vic, NSW, WA)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            days_before: Extra days to fetch before start_date for rolling calculations

        Returns:
            DataFrame with daily weather data
        """
        # Extend date range to include days before for rolling calculations
        start = pd.to_datetime(start_date) - timedelta(days=days_before)
        end = pd.to_datetime(end_date)

        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')

        # Check cache
        cache_key = self._get_cache_key(state, start_str, end_str)
        cached_data = self._load_from_cache(cache_key)

        if cached_data is not None:
            print(f"✓ Loaded from cache: {state} ({start_str} to {end_str})")
            df = pd.DataFrame(cached_data)
            df['date'] = pd.to_datetime(df['date'])
            return df

        # Fetch from API
        location = self.LOCATIONS[state]
        params = {
            'latitude': location['lat'],
            'longitude': location['lon'],
            'start_date': start_str,
            'end_date': end_str,
            'daily': [
                'temperature_2m_max',
                'temperature_2m_min',
                'temperature_2m_mean',
                'precipitation_sum',
                'et0_fao_evapotranspiration',  # Reference evapotranspiration
                'rain_sum',
                'shortwave_radiation_sum',
                'windspeed_10m_max'
            ],
            'timezone': 'Australia/Sydney'
        }

        print(f"→ Fetching from API: {state} ({location['name']}) {start_str} to {end_str}")

        try:
            # Retry logic for rate limiting
            max_retries = 5
            retry_delay = 10  # Start with 10 seconds for large requests

            for attempt in range(max_retries):
                response = self.session.get(self.base_url, params=params, timeout=30)

                if response.status_code == 429:  # Rate limited
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff: 10s, 20s, 40s, 80s
                        print(f"  ⚠ Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"  ✗ Rate limit exceeded after {max_retries} retries")
                        raise

                response.raise_for_status()
                data = response.json()
                break

            # Convert to DataFrame
            daily_data = data['daily']
            df = pd.DataFrame({
                'date': pd.to_datetime(daily_data['time']),
                'temp_max': daily_data['temperature_2m_max'],
                'temp_min': daily_data['temperature_2m_min'],
                'temp_mean': daily_data['temperature_2m_mean'],
                'precipitation': daily_data['precipitation_sum'],
                'et0': daily_data['et0_fao_evapotranspiration'],
                'rain': daily_data['rain_sum'],
                'solar_radiation': daily_data['shortwave_radiation_sum'],
                'windspeed_max': daily_data['windspeed_10m_max']
            })

            # Add state and location info
            df['state'] = state
            df['lat'] = location['lat']
            df['lon'] = location['lon']

            # Save to cache (convert dates to strings for JSON serialization)
            df_to_cache = df.copy()
            df_to_cache['date'] = df_to_cache['date'].dt.strftime('%Y-%m-%d')
            self._save_to_cache(cache_key, df_to_cache.to_dict('records'))
            print(f"  ✓ Cached {len(df)} days of data")

            # Rate limiting - be nice to the API
            # Longer delay for large requests (30+ years)
            days_fetched = (end - start).days
            if days_fetched > 3650:  # More than 10 years
                time.sleep(3.0)
            elif days_fetched > 365:  # More than 1 year
                time.sleep(1.5)
            else:
                time.sleep(0.5)

            return df

        except requests.exceptions.RequestException as e:
            print(f"  ✗ API request failed: {e}")
            raise

    def fetch_climatology(self, state: str, start_year: int = 1991,
                         end_year: int = 2020) -> pd.DataFrame:
        """
        Fetch long-term climatology for calculating anomalies.

        Args:
            state: State code
            start_year: Start year for climatology period
            end_year: End year for climatology period

        Returns:
            DataFrame with climatology statistics by day-of-year
        """
        cache_key = self._get_cache_key(state, f"clima_{start_year}", f"{end_year}")
        cached_data = self._load_from_cache(cache_key)

        if cached_data is not None:
            print(f"✓ Loaded climatology from cache: {state}")
            return pd.DataFrame(cached_data)

        print(f"→ Fetching climatology: {state} ({start_year}-{end_year})")

        # Fetch data for climatology period
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"

        df = self.fetch_weather_data(state, start_date, end_date, days_before=0)

        # Calculate statistics by day of year
        df['dayofyear'] = df['date'].dt.dayofyear

        climatology = df.groupby('dayofyear').agg({
            'precipitation': ['mean', 'std'],
            'temp_mean': ['mean', 'std'],
            'temp_max': ['mean', 'std'],
            'temp_min': ['mean', 'std'],
            'et0': ['mean', 'std']
        }).reset_index()

        # Flatten column names
        climatology.columns = ['_'.join(col).strip('_') for col in climatology.columns.values]
        climatology['state'] = state

        # Save to cache
        self._save_to_cache(cache_key, climatology.to_dict('records'))
        print(f"  ✓ Cached climatology for {state}")

        return climatology


def calculate_daylength(latitude: float, date: pd.Timestamp) -> float:
    """
    Calculate hours of daylight for a given latitude and date.
    Uses a simplified formula (accurate within a few minutes).

    Args:
        latitude: Latitude in degrees
        date: Date

    Returns:
        Hours of daylight
    """
    # Day of year
    day_of_year = date.dayofyear

    # Solar declination (degrees)
    declination = 23.45 * np.sin(np.radians(360/365 * (day_of_year + 284)))

    # Hour angle at sunrise/sunset
    lat_rad = np.radians(latitude)
    dec_rad = np.radians(declination)

    cos_hour_angle = -np.tan(lat_rad) * np.tan(dec_rad)

    # Handle polar day/night
    if cos_hour_angle > 1:
        return 0.0  # Polar night
    elif cos_hour_angle < -1:
        return 24.0  # Polar day

    hour_angle = np.degrees(np.arccos(cos_hour_angle))
    daylength = 2 * hour_angle / 15  # Convert degrees to hours

    return daylength


def get_southern_hemisphere_season(date: pd.Timestamp) -> int:
    """
    Get season for Southern Hemisphere.

    Returns:
        0: Summer (Dec-Feb)
        1: Autumn (Mar-May)
        2: Winter (Jun-Aug)
        3: Spring (Sep-Nov)
    """
    month = date.month
    if month in [12, 1, 2]:
        return 0  # Summer
    elif month in [3, 4, 5]:
        return 1  # Autumn
    elif month in [6, 7, 8]:
        return 2  # Winter
    else:  # 9, 10, 11
        return 3  # Spring


if __name__ == "__main__":
    # Test the API client
    print("Testing Weather API Client\n")
    print("="*60)

    client = WeatherAPIClient()

    # Test fetching data for Tasmania in 2015
    df = client.fetch_weather_data('Tas', '2015-05-01', '2015-05-31')
    print(f"\nFetched {len(df)} days of data")
    print(df.head())

    # Test daylength calculation
    test_date = pd.Timestamp('2015-05-15')
    daylength = calculate_daylength(-40.85, test_date)
    print(f"\nDaylength on {test_date.date()} at -40.85° latitude: {daylength:.2f} hours")

    # Test season
    season = get_southern_hemisphere_season(test_date)
    season_names = ['Summer', 'Autumn', 'Winter', 'Spring']
    print(f"Season: {season_names[season]}")
