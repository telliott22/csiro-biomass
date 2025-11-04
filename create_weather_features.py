"""
Create weather features for the training data.
Run this script to generate train_enriched.csv
"""

import pandas as pd
import numpy as np
from weather_api import WeatherAPIClient, calculate_daylength, get_southern_hemisphere_season

print("="*60)
print("WEATHER FEATURE ENGINEERING")
print("="*60)

# 1. Load training data
print("\n1. Loading training data...")
train_df = pd.read_csv('competition/train.csv')
train_wide = train_df.pivot_table(
    index=['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'],
    columns='target_name',
    values='target'
).reset_index()

train_wide['Sampling_Date'] = pd.to_datetime(train_wide['Sampling_Date'])
print(f"   ✓ Loaded {len(train_wide)} samples")

# 2. Fetch weather data
print("\n2. Fetching weather data...")
weather_client = WeatherAPIClient(cache_dir='weather_cache')
min_date = train_wide['Sampling_Date'].min()
max_date = train_wide['Sampling_Date'].max()

weather_data = {}
for state in train_wide['State'].unique():
    print(f"   {state}...", end=" ")
    df = weather_client.fetch_weather_data(
        state=state,
        start_date=min_date.strftime('%Y-%m-%d'),
        end_date=max_date.strftime('%Y-%m-%d'),
        days_before=30
    )
    weather_data[state] = df
    print("✓")

# 3. Calculate features
print("\n3. Calculating weather features...")

def enrich_weather_data(df):
    """Add all weather features."""
    df = df.copy().sort_values('date')

    # Rolling features
    df['rainfall_7d'] = df['precipitation'].rolling(7, min_periods=1).sum()
    df['rainfall_30d'] = df['precipitation'].rolling(30, min_periods=1).sum()
    df['temp_max_7d'] = df['temp_max'].rolling(7, min_periods=1).mean()
    df['temp_min_7d'] = df['temp_min'].rolling(7, min_periods=1).mean()
    df['temp_mean_7d'] = df['temp_mean'].rolling(7, min_periods=1).mean()
    df['temp_mean_30d'] = df['temp_mean'].rolling(30, min_periods=1).mean()
    df['temp_range_7d'] = (df['temp_max'] - df['temp_min']).rolling(7, min_periods=1).mean()
    df['et0_7d'] = df['et0'].rolling(7, min_periods=1).sum()
    df['et0_30d'] = df['et0'].rolling(30, min_periods=1).sum()

    # Water balance
    df['water_balance_7d'] = df['rainfall_7d'] - df['et0_7d']
    df['water_balance_30d'] = df['rainfall_30d'] - df['et0_30d']

    # Days since rain
    days_counter = 0
    days_list = []
    for precip in df['precipitation']:
        if precip > 5:
            days_counter = 0
        else:
            days_counter += 1
        days_list.append(days_counter)
    df['days_since_rain'] = days_list

    # Daylength and season
    df['daylength'] = df.apply(lambda row: calculate_daylength(row['lat'], row['date']), axis=1)
    df['season'] = df['date'].apply(get_southern_hemisphere_season)

    return df

for state in weather_data.keys():
    weather_data[state] = enrich_weather_data(weather_data[state])
    print(f"   ✓ {state}")

# 4. Merge with training data
print("\n4. Merging with training data...")
all_weather = pd.concat(weather_data.values(), ignore_index=True)

weather_features = [
    'rainfall_7d', 'rainfall_30d',
    'temp_max_7d', 'temp_min_7d', 'temp_mean_7d', 'temp_mean_30d', 'temp_range_7d',
    'et0_7d', 'et0_30d',
    'water_balance_7d', 'water_balance_30d',
    'days_since_rain',
    'daylength', 'season'
]

weather_for_merge = all_weather[['date', 'state'] + weather_features].copy()
weather_for_merge.columns = ['Sampling_Date', 'State'] + weather_features

train_enriched = train_wide.merge(weather_for_merge, on=['Sampling_Date', 'State'], how='left')
print(f"   ✓ Merged {len(weather_features)} weather features")

# 5. Add NDVI anomaly
print("\n5. Calculating NDVI anomaly...")
ndvi_stats = train_enriched.groupby('State')['Pre_GSHH_NDVI'].agg(['mean', 'std']).reset_index()
ndvi_stats.columns = ['State', 'ndvi_mean', 'ndvi_std']

train_enriched = train_enriched.merge(ndvi_stats, on='State')
train_enriched['ndvi_anomaly'] = (
    (train_enriched['Pre_GSHH_NDVI'] - train_enriched['ndvi_mean']) / train_enriched['ndvi_std']
)

all_features = weather_features + ['ndvi_anomaly']
print(f"   ✓ Total features: {len(all_features)}")

# 6. Save
print("\n6. Saving enriched dataset...")
output_file = 'competition/train_enriched.csv'
train_enriched.to_csv(output_file, index=False)

print(f"   ✓ Saved to {output_file}")
print(f"   Shape: {train_enriched.shape}")
print(f"   Missing values: {train_enriched[all_features].isnull().sum().sum()}")

print("\n" + "="*60)
print("✓ COMPLETE!")
print("="*60)
print(f"\nNew features ({len(all_features)}):")
for feat in all_features:
    print(f"  - {feat}")

print("\n✓ Ready for modeling with train_enriched.csv")
