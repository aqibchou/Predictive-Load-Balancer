"""
FEATURE ENGINEERING SCRIPT
Purpose: Transform request-level data into minute-level traffic with predictive features

Input:  data/cleaned_logs.csv (1.88M rows, one per request)
Output: data/featured_traffic.csv (~60k rows, one per minute)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, time
import sys

# Business hours definition to be used
BUSINESS_START = 9  # 9am
BUSINESS_END = 17   # 5pm


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from timestamp
    
    Features:
    - hour: 0-23 (captures daily patterns like lunch rush)
    - day_of_week: 0=Mon, 6=Sun (captures weekly patterns)
    - is_weekend: Binary indicator for Sat/Sun
    - is_business_hours: Binary for 9am-5pm EST defined above
    - time_of_day: Categorical (night/morning/afternoon/evening)
    """
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_business_hours'] = df['hour'].between(BUSINESS_START, BUSINESS_END).astype(int)
    
    # Group hours into time periods
    df['time_of_day'] = pd.cut(
        df['hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['night', 'morning', 'afternoon', 'evening'],
        include_lowest=True
    )
    
    return df


def aggregate_to_minute_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform request-level data to minute-level aggregates
    
    Aggregations:
    - request_count: Number of requests per minute (TARGET VARIABLE)
    - total_bytes: Sum of bytes transferred
    - avg_bytes: Average response size
    - success_rate: Percentage of HTTP 200 responses
    """
    # Round timestamps to minute boundary
    df['minute'] = df['timestamp'].dt.floor('min')
    
    # Group by minute and calculate aggregates
    minute_data = df.groupby('minute').agg({
        'timestamp': 'count',    # Count requests
        'bytes': ['sum', 'mean'], # Total and average bytes
        'status': lambda x: (x == 200).sum()  # Count successes
    }).reset_index()
    
    # Flatten column names
    minute_data.columns = ['timestamp', 'request_count', 'total_bytes', 'avg_bytes', 'success_count']
    
    # Calculate success rate
    minute_data['success_rate'] = minute_data['success_count'] / minute_data['request_count']
    minute_data = minute_data.drop(columns=['success_count'])
    
    return minute_data


def create_rolling_features(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    """
    Create rolling window features (moving averages and std dev)
    
    Different window sizes capture different timescales:
    - 5min: Immediate trend
    - 15min: Short-term trend
    - 30min: Medium-term trend
    - 60min: Long-term trend
    
    Also calculates 15min std dev for volatility measure
    """
    for window in windows:
        df[f'traffic_{window}min_avg'] = df['request_count'].rolling(
            window=window,
            min_periods=1
        ).mean()
        
        # Add standard deviation for 15min window
        if window == 15:
            df[f'traffic_{window}min_std'] = df['request_count'].rolling(
                window=window,
                min_periods=1
            ).std()
    
    return df


def create_lag_features(df: pd.DataFrame, lags: list) -> pd.DataFrame:
    """
    Create lag features (what was traffic N minutes ago?)
    
    Lags capture momentum and recent history:
    - 1min: Immediate past (strongest predictor)
    - 5min: Very recent
    - 15min: Short-term memory
    - 60min: Longer-term patterns
    """
    for lag in lags:
        df[f'traffic_lag_{lag}min'] = df['request_count'].shift(lag)
    
    return df


def create_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create statistical features (rate of change, historical context)
    
    Features:
    - traffic_change_5min: How fast traffic is changing
    - same_hour_historical_avg: Typical traffic for this hour/day combo
    - deviation_from_historical: Is current traffic unusual?
    """
    # Rate of change over last 5 minutes
    df['traffic_change_5min'] = (df['request_count'] - df['traffic_lag_5min']) / 5
    
    # Historical average for same hour and day of week
    historical_avg = df.groupby(['hour', 'day_of_week'])['request_count'].transform('mean')
    df['same_hour_historical_avg'] = historical_avg
    
    # Deviation from historical average
    df['deviation_from_historical'] = (
        df['request_count'] - df['same_hour_historical_avg']
    ) / df['same_hour_historical_avg']
    
    # Handle division by zero
    df['deviation_from_historical'] = df['deviation_from_historical'].replace([np.inf, -np.inf], 0)
    
    return df


def engineer_features(input_file: Path, output_file: Path):
    """
    Main feature engineering pipeline
    
    Pipeline:
    1. Load cleaned data
    2. Aggregate to minute-level
    3. Create time features
    4. Create rolling windows
    5. Create lag features
    6. Create statistical features
    7. Handle missing values
    8. Save featured data
    """
    print(f"Reading from: {input_file}")
    print("Loading cleaned data...")
    
    # Load cleaned data
    df = pd.read_csv(input_file, parse_dates=['timestamp'])
    initial_rows = len(df)
    print(f"Initial rows (request-level): {initial_rows:,}\n")
    
    # STEP 1: Aggregate to minute-level
    print("Step 1: Aggregating to minute-level.")
    df_minute = aggregate_to_minute_level(df)
    print(f"  Minute-level rows: {len(df_minute):,}")
    print(f"  Time range: {df_minute['timestamp'].min()} to {df_minute['timestamp'].max()}")
    print(f"  Average requests per minute: {df_minute['request_count'].mean():.1f}\n")
    
    # STEP 2: Create time features
    print("Step 2: Creating time features...")
    df_minute = create_time_features(df_minute)
    print("  Added: hour, day_of_week, is_weekend, is_business_hours, time_of_day\n")
    
    # STEP 3: Create rolling window features
    print("Step 3: Creating rolling window features...")
    windows = [5, 15, 30, 60]
    df_minute = create_rolling_features(df_minute, windows)
    print(f"  Added: {len(windows)} rolling averages (5/15/30/60 min)")
    print("  Added: 1 rolling std dev (15 min)\n")
    
    # STEP 4: Create lag features
    print("Step 4: Creating lag features...")
    lags = [1, 5, 10, 15, 30, 60]
    df_minute = create_lag_features(df_minute, lags)
    print(f"  Added: {len(lags)} lag features (1/5/10/15/30/60 min ago)\n")
    
    # STEP 5: Create statistical features
    print("Step 5: Creating statistical features...")
    df_minute = create_statistical_features(df_minute)
    print("  Added: traffic_change_5min, same_hour_historical_avg, deviation_from_historical\n")

    # STEP 5.5: Extra step needed to fix target variable
    print("Step 5.5: Creating future target variable...")
    # Shift request_count backward by 5 to get traffic 5 minutes ahead
    df_minute['target'] = df_minute['request_count'].shift(-5)
    print("  Added: target (request_count 5 minutes in future)\n")
    
    # STEP 6: Handle missing values
    print("Step 6: Handling missing values...")
    before_dropna = len(df_minute)
    df_minute = df_minute.dropna()
    after_dropna = len(df_minute)
    dropped = before_dropna - after_dropna
    print(f"  Dropped {dropped:,} rows with missing values")
    print(f"  Remaining: {after_dropna:,} rows\n")
    
    # Final statistics
    print("="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"Initial rows (request-level):  {initial_rows:,}")
    print(f"Minute-level rows:             {before_dropna:,}")
    print(f"After handling NaN:            {after_dropna:,}")
    print(f"Total features created:        {len(df_minute.columns) - 1}")
    print("\nFeature categories:")
    print("  Time features:       5")
    print("  Aggregated metrics:  4")
    print("  Rolling windows:     5")
    print("  Lag features:        6")
    print("  Statistical:         3")
    
    # Save featured data
    print(f"\nSaving to: {output_file}")
    df_minute.to_csv(output_file, index=False)
    print("Saved successfully")
    
    return df_minute


if __name__ == "__main__":
    # Path to the data directory
    data_dir = Path(__file__).resolve().parents[1] / "data"
    
    # Input and output paths
    input_file = data_dir / "cleaned_logs.csv"
    output_file = data_dir / "featured_traffic.csv"
    
    # Validate input exists
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        print("Run Step 1.2 first: python scripts/02_clean_data.py")
        sys.exit(1)
    
    # Run feature engineering
    df_featured = engineer_features(input_file, output_file)