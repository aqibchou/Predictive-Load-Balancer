"""
Retrain Prophet with timestamps shifted to recent dates.

Problem: Original model trained on July 1995 data. Predicting for Dec 2025
means Prophet extrapolates 30 years, returning near-zero predictions.

Solution: Shift training data to recent dates (keeping time-of-day patterns).
"""

import pickle
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
import shutil
import os

DATA_PATH = 'C:/Users/moham/project-group-101/data/featured_traffic.csv'
OUTPUT_PATH = 'C:/Users/moham/project-group-101/models/prophet_model.pkl'
BACKUP_PATH = 'C:/Users/moham/project-group-101/models/prophet_model_backup.pkl'


def shift_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    original_start = df['timestamp'].min()
    target_start = datetime(2025, 11, 1, 0, 0, 0)
    
    time_shift = target_start - original_start
    
    df['timestamp'] = df['timestamp'] + time_shift
    
    print(f"Shifted timestamps by {time_shift.days} days")
    print(f"New date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ds'] = df['timestamp']
    df['y'] = df['request_count']
    
    df['hour'] = df['ds'].dt.hour
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
    
    df['traffic_5min_avg'] = df['y'].rolling(window=5, min_periods=1).mean()
    df['traffic_15min_avg'] = df['y'].rolling(window=15, min_periods=1).mean()
    df['traffic_30min_avg'] = df['y'].rolling(window=30, min_periods=1).mean()
    df['traffic_60min_avg'] = df['y'].rolling(window=60, min_periods=1).mean()
    
    df['traffic_15min_std'] = df['y'].rolling(window=15, min_periods=1).std().fillna(0)
    
    df['traffic_lag_1min'] = df['y'].shift(1).fillna(0)
    df['traffic_lag_5min'] = df['y'].shift(5).fillna(0)
    df['traffic_lag_10min'] = df['y'].shift(10).fillna(0)
    df['traffic_lag_15min'] = df['y'].shift(15).fillna(0)
    df['traffic_lag_30min'] = df['y'].shift(30).fillna(0)
    df['traffic_lag_60min'] = df['y'].shift(60).fillna(0)
    
    df['traffic_change_5min'] = df['y'] - df['traffic_lag_5min']
    
    hourly_avg = df.groupby('hour')['y'].transform('mean')
    df['same_hour_historical_avg'] = hourly_avg
    df['deviation_from_historical'] = np.where(
        hourly_avg > 0,
        (df['y'] - hourly_avg) / hourly_avg,
        0
    )
    
    if 'total_bytes' not in df.columns:
        df['total_bytes'] = 0
    if 'avg_bytes' not in df.columns:
        df['avg_bytes'] = 0
    if 'success_rate' not in df.columns:
        df['success_rate'] = 1.0
    
    return df


def main():
    print("Loading training data")
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Original date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total rows: {len(df)}")
    
    print("\nShifting timestamps to 2025")
    df = shift_timestamps(df)
    
    print("\nComputing features")
    df = add_features(df)
    
    feature_cols = [
        'hour', 'day_of_week', 'is_weekend', 'is_business_hours',
        'traffic_5min_avg', 'traffic_15min_avg', 'traffic_30min_avg', 'traffic_60min_avg',
        'traffic_15min_std',
        'traffic_lag_1min', 'traffic_lag_5min', 'traffic_lag_10min',
        'traffic_lag_15min', 'traffic_lag_30min', 'traffic_lag_60min',
        'traffic_change_5min',
        'same_hour_historical_avg', 'deviation_from_historical',
        'total_bytes', 'avg_bytes', 'success_rate'
    ]
    
    print(f"\nTraining Prophet with {len(feature_cols)} regressors")
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False
    )
    
    for col in feature_cols:
        model.add_regressor(col)
    
    train_cols = ['ds', 'y'] + feature_cols
    train_df = df[train_cols].dropna()
    
    print(f"Training on {len(train_df)} rows...")
    model.fit(train_df)
    
    if os.path.exists(OUTPUT_PATH):
        print(f"\nBacking up old model to {BACKUP_PATH}")
        shutil.copy(OUTPUT_PATH, BACKUP_PATH)
    
    print(f"Saving new model to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    print("\nTesting prediction")
    now = datetime.now()
    future = pd.DataFrame({
        'ds': [now + timedelta(minutes=i) for i in [1, 3, 5]]
    })
    
    future['hour'] = future['ds'].dt.hour
    future['day_of_week'] = future['ds'].dt.dayofweek
    future['is_weekend'] = (future['day_of_week'] >= 5).astype(int)
    future['is_business_hours'] = ((future['hour'] >= 9) & (future['hour'] <= 17)).astype(int)
    
    avg_traffic = df['y'].mean()
    future['traffic_5min_avg'] = avg_traffic
    future['traffic_15min_avg'] = avg_traffic
    future['traffic_30min_avg'] = avg_traffic
    future['traffic_60min_avg'] = avg_traffic
    future['traffic_15min_std'] = df['y'].std()
    future['traffic_lag_1min'] = avg_traffic
    future['traffic_lag_5min'] = avg_traffic
    future['traffic_lag_10min'] = avg_traffic
    future['traffic_lag_15min'] = avg_traffic
    future['traffic_lag_30min'] = avg_traffic
    future['traffic_lag_60min'] = avg_traffic
    future['traffic_change_5min'] = 0
    future['same_hour_historical_avg'] = avg_traffic
    future['deviation_from_historical'] = 0
    future['total_bytes'] = 0
    future['avg_bytes'] = 0
    future['success_rate'] = 1.0
    
    forecast = model.predict(future)
    
    print("\nSample predictions (with average traffic as features):")
    for i, row in forecast.iterrows():
        mins = [1, 3, 5][i]
        print(f"  +{mins} min: {row['yhat']:.1f} req/min (range: {row['yhat_lower']:.1f} - {row['yhat_upper']:.1f})")
    
    print("\nDone")
    print("  copy models\\prophet_model.pkl load_balancer\\models\\")
    print("  docker-compose down")
    print("  docker-compose build load_balancer")
    print("  docker-compose up")


if __name__ == "__main__":
    main()