"""
MODEL TRAINING SCRIPT
Purpose: Train and compare multiple models for traffic prediction

Models:
1. Facebook Prophet - Main model, handles seasonality
2. Ridge Regression - Regularized baseline
3. Linear Regression - Simple baseline

will output -> models/*.pkl (trained models), results/*.csv (metrics), results/*.png (plots)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
import pickle
import matplotlib.pyplot as plt
import sys

# Suppress Prophet warnings
import logging
logging.getLogger('prophet').setLevel(logging.ERROR)


def prepare_data(df: pd.DataFrame):
    """
    Prepare data for training
    
    Steps:
    1. Drop categorical columns 
    2. Split into train/test (80/20, chronological order)
    3. Separate features (X) and target (y)
    """
    # Drop categorical column (time_of_day)
    # Prophet and sklearn will need numeric features only
    df_numeric = df.drop(columns=['time_of_day'])

    # Need to make sure the target column exists
    if 'target' not in df_numeric.columns:
        print("'target' column not found. Re-run feature engineering again.")
        sys.exit(1)
    
    # Split chronologically
    # First 80% = training, last 20% = testing
    train_size = int(0.8 * len(df_numeric))
    train_data = df_numeric.iloc[:train_size]
    test_data = df_numeric.iloc[train_size:]
    
    print(f"Train size: {len(train_data):,} rows ({len(train_data)/len(df_numeric)*100:.1f}%)")
    print(f"Test size:  {len(test_data):,} rows ({len(test_data)/len(df_numeric)*100:.1f}%)")
    print(f"Train range: {train_data['timestamp'].min()} to {train_data['timestamp'].max()}")
    print(f"Test range:  {test_data['timestamp'].min()} to {test_data['timestamp'].max()}\n")
    
    return train_data, test_data


def train_prophet(train_data: pd.DataFrame, test_data: pd.DataFrame):
    """
    Train Facebook Prophet model for simple time series predictions
    
    Prophet specifics:
    - Needs columns named 'ds' (datestamp) and 'y' (target)
    - Automatically detects daily/weekly seasonality
    - Can add custom regressors (our features)
    """
    print("Training the Prophet model...")
    
    # Prepare data for Prophet 
    prophet_train = pd.DataFrame({
        'ds': train_data['timestamp'],
        'y': train_data['target']
    })
    
    # Initialize Prophet with seasonality settings
    model = Prophet(
        daily_seasonality=True,    # Detect daily patterns
        weekly_seasonality=True,   # Detect weekly patterns
        yearly_seasonality=False,  # Don't need yearly since my current dataset is only monthly 
        interval_width=0.95,       # 95% confidence intervals
        changepoint_prior_scale=0.05  # Allow  Flexibility in trend changes
    )
    
    # Add the created engineered features as regressors
    # This will help my Prophet understand context beyond just simple time 
    feature_columns = [
        'hour', 'day_of_week', 'is_weekend', 'is_business_hours',
        'traffic_5min_avg', 'traffic_15min_avg', 'traffic_30min_avg', 'traffic_60min_avg',
        'traffic_15min_std',
        'traffic_lag_1min', 'traffic_lag_5min', 'traffic_lag_10min',
        'traffic_lag_15min', 'traffic_lag_30min', 'traffic_lag_60min',
        'traffic_change_5min', 'same_hour_historical_avg', 'deviation_from_historical',
        'total_bytes', 'avg_bytes', 'success_rate'
    ]
    
    for col in feature_columns:
        model.add_regressor(col)
        prophet_train[col] = train_data[col].values
    
    # Train the actual model
    model.fit(prophet_train)
    
    # Make predictions on test set
    prophet_test = pd.DataFrame({
        'ds': test_data['timestamp']
    })
    
    for col in feature_columns:
        prophet_test[col] = test_data[col].values
    
    forecast = model.predict(prophet_test)
    predictions = forecast['yhat'].values  
    
    print("  Prophet training complete\n")
    
    return model, predictions


def train_ridge(train_data: pd.DataFrame, test_data: pd.DataFrame):
    """
    Train Ridge Regression model
    
    Ridge = Linear Regression + L2 regularization
    - Prevents overfitting by penalizing large coefficients
    - Alpha = regularization strength (10.0 is moderate)
    """
    print("Training Ridge Regression model...")
    
    # Features (all except timestamp and target)
    feature_columns = [col for col in train_data.columns 
                      if col not in ['timestamp', 'request_count', 'target']]
    
    X_train = train_data[feature_columns]
    y_train = train_data['target']
    X_test = test_data[feature_columns]
    
    # Train Ridge model
    model = Ridge(alpha=10.0)  # alpha=10.0 is the regularization strength
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    print("  Ridge training complete\n")
    
    return model, predictions


def train_linear(train_data: pd.DataFrame, test_data: pd.DataFrame):
    """
    Train Linear Regression model
    
    Simplest possible model - no regularization
    Good baseline to compare against
    """
    print("Training Linear Regression model...")
    
    # Features (all except timestamp and target)
    feature_columns = [col for col in train_data.columns 
                      if col not in ['timestamp', 'request_count', 'target']]
    
    X_train = train_data[feature_columns]
    y_train = train_data['target']
    X_test = test_data[feature_columns]
    
    # Train Linear model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    print("  Linear Regression training complete\n")
    
    return model, predictions


def evaluate_model(y_true, y_pred, model_name):
    """
    Calculate evaluation metrics
    
    Metrics:
    - MAE: Mean Absolute Error (average error in req/min)
    - RMSE: Root Mean Squared Error (penalizes large errors)
    - R²: Coefficient of determination (% variance explained)
    - MAPE: Mean Absolute Percentage Error (error as %)
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse) # this is the sqrt(mse) -> good to notice change in large values
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }


def plot_predictions(test_data, predictions_dict, output_file):
    """
    Create plot comparing actual vs predicted traffic
    
    Will Shows:
    - Actual traffic (black line)
    - Prophet predictions (blue)
    - Ridge predictions (green)
    - Linear predictions (red)
    """
    plt.figure(figsize=(15, 6))
    
    # Plot actual values
    plt.plot(test_data['timestamp'], test_data['target'], 
             label='Actual', color='black', linewidth=2, alpha=0.7)
    
    # Plot predictions from each model
    colors = {'Prophet': 'blue', 'Ridge': 'green', 'Linear': 'red'}
    for model_name, predictions in predictions_dict.items():
        plt.plot(test_data['timestamp'], predictions, 
                label=model_name, color=colors[model_name], 
                linewidth=1.5, alpha=0.6)
    
    plt.xlabel('Time')
    plt.ylabel('Request Count (per minute)')
    plt.title('Traffic Prediction: Actual vs Predicted')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


def train_models(input_file: Path, models_dir: Path, results_dir: Path):
    """
    Main training pipeline
    
    Steps:
    1. Load featured data
    2. Prepare train/test split
    3. Train all three models
    4. Evaluate and compare
    5. Save models and results
    """
    print(f"Reading from: {input_file}")
    print("Loading featured dat\n")
    
    # Load data
    df = pd.read_csv(input_file, parse_dates=['timestamp'])
    print(f"Total rows: {len(df):,}")
    print(f"Features: {len(df.columns) - 1}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Average traffic: {df['request_count'].mean():.1f} req/min\n")
    
    # Prepare the data
    print("="*60)
    print("PREPARING DATA")
    print("="*40)
    # train test split (80/20)
    train_data, test_data = prepare_data(df)
    
    # Train models
    print("="*60)
    print("TRAINING MODELS now")
    print("="*60)
    
    # model + predictions 
    prophet_model, prophet_pred = train_prophet(train_data, test_data)
    ridge_model, ridge_pred = train_ridge(train_data, test_data)
    linear_model, linear_pred = train_linear(train_data, test_data)
    
    # Evaluate models
    print("="*60)
    print("EVALUATING MODELS")
    print("="*60)
    
    y_true = test_data['target'].values
    
    prophet_metrics = evaluate_model(y_true, prophet_pred, 'Prophet')
    ridge_metrics = evaluate_model(y_true, ridge_pred, 'Ridge')
    linear_metrics = evaluate_model(y_true, linear_pred, 'Linear')
    
    # Create comparison DataFrame
    results_df = pd.DataFrame([prophet_metrics, ridge_metrics, linear_metrics])
    
    print("\nModel Comparison:")
    print(results_df.to_string(index=False))
    print()
    
    # Determine best model
    best_model = results_df.loc[results_df['MAE'].idxmin(), 'Model']
    print(f"Best model (lowest MAE): {best_model}")
    
    # Check if Prophet meets target
    avg_traffic = df['request_count'].mean()
    target_mae = 0.15 * avg_traffic
    prophet_mae = prophet_metrics['MAE']
    
    print(f"\nProphet performance:")
    print(f"  MAE: {prophet_mae:.2f} req/min")
    print(f"  Target: < {target_mae:.2f} req/min (15% of avg traffic)")
    
    if prophet_mae < target_mae:
        print(f"  Status: PASSED")
    else:
        print(f"  Status: FAIL")
    
    # Save models
    print(f"\n{'='*60}")
    print("SAVING MODELS")
    print(f"{'='*60}")
    
    models_dir.mkdir(exist_ok=True)
    
    with open(models_dir / 'prophet_model.pkl', 'wb') as f:
        pickle.dump(prophet_model, f)
    print(f"Saved: {models_dir / 'prophet_model.pkl'}")
    
    with open(models_dir / 'ridge_model.pkl', 'wb') as f:
        pickle.dump(ridge_model, f)
    print(f"Saved: {models_dir / 'ridge_model.pkl'}")
    
    with open(models_dir / 'linear_model.pkl', 'wb') as f:
        pickle.dump(linear_model, f)
    print(f"Saved: {models_dir / 'linear_model.pkl'}")
    
    # Save results
    results_dir.mkdir(exist_ok=True)
    
    results_df.to_csv(results_dir / 'model_comparison.csv', index=False)
    print(f"\nSaved: {results_dir / 'model_comparison.csv'}")
    
    # Create plot
    predictions_dict = {
        'Prophet': prophet_pred,
        'Ridge': ridge_pred,
        'Linear': linear_pred
    }
    
    plot_predictions(test_data, predictions_dict, results_dir / 'predictions_plot.png')
    print(f"Saved: {results_dir / 'predictions_plot.png'}")
    
    print("\nTraining complete")


if __name__ == "__main__":
    # Path to the data directory
    data_dir = Path(__file__).resolve().parents[1] / "data"
    models_dir = Path(__file__).resolve().parents[1] / "models"
    results_dir = Path(__file__).resolve().parents[1] / "results"
    
    # Input: featured traffic data
    input_file = data_dir / "featured_traffic.csv"
    
    # Validate input exists
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        print(f"Run Step 1.3 first: python scripts/03_engineer_features.py")
        sys.exit(1)
    
    # Run training
    train_models(input_file, models_dir, results_dir)