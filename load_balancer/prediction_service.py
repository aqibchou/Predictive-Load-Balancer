import asyncio
import pickle
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import pandas as pd
import numpy as np

from database import get_recent_traffic
from datetime import datetime
from prophet import Prophet  
import pandas as pd


# Model path is taken from env or defaults to inside container
MODEL_PATH = os.getenv("PROPHET_MODEL_PATH", "/app/models/prophet_model.pkl")

# Prediction loop interval
PREDICTION_INTERVAL_SECONDS = 60
HISTORY_WINDOW_MINUTES = 60


@dataclass  # type hints + automatic constructor
class TrafficPrediction:
    timestamp: datetime
    predicted_1min: float
    predicted_3min: float
    predicted_5min: float
    lower_bound_5min: float
    upper_bound_5min: float
    uncertainty: float
    model_loaded: bool = True
    features_available: bool = True  # feature logic

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "predicted_1min": round(self.predicted_1min, 2),
            "predicted_3min": round(self.predicted_3min, 2),
            "predicted_5min": round(self.predicted_5min, 2),
            "lower_bound_5min": round(self.lower_bound_5min, 2),
            "upper_bound_5min": round(self.upper_bound_5min, 2),
            "uncertainty": round(self.uncertainty, 2),
            "model_loaded": self.model_loaded,
            "features_available": self.features_available,
        }


class PredictionService:
    def __init__(self):
        self.model = None
        self.current_prediction: Optional[TrafficPrediction] = None
        self.prediction_history: list = []
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        self._aggregate_task: Optional[asyncio.Task] = None

    # Loads Prophet model at startup
    '''def load_model(self) -> bool:
        try:
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            print(f"Prophet model loaded from {MODEL_PATH}")
            return True

        except FileNotFoundError:
            print(f"Prophet model not found at {MODEL_PATH}")
            return False

        except Exception as e:
            print(f"Error loading Prophet model: {e}")
            return False'''
    # training model at runtime
    def load_model(self) -> bool:
        print("Using real-time Prophet retraining (no pre-trained model needed)")
        return True  

    # Start background prediction + aggregate loop
    async def start(self):
        if self.is_running:
            return

        self.is_running = True
        self._task = asyncio.create_task(self._prediction_loop())
        self._aggregate_task = asyncio.create_task(self._aggregate_loop())
        print("Prediction service started")

    # Stop background tasks cleanly
    async def stop(self):
        self.is_running = False
        for task in [self._task, self._aggregate_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        print("Prediction service stopped")

    # Runs every minute to update database aggregates
    async def _aggregate_loop(self):
        from database import update_minute_aggregate

        while self.is_running:
            try:
                await update_minute_aggregate()
            except Exception as e:
                print(f"Aggregate update error: {e}")

            await asyncio.sleep(60)

    # Background prediction loop running every 60 seconds
    async def _prediction_loop(self):
        await asyncio.sleep(5)  # initial delay before first prediction

        while self.is_running:
            try:
                await self._run_prediction()
            except Exception as e:
                print(f"Prediction error: {e}")

            await asyncio.sleep(PREDICTION_INTERVAL_SECONDS)

    # Compute  feature set for Prophet regression
    def _compute_features(self, ds: datetime, traffic_history: List[Dict]) -> Dict[str, float]:
        features = {}

        # Base calendar features
        features["hour"] = ds.hour
        features["day_of_week"] = ds.weekday()
        features["is_weekend"] = 1 if ds.weekday() >= 5 else 0
        features["is_business_hours"] = 1 if 9 <= ds.hour <= 17 else 0

        # No historical data
        if not traffic_history:
            features.update({
                "traffic_5min_avg": 0,
                "traffic_15min_avg": 0,
                "traffic_30min_avg": 0,
                "traffic_60min_avg": 0,
                "traffic_15min_std": 0,
                "traffic_lag_1min": 0,
                "traffic_lag_5min": 0,
                "traffic_lag_10min": 0,
                "traffic_lag_15min": 0,
                "traffic_lag_30min": 0,
                "traffic_lag_60min": 0,
                "traffic_change_5min": 0,
                "same_hour_historical_avg": 0,
                "deviation_from_historical": 0,
                "total_bytes": 0,
                "avg_bytes": 0,
                "success_rate": 1.0,
            })
            return features

        # Build dataframe
        df = pd.DataFrame(traffic_history)
        df["minute_bucket"] = pd.to_datetime(df["minute_bucket"])
        df = df.sort_values("minute_bucket")

        counts = df["request_count"].values

        # Rolling averages
        features["traffic_5min_avg"] = float(np.mean(counts[-5:])) if len(counts) >= 5 else float(np.mean(counts))
        features["traffic_15min_avg"] = float(np.mean(counts[-15:])) if len(counts) >= 15 else float(np.mean(counts))
        features["traffic_30min_avg"] = float(np.mean(counts[-30:])) if len(counts) >= 30 else float(np.mean(counts))
        features["traffic_60min_avg"] = float(np.mean(counts))

        # Variation / std
        if len(counts) >= 15:
            features["traffic_15min_std"] = float(np.std(counts[-15:]))
        else:
            features["traffic_15min_std"] = float(np.std(counts)) if len(counts) > 1 else 0

        # Lags
        features["traffic_lag_1min"] = float(counts[-1]) if len(counts) >= 1 else 0
        features["traffic_lag_5min"] = float(counts[-5]) if len(counts) >= 5 else 0
        features["traffic_lag_10min"] = float(counts[-10]) if len(counts) >= 10 else 0
        features["traffic_lag_15min"] = float(counts[-15]) if len(counts) >= 15 else 0
        features["traffic_lag_30min"] = float(counts[-30]) if len(counts) >= 30 else 0
        features["traffic_lag_60min"] = float(counts[-60]) if len(counts) >= 60 else 0

        # Short-term spike indicator
        if len(counts) >= 5:
            features["traffic_change_5min"] = float(counts[-1] - counts[-5])
        else:
            features["traffic_change_5min"] = 0

        # Same hour historical average
        same_hour = df[df["minute_bucket"].dt.hour == ds.hour]
        features["same_hour_historical_avg"] = (
            float(same_hour["request_count"].mean()) if len(same_hour) > 0 else features["traffic_60min_avg"]
        )

        # Deviation from typical hour pattern
        if features["same_hour_historical_avg"] > 0:
            features["deviation_from_historical"] = (
                (features["traffic_lag_1min"] - features["same_hour_historical_avg"])
                / features["same_hour_historical_avg"]
            )
        else:
            features["deviation_from_historical"] = 0

        # Byte-level features (optional)
        features["total_bytes"] = float(df["total_bytes"].sum()) if "total_bytes" in df.columns else 0
        features["avg_bytes"] = float(df["total_bytes"].mean()) if "total_bytes" in df.columns else 0

        # Success rate (if available)
        if "success_count" in df.columns and "total_count" in df.columns:
            total = df["total_count"].sum()
            success = df["success_count"].sum()
            features["success_rate"] = float(success / total) if total > 0 else 1.0
        else:
            features["success_rate"] = 1.0

        return features
    '''
    # Actual prediction execution
    async def _run_prediction(self):
        from database import get_recent_traffic

        # If model failed to load, return zeros
        if self.model is None:
            self.current_prediction = TrafficPrediction(
                timestamp=datetime.now(),
                predicted_1min=0,
                predicted_3min=0,
                predicted_5min=0,
                lower_bound_5min=0,
                upper_bound_5min=0,
                uncertainty=0,
                model_loaded=False,
                features_available=False,
            )
            return

        # Pull recent traffic history
        traffic_history = await get_recent_traffic(minutes=60)
        features_available = len(traffic_history) > 0

        now = datetime.now()

        # Predict at +1, +3, +5 min
        future_times = [
            now + timedelta(minutes=1),
            now + timedelta(minutes=3),
            now + timedelta(minutes=5),
        ]

        rows = []
        for ds in future_times:
            row = {"ds": ds}
            row.update(self._compute_features(ds, traffic_history))
            rows.append(row)

        future_df = pd.DataFrame(rows)

        # Prophet prediction
        forecast = self.model.predict(future_df)

        # Extract values safely (non-negative)
        pred_1min = max(0, forecast.iloc[0]["yhat"])
        pred_3min = max(0, forecast.iloc[1]["yhat"])
        pred_5min = max(0, forecast.iloc[2]["yhat"])
        lower_5min = max(0, forecast.iloc[2]["yhat_lower"])
        upper_5min = max(0, forecast.iloc[2]["yhat_upper"])

        # Wrap results in dataclass
        self.current_prediction = TrafficPrediction(
            timestamp=now,
            predicted_1min=pred_1min,
            predicted_3min=pred_3min,
            predicted_5min=pred_5min,
            lower_bound_5min=lower_5min,
            upper_bound_5min=upper_5min,
            uncertainty=upper_5min - lower_5min,
            features_available=features_available,
        )

        # Keep last 60 predictions
        self.prediction_history.append(self.current_prediction)
        if len(self.prediction_history) > 60:
            self.prediction_history = self.prediction_history[-60:]

        print(f"Prediction: {pred_5min:.1f} req/min in 5min (±{self.current_prediction.uncertainty:.1f})")
    '''
    async def _run_prediction(self):

        # Pull recent traffic history
        traffic_history = await get_recent_traffic(minutes=60)
        
        # use defaults
        if not traffic_history or len(traffic_history) < 3:
            now = datetime.now()
            self.current_prediction = TrafficPrediction(
                timestamp=now,
                predicted_1min=25.0,
                predicted_3min=25.0,
                predicted_5min=25.0,
                lower_bound_5min=20.0,
                upper_bound_5min=30.0,
                uncertainty=10.0,
                model_loaded=True,
                features_available=False,
            )
            print("Not enough traffic data yet, using defaults")
            return

        try:
            # Build DataFrame from traffic history
            df_data = []
            for row in traffic_history:
                # Find timestamp column
                timestamp_value = None
                for key in row.keys():
                    if 'time' in key.lower():
                        timestamp_value = row[key]
                        break
                
                if timestamp_value is None:
                    continue
                    
                # Convert to datetime if it's a float (unix timestamp)
                if isinstance(timestamp_value, (int, float)):
                    ts = datetime.fromtimestamp(timestamp_value)
                else:
                    # Already a datetime object
                    ts = timestamp_value
                    if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
                        ts = ts.replace(tzinfo=None)
                
                df_data.append({
                    'ds': ts,
                    'y': float(row['request_count'])
                })
            
            if len(df_data) < 3:
                print("Not enough valid data points for Prophet")
                raise ValueError("Insufficient data")
            
            df = pd.DataFrame(df_data)
            
            # Train Prophet model
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=False,
                interval_width=0.8
            )
            model.fit(df)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=5, freq='min')
            forecast = model.predict(future)
            
            # Extract predictions
            pred_1min = forecast['yhat'].iloc[-5]
            pred_3min = forecast['yhat'].iloc[-3]
            pred_5min = forecast['yhat'].iloc[-1]
            lower_5min = forecast['yhat_lower'].iloc[-1]
            upper_5min = forecast['yhat_upper'].iloc[-1]
            uncertainty = (upper_5min - lower_5min) / 2
            
            now = datetime.now()
            self.current_prediction = TrafficPrediction(
                timestamp=now,
                predicted_1min=max(0, pred_1min),
                predicted_3min=max(0, pred_3min),
                predicted_5min=max(0, pred_5min),
                lower_bound_5min=max(0, lower_5min),
                upper_bound_5min=max(0, upper_5min),
                uncertainty=max(0, uncertainty),
                features_available=True,
            )
            
            print(f"Prophet prediction: {pred_5min:.1f} req/min in 5min (±{uncertainty:.1f})")
            
        except Exception as e:
            print(f"Prophet fit error: {e}")
            # Fallback to defaults
            now = datetime.now()
            self.current_prediction = TrafficPrediction(
                timestamp=now,
                predicted_1min=25.0,
                predicted_3min=25.0,
                predicted_5min=25.0,
                lower_bound_5min=20.0,
                upper_bound_5min=30.0,
                uncertainty=10.0,
                model_loaded=True,
                features_available=False,
            )
            return
        
        # Keep last 60 predictions
        self.prediction_history.append(self.current_prediction)
        if len(self.prediction_history) > 60:
            self.prediction_history.pop(0)
            
    # Basic accessors
    def get_current_prediction(self) -> Optional[TrafficPrediction]:
        return self.current_prediction

    def get_prediction_for_scaler(self) -> Dict[str, float]:
        if self.current_prediction is None:
            return {
                "predicted_load_5min": 0,
                "prediction_uncertainty": 100,
                "prediction_available": False,
            }

        return {
            "predicted_load_5min": self.current_prediction.predicted_5min,
            "prediction_uncertainty": self.current_prediction.uncertainty,
            "prediction_available": self.current_prediction.model_loaded,
        }


# Global singleton instance
prediction_service = PredictionService()
