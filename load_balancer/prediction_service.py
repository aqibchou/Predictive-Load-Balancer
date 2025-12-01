import asyncio
import pickle
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import pandas as pd

MODEL_PATH = os.getenv("PROPHET_MODEL_PATH", "/app/models/prophet_model.pkl")
PREDICTION_INTERVAL_SECONDS = 60
HISTORY_WINDOW_MINUTES = 60


@dataclass # type hints, automatic constructors 
class TrafficPrediction:
    timestamp: datetime
    predicted_1min: float
    predicted_3min: float
    predicted_5min: float
    lower_bound_5min: float
    upper_bound_5min: float
    uncertainty: float
    model_loaded: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "predicted_1min": round(self.predicted_1min, 2),
            "predicted_3min": round(self.predicted_3min, 2),
            "predicted_5min": round(self.predicted_5min, 2),
            "lower_bound_5min": round(self.lower_bound_5min, 2),
            "upper_bound_5min": round(self.upper_bound_5min, 2),
            "uncertainty": round(self.uncertainty, 2),
            "model_loaded": self.model_loaded
        }


class PredictionService:
    def __init__(self):
        self.model = None
        self.current_prediction: Optional[TrafficPrediction] = None
        self.prediction_history: list = []
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
    
    # Loads prophet model(trained) at  startup
    def load_model(self) -> bool:
        try:
            with open(MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Prophet model loaded from {MODEL_PATH}")
            return True
        # Should not happen - just to be safe 
        except FileNotFoundError:
            print(f"Prophet model not found at {MODEL_PATH}")
            return False
        except Exception as e:
            print(f"Error loading Prophet model: {e}")
            return False
    
    async def start(self):
        if self.is_running:
            return
        
        self.is_running = True
        self._task = asyncio.create_task(self._prediction_loop())
        print("Prediction service started")
    
    async def stop(self):
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        print("Prediction service stopped")
    

    # Runs every 60 seconds and is the backround service
    async def _prediction_loop(self):
        while self.is_running:
            try:
                await self._run_prediction()
            except Exception as e:
                print(f"Prediction error: {e}")
            
            await asyncio.sleep(PREDICTION_INTERVAL_SECONDS)
    
    # Actual prediction logic is here 
    async def _run_prediction(self):
        if self.model is None:
            # Dataframe stores future timesteams 1,3,5
            self.current_prediction = TrafficPrediction(
                timestamp=datetime.now(),
                predicted_1min=0,
                predicted_3min=0,
                predicted_5min=0,
                lower_bound_5min=0,
                upper_bound_5min=0,
                uncertainty=0,
                model_loaded=False
            )
            return
        
        now = datetime.now()
        
        # prophet accepts ds format 
        future_dates = pd.DataFrame({
            'ds': [
                now + timedelta(minutes=1),
                now + timedelta(minutes=3),
                now + timedelta(minutes=5)
            ]
        })

        # return yhat as well as the upper and lower bounds 
        forecast = self.model.predict(future_dates)
        
        # Prophet can give negative vals sometimes, so keep the max 
        pred_1min = max(0, forecast.iloc[0]['yhat'])
        pred_3min = max(0, forecast.iloc[1]['yhat'])
        pred_5min = max(0, forecast.iloc[2]['yhat'])
        lower_5min = max(0, forecast.iloc[2]['yhat_lower'])
        upper_5min = max(0, forecast.iloc[2]['yhat_upper'])
        
        # Wrap in traffic prediction dataclass 
        self.current_prediction = TrafficPrediction(
            timestamp=now,
            predicted_1min=pred_1min,
            predicted_3min=pred_3min,
            predicted_5min=pred_5min,
            lower_bound_5min=lower_5min,
            upper_bound_5min=upper_5min,
            uncertainty=upper_5min - lower_5min
        )
        
        self.prediction_history.append(self.current_prediction)
        if len(self.prediction_history) > 60:
            self.prediction_history = self.prediction_history[-60:]
        
        print(f"Prediction: {pred_5min:.1f} req/min in 5min (±{self.current_prediction.uncertainty:.1f})")
    
    def get_current_prediction(self) -> Optional[TrafficPrediction]:
        return self.current_prediction
    
    def get_prediction_for_scaler(self) -> Dict[str, float]:
        if self.current_prediction is None:
            return {
                "predicted_load_5min": 0,
                "prediction_uncertainty": 100,
                "prediction_available": False
            }
        
        return {
            "predicted_load_5min": self.current_prediction.predicted_5min,
            "prediction_uncertainty": self.current_prediction.uncertainty,
            "prediction_available": self.current_prediction.model_loaded
        }


prediction_service = PredictionService()