import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler

from database import get_recent_traffic
from bistacking import BiStackingEnsemble
from mask_store import load_mask

logger = logging.getLogger(__name__)

PREDICTION_INTERVAL_SECONDS = 60
HISTORY_WINDOW_MINUTES      = 120  # lag_60 needs ≥60 rows after dropna

REDIS_URL          = os.getenv("REDIS_URL", "redis://redis:6379")
MASK_FALLBACK_PATH = Path(os.getenv("MASK_FALLBACK_PATH", "/app/models/feature_mask.json"))
MIN_ROWS_FOR_FIT   = 30

# Order is a hard contract with async_worker.py — mask indices index into this list
FEATURE_COLS: List[str] = [
    "lag_1",  "lag_2",  "lag_3",  "lag_5",
    "lag_10", "lag_15", "lag_30", "lag_60",
    "roll_mean_5",  "roll_std_5",  "roll_max_5",  "roll_min_5",
    "roll_mean_10", "roll_std_10", "roll_max_10", "roll_min_10",
    "roll_mean_15", "roll_std_15", "roll_max_15", "roll_min_15",
    "roll_mean_30", "roll_std_30", "roll_max_30", "roll_min_30",
    "roll_mean_60", "roll_std_60", "roll_max_60", "roll_min_60",
    "hour", "dayofweek", "minute_mod", "is_weekend",
    "ema_5", "ema_15", "ema_30",
    "diff_1", "diff_5", "diff_60",
]


@dataclass
class TrafficPrediction:
    timestamp:        datetime
    predicted_1min:   float
    predicted_3min:   float
    predicted_5min:   float
    lower_bound_5min: float
    upper_bound_5min: float
    uncertainty:      float
    model_loaded:     bool = True
    features_available: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp":        self.timestamp.isoformat(),
            "predicted_1min":   round(self.predicted_1min,   2),
            "predicted_3min":   round(self.predicted_3min,   2),
            "predicted_5min":   round(self.predicted_5min,   2),
            "lower_bound_5min": round(self.lower_bound_5min, 2),
            "upper_bound_5min": round(self.upper_bound_5min, 2),
            "uncertainty":      round(self.uncertainty,      2),
            "model_loaded":     self.model_loaded,
            "features_available": self.features_available,
        }


class PredictionService:
    def __init__(self):
        self.model:   Optional[BiStackingEnsemble] = None
        self._scaler: Optional[StandardScaler]     = None
        self._mask:              Optional[np.ndarray] = None
        self._mask_feature_names: List[str]           = []
        self._mask_loaded:        bool                = False
        self.current_prediction:  Optional[TrafficPrediction] = None
        self.prediction_history:  List[TrafficPrediction]     = []
        self.is_running           = False
        self._task:               Optional[asyncio.Task] = None
        self._aggregate_task:     Optional[asyncio.Task] = None
        self._last_traffic_stats: dict = {}

    def load_model(self) -> bool:
        try:
            mask, names = load_mask(redis_url=REDIS_URL, fallback_path=MASK_FALLBACK_PATH)
            if len(mask) != len(FEATURE_COLS):
                logger.warning("Mask length %d != expected %d — using all features.", len(mask), len(FEATURE_COLS))
                self._mask = np.ones(len(FEATURE_COLS), dtype=bool)
            else:
                self._mask = mask
            self._mask_feature_names = names
            self._mask_loaded = True
            logger.info("Feature mask loaded (%d/%d features)", int(self._mask.sum()), len(FEATURE_COLS))
        except FileNotFoundError:
            logger.warning("No feature mask found — using all 38 features until async_worker runs.")
            self._mask        = np.ones(len(FEATURE_COLS), dtype=bool)
            self._mask_loaded = False
        return True

    async def start(self):
        if self.is_running:
            return
        self.is_running      = True
        self._task           = asyncio.create_task(self._prediction_loop())
        self._aggregate_task = asyncio.create_task(self._aggregate_loop())
        logger.info("Prediction service started")

    async def stop(self):
        self.is_running = False
        for task in [self._task, self._aggregate_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def _aggregate_loop(self):
        from database import update_minute_aggregate
        while self.is_running:
            try:
                await update_minute_aggregate()
            except Exception as e:
                logger.error("Aggregate error: %s", e)
            await asyncio.sleep(60)

    async def _prediction_loop(self):
        await asyncio.sleep(5)
        while self.is_running:
            try:
                await self._run_prediction()
            except Exception as e:
                logger.error("Prediction loop error: %s", e, exc_info=True)
            await asyncio.sleep(PREDICTION_INTERVAL_SECONDS)

    def _engineer_features(self, traffic_history: List[Dict]) -> Optional[pd.DataFrame]:
        df = pd.DataFrame(traffic_history)
        if "minute_bucket" in df.columns:
            df = df.rename(columns={"minute_bucket": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["request_count"] = pd.to_numeric(df["request_count"], errors="coerce").fillna(0)

        rc   = df["request_count"].astype(float)
        ts   = df["timestamp"]
        base = rc.shift(1)

        for lag in [1, 2, 3, 5, 10, 15, 30, 60]:
            df[f"lag_{lag}"] = rc.shift(lag)
        for w in [5, 10, 15, 30, 60]:
            df[f"roll_mean_{w}"] = base.rolling(w).mean()
            df[f"roll_std_{w}"]  = base.rolling(w).std()
            df[f"roll_max_{w}"]  = base.rolling(w).max()
            df[f"roll_min_{w}"]  = base.rolling(w).min()

        df["hour"]       = ts.dt.hour
        df["dayofweek"]  = ts.dt.dayofweek
        df["minute_mod"] = ts.dt.minute
        df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

        for span in [5, 15, 30]:
            df[f"ema_{span}"] = rc.ewm(span=span, adjust=False).mean().shift(1)

        df["diff_1"]  = rc.diff(1)
        df["diff_5"]  = rc.diff(5)
        df["diff_60"] = rc.diff(60)

        df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
        return df[FEATURE_COLS + ["request_count", "timestamp"]] if len(df) >= MIN_ROWS_FOR_FIT else None

    def _fit_and_predict_bistacking(self, feat_df: pd.DataFrame) -> Tuple[float, float, float, float, float]:
        rc     = feat_df["request_count"].values.astype(np.float32)
        X_full = feat_df[FEATURE_COLS].values.astype(np.float32)
        mask   = self._mask if self._mask is not None else np.ones(len(FEATURE_COLS), dtype=bool)
        X_masked = X_full[:, mask]

        X_train, y_train = X_masked[:-1], rc[1:]
        if len(X_train) < MIN_ROWS_FOR_FIT:
            raise ValueError(f"Only {len(X_train)} aligned rows — need ≥ {MIN_ROWS_FOR_FIT}")

        scaler    = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        self._scaler = scaler

        ensemble = BiStackingEnsemble(n_folds=5)
        ensemble.fit(X_train_s, y_train)
        self.model = ensemble

        X_last_s  = scaler.transform(X_masked[[-1]])
        pred_1min = max(0.0, float(ensemble.predict(X_last_s)[0]))

        l0_values = np.array([v[0] for v in ensemble.predict_proba(X_last_s).values()])
        l0_std    = float(np.std(l0_values))

        last_diff = float(X_full[-1, FEATURE_COLS.index("diff_1")])
        pred_3min = max(0.0, pred_1min + 2 * last_diff * 0.5)
        pred_5min = max(0.0, pred_1min + 4 * last_diff * 0.5)
        lower_5min = max(0.0, pred_5min - 1.28 * l0_std)
        upper_5min = max(0.0, pred_5min + 1.28 * l0_std)

        return pred_1min, pred_3min, pred_5min, lower_5min, upper_5min

    def _default_prediction(self, model_loaded: bool, features_available: bool) -> TrafficPrediction:
        return TrafficPrediction(
            timestamp=datetime.now(),
            predicted_1min=25.0, predicted_3min=25.0, predicted_5min=25.0,
            lower_bound_5min=20.0, upper_bound_5min=30.0,
            uncertainty=10.0, model_loaded=model_loaded, features_available=features_available,
        )

    async def _run_prediction(self):
        traffic_history = await get_recent_traffic(minutes=HISTORY_WINDOW_MINUTES)
        if not traffic_history or len(traffic_history) < 62:
            self.current_prediction = self._default_prediction(False, False)
            logger.info("Not enough data (%d rows), using defaults", len(traffic_history) if traffic_history else 0)
            return

        try:
            feat_df = self._engineer_features(traffic_history)
            if feat_df is None:
                raise ValueError("Insufficient rows after feature engineering")

            counts    = feat_df["request_count"].values
            curr_mean = float(np.mean(counts))
            curr_std  = float(np.std(counts))

            if self._last_traffic_stats and self.current_prediction is not None:
                last_mean = self._last_traffic_stats.get("mean", 0)
                last_std  = self._last_traffic_stats.get("std", 0)
                if last_mean > 0:
                    if (abs(curr_mean - last_mean) / last_mean < 0.05 and
                            abs(curr_std - last_std) / max(last_std, 1.0) < 0.05):
                        logger.info("Traffic stable, skipping refit")
                        return

            if not self._mask_loaded:
                try:
                    mask, names = load_mask(REDIS_URL, MASK_FALLBACK_PATH)
                    if len(mask) == len(FEATURE_COLS):
                        self._mask = mask
                        self._mask_feature_names = names
                        self._mask_loaded = True
                except FileNotFoundError:
                    pass

            pred_1min, pred_3min, pred_5min, lower_5min, upper_5min = await asyncio.to_thread(
                self._fit_and_predict_bistacking, feat_df
            )

            self._last_traffic_stats = {"mean": curr_mean, "std": curr_std, "n": len(counts)}
            uncertainty = max(0.0, (upper_5min - lower_5min) / 2)
            self.current_prediction = TrafficPrediction(
                timestamp=datetime.now(),
                predicted_1min=pred_1min, predicted_3min=pred_3min, predicted_5min=pred_5min,
                lower_bound_5min=lower_5min, upper_bound_5min=upper_5min,
                uncertainty=uncertainty, model_loaded=True, features_available=True,
            )
            n_sel = int(self._mask.sum()) if self._mask is not None else len(FEATURE_COLS)
            logger.info("BiStacking: 1min=%.1f  5min=%.1f (±%.1f) | mask=%d/%d | rows=%d",
                        pred_1min, pred_5min, uncertainty, n_sel, len(FEATURE_COLS), len(feat_df))

        except Exception as e:
            logger.error("BiStacking fit error: %s", e, exc_info=True)
            self.current_prediction = self._default_prediction(True, False)
            return

        self.prediction_history.append(self.current_prediction)
        if len(self.prediction_history) > 60:
            self.prediction_history.pop(0)

    def get_current_prediction(self) -> Optional[TrafficPrediction]:
        return self.current_prediction

    def get_prediction_for_scaler(self) -> Dict[str, float]:
        if self.current_prediction is None:
            return {"predicted_load_5min": 0, "prediction_uncertainty": 100, "prediction_available": False}
        return {
            "predicted_load_5min":    self.current_prediction.predicted_5min,
            "prediction_uncertainty": self.current_prediction.uncertainty,
            "prediction_available":   self.current_prediction.model_loaded,
        }


prediction_service = PredictionService()
