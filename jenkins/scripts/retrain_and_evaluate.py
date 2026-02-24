#!/usr/bin/env python3
"""
MLOps retraining pipeline — run by Jenkins weekly (Sunday ~2 AM).

Exit codes:
  0 — new model is ≥5% better MAPE than production; model files written
  1 — no improvement (or error); Jenkins skips build/push stages

Usage:
  DATABASE_URL=postgresql://... python3 jenkins/scripts/retrain_and_evaluate.py
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import asyncpg
import numpy as np
from sklearn.preprocessing import StandardScaler

# ── Path setup — import from existing project code without installing packages ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "load_balancer"))   # BiStackingEnsemble
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))          # OOADPSOGASelector

from bistacking import BiStackingEnsemble                  # noqa: E402
from feature_selection import OOADPSOGASelector            # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@postgres:5432/loadbalancer",
)
MODELS_DIR  = PROJECT_ROOT / "load_balancer" / "models"
MAPE_FILE   = MODELS_DIR / "production_mape.json"
MASK_FILE   = MODELS_DIR / "feature_mask.json"
MIN_ROWS_FOR_FIT     = 30
IMPROVEMENT_THRESHOLD = 0.05   # 5% relative improvement required

# Hard contract — must match prediction_service.py FEATURE_COLS exactly
FEATURE_COLS = [
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("retrain")


# ── Data fetching ──────────────────────────────────────────────────────────────

async def fetch_traffic(days: int = 7) -> list[dict]:
    """Fetch traffic_aggregates rows from the last `days` days."""
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        rows = await conn.fetch(
            """
            SELECT minute_bucket, request_count
            FROM   traffic_aggregates
            WHERE  minute_bucket >= NOW() - INTERVAL '1 day' * $1
            ORDER  BY minute_bucket ASC
            """,
            days,
        )
        return [dict(r) for r in rows]
    finally:
        await conn.close()


# ── Feature engineering ────────────────────────────────────────────────────────

def engineer_features(traffic: list[dict]):
    """
    Identical logic to PredictionService._engineer_features().
    Returns (X, y, timestamps) or raises ValueError if insufficient rows.
    """
    import pandas as pd

    df = pd.DataFrame(traffic)
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
    if len(df) < MIN_ROWS_FOR_FIT:
        raise ValueError(f"Only {len(df)} rows after feature engineering — need ≥ {MIN_ROWS_FOR_FIT}")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["request_count"].values.astype(np.float32)
    return X, y, df["timestamp"]


# ── MAPE helpers ───────────────────────────────────────────────────────────────

def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error — skips zero-valued targets."""
    mask = y_true > 0
    if not mask.any():
        raise ValueError("All target values are zero — cannot compute MAPE")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]))


def load_production_mape() -> float:
    """Return stored production MAPE, or 1.0 (worst) if file doesn't exist."""
    if not MAPE_FILE.exists():
        logger.info("No %s found — treating production MAPE as 1.0 (new model always wins first run)", MAPE_FILE)
        return 1.0
    data = json.loads(MAPE_FILE.read_text())
    mape = float(data["mape"])
    logger.info("Production MAPE loaded: %.4f (recorded at %s)", mape, data.get("updated_at", "unknown"))
    return mape


def save_production_mape(mape: float) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MAPE_FILE.write_text(json.dumps({
        "mape":       round(mape, 6),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    logger.info("Saved new production MAPE: %.4f → %s", mape, MAPE_FILE)


def save_feature_mask(mask: np.ndarray) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    selected_names = [FEATURE_COLS[i] for i, v in enumerate(mask) if v]
    MASK_FILE.write_text(json.dumps({
        "mask":     mask.astype(int).tolist(),
        "features": selected_names,
        "n_selected": int(mask.sum()),
        "n_total":    len(mask),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    logger.info("Saved feature mask: %d/%d features selected → %s", int(mask.sum()), len(mask), MASK_FILE)


# ── Main pipeline ──────────────────────────────────────────────────────────────

async def main() -> int:
    logger.info("=== Weekly retraining pipeline start ===")

    # 1. Fetch data
    logger.info("Fetching 7 days of traffic_aggregates from %s …", DATABASE_URL.split("@")[-1])
    traffic = await fetch_traffic(days=7)
    logger.info("Fetched %d rows", len(traffic))
    if len(traffic) < 62:
        logger.error("Insufficient data (%d rows) — need ≥ 62 for meaningful retraining", len(traffic))
        return 1

    # 2. Feature engineering
    try:
        X, y, timestamps = engineer_features(traffic)
    except ValueError as exc:
        logger.error("Feature engineering failed: %s", exc)
        return 1
    logger.info("Features ready: X=%s  y=%s", X.shape, y.shape)

    # 3. 80/20 time-ordered split (no shuffle — preserves temporal ordering)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    logger.info("Split: train=%d  test=%d", len(X_train), len(X_test))

    # 4. Feature selection via OOA-DPSO-GA
    best_mask = np.ones(len(FEATURE_COLS), dtype=bool)
    try:
        selector = OOADPSOGASelector(n_particles=30, n_iterations=50)
        best_mask, fitness = selector.optimize(X_train, y_train)
        logger.info("Feature selector: %d/%d features selected  fitness=%.4f",
                    int(best_mask.sum()), len(FEATURE_COLS), fitness)
    except Exception as exc:
        logger.warning("Feature selector failed (%s) — falling back to all features", exc)

    # 5. Fit BiStacking on selected features
    X_train_sel = X_train[:, best_mask]
    X_test_sel  = X_test[:, best_mask]

    scaler       = StandardScaler()
    X_train_s    = scaler.fit_transform(X_train_sel)
    X_test_s     = scaler.transform(X_test_sel)

    logger.info("Fitting BiStackingEnsemble (n_folds=5) on %d samples …", len(X_train_s))
    ensemble = BiStackingEnsemble(n_folds=5)
    ensemble.fit(X_train_s, y_train)

    # 6. Evaluate on holdout
    y_pred   = ensemble.predict(X_test_s)
    new_mape = compute_mape(y_test, y_pred)
    logger.info("Holdout MAPE: %.4f  (%.2f%%)", new_mape, new_mape * 100)

    # 7. Compare with production
    prod_mape = load_production_mape()
    relative_improvement = (prod_mape - new_mape) / prod_mape if prod_mape > 0 else 1.0
    logger.info(
        "Improvement: %.4f → %.4f  (%.1f%% relative, threshold=%.0f%%)",
        prod_mape, new_mape, relative_improvement * 100, IMPROVEMENT_THRESHOLD * 100,
    )

    # Set env vars for Jenkins to read (printed so the shell can capture them)
    print(f"PROD_MAPE={prod_mape:.6f}")
    print(f"NEW_MAPE={new_mape:.6f}")

    if relative_improvement < IMPROVEMENT_THRESHOLD:
        logger.info(
            "No meaningful improvement (%.1f%% < %.0f%% threshold) — skipping model update",
            relative_improvement * 100, IMPROVEMENT_THRESHOLD * 100,
        )
        return 1

    # 8. Save model artifacts
    logger.info("Improvement threshold met — saving model artifacts")
    save_feature_mask(best_mask)
    save_production_mape(new_mape)

    logger.info("=== Retraining pipeline complete — model improved ===")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
