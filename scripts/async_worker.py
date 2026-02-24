"""Slow-path feature selection worker. Runs OOA-DPSO-GA on historical data
and writes a binary feature mask to Redis + JSON fallback via mask_store."""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add scripts/ and load_balancer/ to path for local imports
SCRIPTS = Path(__file__).resolve().parent
LB      = SCRIPTS.parent / "load_balancer"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(LB))

from feature_selection import OOADPSOGASelector
from mask_store import save_mask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("async_worker")


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # IMPORTANT: feature order must match prediction_service.py FEATURE_COLS
    rc = df["request_count"].astype(float)
    ts = df["timestamp"]

    for lag in [1, 2, 3, 5, 10, 15, 30, 60]:
        df[f"lag_{lag}"] = rc.shift(lag)

    for w in [5, 10, 15, 30, 60]:
        df[f"roll_mean_{w}"] = rc.shift(1).rolling(w).mean()
        df[f"roll_std_{w}"]  = rc.shift(1).rolling(w).std()
        df[f"roll_max_{w}"]  = rc.shift(1).rolling(w).max()
        df[f"roll_min_{w}"]  = rc.shift(1).rolling(w).min()

    df["hour"]       = ts.dt.hour
    df["dayofweek"]  = ts.dt.dayofweek
    df["minute_mod"] = ts.dt.minute
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    for span in [5, 15, 30]:
        df[f"ema_{span}"] = rc.ewm(span=span, adjust=False).mean().shift(1)

    df["diff_1"]  = rc.diff(1)
    df["diff_5"]  = rc.diff(5)
    df["diff_60"] = rc.diff(60)

    df = df.dropna().reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ("timestamp", "request_count")]
    return df, feature_cols


def load_recent_data(data_path: Path, window_hours: int) -> pd.DataFrame:
    logger.info("Loading data from %s (last %d hours) ...", data_path, window_hours)
    df = pd.read_csv(data_path)

    # Normalise column names
    if "minute" in df.columns and "count" in df.columns:
        df = df.rename(columns={"minute": "timestamp", "count": "request_count"})
    if "timestamp" not in df.columns or "request_count" not in df.columns:
        df.columns = ["timestamp", "request_count"] + list(df.columns[2:])

    df["timestamp"]     = pd.to_datetime(df["timestamp"])
    df["request_count"] = pd.to_numeric(df["request_count"], errors="coerce").fillna(0)

    if "node" in df.columns or "node_idx" in df.columns:
        df = df.groupby("timestamp")["request_count"].sum().reset_index()

    df = df.sort_values("timestamp").reset_index(drop=True)

    if window_hours > 0:
        cutoff = df["timestamp"].max() - pd.Timedelta(hours=window_hours)
        df     = df[df["timestamp"] >= cutoff].reset_index(drop=True)

    logger.info("  %d rows after window filter (≥ %s)",
                len(df), df["timestamp"].min() if len(df) else "N/A")
    return df[["timestamp", "request_count"]]


def run_worker(args: argparse.Namespace) -> None:
    t0 = time.time()
    logger.info("ASYNC FEATURE SELECTION WORKER — OOA-DPSO-GA")

    raw = load_recent_data(Path(args.data), args.window_hours)
    if len(raw) < 200:
        logger.error("Only %d rows. Need ≥ 200 — increase --window-hours or check data path.", len(raw))
        sys.exit(1)

    logger.info("Engineering features ...")
    feat_df, feature_cols = engineer_features(raw.copy())
    X = feat_df[feature_cols].values.astype(np.float32)
    y = feat_df["request_count"].values.astype(np.float32)
    logger.info("  Feature matrix: %d rows × %d features", *X.shape)

    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X_s = (X - mean) / std

    logger.info("Running OOA-DPSO-GA ...")
    selector = OOADPSOGASelector(
        n_particles  = args.n_particles,
        n_iterations = args.n_iterations,
        w_max        = 0.9,
        w_min        = 0.4,
        c1           = 1.5,
        c2           = 1.5,
        pc           = 0.8,
        pm_scale     = 1.0,
        sparsity_pen = args.sparsity_pen,
        random_state = args.seed,
    )
    selector.fit(X_s, y)

    redis_url     = args.redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379")
    fallback_path = Path(args.mask_path)
    logger.info("Saving mask to Redis=%s, file=%s ...", redis_url, fallback_path)
    save_mask(
        mask          = selector.mask_,
        feature_names = feature_cols,
        redis_url     = redis_url,
        fallback_path = fallback_path,
        fitness       = selector.best_fitness_,
    )

    logger.info("Worker complete in %.1fs | %d/%d features selected | fitness=%.4f",
                time.time() - t0, len(selector.selected_features_), len(feature_cols),
                selector.best_fitness_)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OOA-DPSO-GA async feature selection worker")
    p.add_argument("--data",         required=True,                             help="Path to input CSV")
    p.add_argument("--window-hours", type=int,   default=0,                     help="Hours of recent data (0 = all)")
    p.add_argument("--redis-url",    default=None,                              help="Redis URL (default: REDIS_URL env or redis://localhost:6379)")
    p.add_argument("--mask-path",    default="results/feature_mask.json",       help="JSON fallback path for the mask")
    p.add_argument("--n-particles",  type=int,   default=30,                    help="PSO swarm size")
    p.add_argument("--n-iterations", type=int,   default=50,                    help="PSO iterations")
    p.add_argument("--sparsity-pen", type=float, default=0.05,                  help="Sparsity penalty (higher = fewer features)")
    p.add_argument("--seed",         type=int,   default=42,                    help="Random seed")
    return p


if __name__ == "__main__":
    run_worker(_build_parser().parse_args())
