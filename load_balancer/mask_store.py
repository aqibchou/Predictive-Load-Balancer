import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_REDIS_KEY     = "feature_mask"
_REDIS_TIMEOUT = 2  # seconds; never block the live path


def save_mask(
    mask:          np.ndarray,
    feature_names: list[str],
    redis_url:     str  = "redis://localhost:6379",
    fallback_path: Path = Path("mask.json"),
    fitness:       float = float("nan"),
) -> None:
    """Save binary feature mask to Redis (primary) and JSON file (fallback)."""
    payload = {
        "mask":          [int(b) for b in mask],
        "feature_names": list(feature_names),
        "n_selected":    int(np.sum(mask)),
        "fitness":       float(fitness),
        "updated_at":    datetime.now(timezone.utc).isoformat(),
    }
    json_str = json.dumps(payload)

    try:
        import redis as _redis
        r = _redis.from_url(redis_url, socket_connect_timeout=_REDIS_TIMEOUT,
                            socket_timeout=_REDIS_TIMEOUT)
        r.set(_REDIS_KEY, json_str)
        logger.info("Mask saved to Redis (%s). n_selected=%d fitness=%.4f",
                    redis_url, payload["n_selected"], payload["fitness"])
    except ImportError:
        logger.warning("redis package not installed — skipping Redis write.")
    except Exception as exc:
        logger.warning("Redis write failed (%s) — will use JSON fallback.", exc)

    fallback_path = Path(fallback_path)
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fallback_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Mask saved to JSON fallback: %s", fallback_path)

    selected = [name for name, bit in zip(feature_names, mask) if bit]
    print(f"  Mask saved | {payload['n_selected']}/{len(mask)} features | "
          f"fitness={payload['fitness']:.4f} | updated={payload['updated_at']}")
    print(f"  Selected: {selected}")


def load_mask(
    redis_url:     str  = "redis://localhost:6379",
    fallback_path: Path = Path("mask.json"),
) -> tuple[np.ndarray, list[str]]:
    """Load binary feature mask from Redis or JSON fallback. Raises FileNotFoundError if neither exists."""
    payload = _load_from_redis(redis_url)
    if payload is None:
        payload = _load_from_file(fallback_path)
    return np.array(payload["mask"], dtype=bool), list(payload["feature_names"])


def _load_from_redis(redis_url: str) -> Optional[dict]:
    try:
        import redis as _redis
        r   = _redis.from_url(redis_url, socket_connect_timeout=_REDIS_TIMEOUT,
                               socket_timeout=_REDIS_TIMEOUT)
        raw = r.get(_REDIS_KEY)
        if raw is None:
            return None
        return json.loads(raw)
    except ImportError:
        return None
    except Exception as exc:
        logger.debug("Redis load failed (%s) — falling back to file.", exc)
        return None


def _load_from_file(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No feature mask found in Redis or at {path}. "
            "Run the async worker first: python scripts/async_worker.py"
        )
    with open(path) as f:
        return json.load(f)


def apply_mask(X_dense: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply binary feature mask: X_dense[:, mask]."""
    return X_dense[:, mask]


def mask_info(
    redis_url:     str  = "redis://localhost:6379",
    fallback_path: Path = Path("mask.json"),
) -> dict:
    """Return mask metadata without constructing the full numpy array."""
    payload = _load_from_redis(redis_url) or _load_from_file(fallback_path)
    return {
        "n_features": len(payload["mask"]),
        "n_selected": payload["n_selected"],
        "fitness":    payload["fitness"],
        "updated_at": payload["updated_at"],
        "selected":   [name for name, bit in zip(payload["feature_names"], payload["mask"]) if bit],
    }
