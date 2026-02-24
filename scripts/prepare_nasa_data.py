"""
NASA Kennedy Space Center HTTP Logs — Download & Full Pipeline
=============================================================
Downloads the public NASA Kennedy access logs (July and/or August 1995),
runs the full data pipeline (extract → clean → feature engineering), and
produces data/featured_traffic.csv ready for the walk-forward evaluation.

This is the same dataset the model was originally trained on.

Pipeline
--------
  Step 1 — Download + decompress  NASA_access_log_{Jul,Aug}95.gz
  Step 2 — Extract   (extract_data.py)      raw CLF logs → extracted_logs.csv
  Step 3 — Clean     (clean_data.py)        remove bad rows → cleaned_logs.csv
  Step 4 — Features  (feature_generate.py)  minute aggregates → featured_traffic.csv
  Step 5 — Evaluate  (evaluate_new_model.py) walk-forward MAE  [optional, --eval]

Usage
-----
  # Full pipeline — both months (recommended, ~89k minute-rows)
  python scripts/prepare_nasa_data.py

  # Single month only (~44k rows, faster)
  python scripts/prepare_nasa_data.py --month jul
  python scripts/prepare_nasa_data.py --month aug

  # If you already downloaded the raw .gz / plain files
  python scripts/prepare_nasa_data.py --skip-download

  # Download + pipeline + evaluation in one shot
  python scripts/prepare_nasa_data.py --eval

Source
------
  NASA Kennedy Space Center HTTP server logs (1995) — public research dataset
  Info page  : http://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html
  July  file : ftp://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz  (~20 MB)
  August file: ftp://ita.ee.lbl.gov/traces/NASA_access_log_Aug95.gz  (~25 MB)
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

# ── Project paths ──────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT / "data"
SCRIPTS_DIR = ROOT / "scripts"

# Add scripts dir so we can import existing pipeline modules
sys.path.insert(0, str(SCRIPTS_DIR))

# ── NASA log file locations ────────────────────────────────────────────────────
NASA_URLS = {
    "jul": "ftp://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz",
    "aug": "ftp://ita.ee.lbl.gov/traces/NASA_access_log_Aug95.gz",
}
LOG_FILENAMES = {
    "jul": "NASA_access_log_Jul95",
    "aug": "NASA_access_log_Aug95",
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def _progress_hook(block_count: int, block_size: int, total: int):
    """Simple ASCII progress bar for urllib.request.urlretrieve."""
    downloaded = block_count * block_size
    if total > 0:
        pct = min(downloaded / total * 100, 100)
        filled = int(40 * pct / 100)
        bar = "█" * filled + "░" * (40 - filled)
        print(f"\r  [{bar}] {pct:5.1f}%  ({downloaded / 1e6:.1f} / {total / 1e6:.1f} MB)",
              end="", flush=True)


def _section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ── Step 1: Download ───────────────────────────────────────────────────────────
def download_log(month: str, data_dir: Path) -> Path:
    """
    Download and decompress one month's NASA log file.
    Skips the download if the plain (decompressed) file already exists.
    Returns the Path to the decompressed log file.
    """
    url      = NASA_URLS[month]
    gz_path  = data_dir / f"{LOG_FILENAMES[month]}.gz"
    log_path = data_dir / LOG_FILENAMES[month]

    if log_path.exists():
        size_mb = log_path.stat().st_size / 1e6
        print(f"  {log_path.name} already exists ({size_mb:.0f} MB) — skipping download.")
        return log_path

    print(f"\nDownloading {month.upper()} 1995 logs (~20-25 MB compressed)…")
    print(f"  URL: {url}")
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, gz_path, reporthook=_progress_hook)
        print()  # newline after inline progress bar
    except Exception as e:
        gz_path.unlink(missing_ok=True)
        print(f"\n\n  ERROR: Download failed — {e}")
        print()
        print("  If the FTP server is unreachable, download manually:")
        print("    http://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html")
        print(f"  Place the .gz file here:  {data_dir}/")
        print("  Then re-run with:  python scripts/prepare_nasa_data.py --skip-download")
        sys.exit(1)

    print(f"  Decompressing {gz_path.name}…")
    with gzip.open(gz_path, "rb") as f_in, open(log_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()   # remove compressed copy to save ~20-25 MB

    size_mb = log_path.stat().st_size / 1e6
    print(f"  Decompressed → {log_path.name}  ({size_mb:.0f} MB)")
    return log_path


# ── Steps 2-4: Extract → Clean → Feature engineering ──────────────────────────
def run_pipeline(log_paths: list[Path], data_dir: Path) -> Path:
    """
    Reuse the existing pipeline scripts (extract_data, clean_data,
    feature_generate) to build featured_traffic.csv from raw log files.
    """
    import pandas as pd
    from extract_data    import extract_logs
    from clean_data      import clean_data
    from feature_generate import engineer_features

    extracted = data_dir / "extracted_logs.csv"
    cleaned   = data_dir / "cleaned_logs.csv"
    featured  = data_dir / "featured_traffic.csv"

    # ── Step 2: Extract ────────────────────────────────────────────────────────
    _section("STEP 2 — Extract raw CLF logs → extracted_logs.csv")

    if len(log_paths) == 1:
        extract_logs(log_paths[0], extracted)
    else:
        # Extract each month to a temp file, concatenate, sort, then save
        temp_frames = []
        for path in log_paths:
            tmp = data_dir / f"_tmp_{path.stem}.csv"
            extract_logs(path, tmp)
            temp_frames.append(pd.read_csv(tmp, parse_dates=["timestamp"]))
            tmp.unlink()

        combined = (
            pd.concat(temp_frames, ignore_index=True)
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        combined.to_csv(extracted, index=False)
        print(f"\nCombined {len(log_paths)} months → {len(combined):,} total request rows")

    # ── Step 3: Clean ──────────────────────────────────────────────────────────
    _section("STEP 3 — Clean and validate → cleaned_logs.csv")
    clean_data(extracted, cleaned)

    # ── Step 4: Feature engineering ───────────────────────────────────────────
    _section("STEP 4 — Feature engineering → featured_traffic.csv")
    engineer_features(cleaned, featured)

    _section("PIPELINE COMPLETE")
    print(f"  Output: {featured}")

    # Quick summary
    df = pd.read_csv(featured, parse_dates=["timestamp"])
    print(f"  Rows   : {len(df):,} minute-level rows")
    print(f"  Range  : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"  Mean   : {df['request_count'].mean():.1f} req/min")
    print(f"  Std    : {df['request_count'].std():.1f} req/min")
    print(f"  Columns: {list(df.columns)}")

    return featured


# ── Step 5 (optional): Evaluation ─────────────────────────────────────────────
def run_evaluation(featured_path: Path):
    _section("STEP 5 — Walk-forward MAE Evaluation")
    subprocess.run(
        [sys.executable,
         str(SCRIPTS_DIR / "evaluate_new_model.py"),
         "--data", str(featured_path)],
        check=True,
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download NASA Kennedy HTTP logs and build featured_traffic.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--month", choices=["jul", "aug", "both"], default="both",
        help="Which month to use.  both → ~89k rows  |  single → ~44k rows  (default: both)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download — use this if the raw log files already exist in data/",
    )
    parser.add_argument(
        "--skip-pipeline", action="store_true",
        help="Skip extract/clean/feature steps — use if featured_traffic.csv already exists",
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Run walk-forward evaluation after the pipeline finishes",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which months to process
    months = ["jul", "aug"] if args.month == "both" else [args.month]

    # ── Download ───────────────────────────────────────────────────────────────
    log_paths: list[Path] = []
    for month in months:
        log_path = DATA_DIR / LOG_FILENAMES[month]
        if args.skip_download:
            if not log_path.exists():
                print(f"ERROR: --skip-download was set but {log_path} does not exist.")
                sys.exit(1)
            log_paths.append(log_path)
        else:
            log_paths.append(download_log(month, DATA_DIR))

    # ── Pipeline ───────────────────────────────────────────────────────────────
    featured_path = DATA_DIR / "featured_traffic.csv"

    if args.skip_pipeline:
        if not featured_path.exists():
            print(f"ERROR: --skip-pipeline was set but {featured_path} does not exist.")
            sys.exit(1)
        print(f"  Skipping pipeline — using existing {featured_path}")
    else:
        featured_path = run_pipeline(log_paths, DATA_DIR)

    # ── Evaluation ────────────────────────────────────────────────────────────
    if args.eval:
        run_evaluation(featured_path)

    print("\nTo run the evaluation separately:")
    print(f"  python scripts/evaluate_new_model.py --data {featured_path}")


if __name__ == "__main__":
    main()
