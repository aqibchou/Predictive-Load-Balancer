"""
DATA CLEANING SCRIPT
Purpose: Remove duplicates, outliers, and invalid entries from extracted logs

Input:  data/extracted_logs.csv (raw structured data, ~1.9M rows)
Output: data/cleaned_logs.csv (validated data, ~1.89M rows)
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

# Valid HTTP status codes representation
# - 2xx: Success (200 OK, 204 No Content, etc.)
# - 3xx: Redirection (301 Moved, 304 Not Modified, etc.)
# - 4xx: Client errors (404 Not Found, 403 Forbidden, etc.)
# - 5xx: Server errors (500 Internal Error, 503 Unavailable, etc.)
# Anything outside this range = corrupted log entry 
VALID_STATUS_CODES = {
    # 2xx Success
    200, 201, 202, 204, 206,
    # 3xx Redirection
    300, 301, 302, 303, 304, 307, 308,
    # 4xx Client Errors
    400, 401, 403, 404, 405, 406, 408, 409, 410, 413, 414, 415, 429,
    # 5xx Server Errors
    500, 501, 502, 503, 504, 505
}

# Byte size thresholds for pre processing
MAX_BYTES = 100 * 1024 * 1024  # 100 MB
MIN_BYTES = 0  # Can't serve negative bytes or ones that dont exist


def validate_status_code(status: str) -> bool:
    """
    Check if status code is valid HTTP code
    
    Why validate?
    - Corrupted logs might have "---" or random strings
    - Non-HTTP traffic might be in the logs
    - Invalid codes break ML training
    
    Args:
        status: String status code from CSV
    
    Returns:
        True if valid HTTP code, False otherwise
    """
    try:
        code = int(status)
        return code in VALID_STATUS_CODES
    except ValueError:
        # Not even a number (e.g., "ABC", "---")
        return False


def validate_bytes(bytes_val: str) -> bool:
    """
    Check if byte size is realistic
    
    Why these bounds?
    - MIN: Can't serve negative bytes
    - MAX: 100MB is generous for 1995 (most files were <1MB)
    
    Edge cases:
    - 0 bytes is VALID (HTTP 304 Not Modified has 0-byte body)
    - Empty string should be caught in "missing fields" check
    
    Args:
        bytes_val: String byte count from CSV
    
    Returns:
        True if realistic, False if outlier
    """
    try:
        size = int(bytes_val)
        return MIN_BYTES <= size <= MAX_BYTES
    except ValueError:
        # Not a number
        return False


def detect_timestamp_issues(df: pd.DataFrame) -> dict:
    """
    Analyze timestamp quality
    
    What we're checking:
    1. Are timestamps in order? (should be mostly sorted (needed for ML model))
    2. Are there duplicates? (multiple requests same second) -> should not happen but just to be safe
    3. Are there big gaps? (server downtime, missing data)
    
    Args:
        df: DataFrame with 'timestamp' column
    
    Returns:
        Dict with timestamp statistics
    """
    # Convert to datetime for analysis
    timestamps = pd.to_datetime(df['timestamp'])
    
    # Check if sorted
    is_sorted = timestamps.is_monotonic_increasing
    
    # Count duplicates (same second)
    duplicate_timestamps = timestamps.duplicated().sum()
    
    # Find time gaps (difference between consecutive timestamps)
    time_diffs = timestamps.diff()
    max_gap = time_diffs.max()
    
    # Count out-of-order entries
    # If timestamp[i] < timestamp[i-1], that's out of order
    out_of_order = (time_diffs < pd.Timedelta(0)).sum()
    
    return {
        'is_sorted': is_sorted,
        'duplicate_timestamps': duplicate_timestamps,
        'max_gap_seconds': max_gap.total_seconds() if pd.notna(max_gap) else 0,
        'out_of_order_count': out_of_order
    }


def clean_data(input_file: Path, output_file: Path):
    """
    Main cleaning pipeline
    
    Cleaning order done in this project:
    1. Load data
    2. Remove exact duplicates (keep first occurrence)
    3. Drop missing critical fields
    4. Filter invalid status codes
    5. Remove byte outliers (large byte sizes)
    6. Validate timestamps
    7. Sort by timestamp (increasing -> to be used in our time series prediction)
    8. Save cleaned data
    """
    print(f"Reading from: {input_file}")
    print(f"Loading data into memory...\n")
    
    # Load CSV
    # low_memory=False will prevents pandas from guessing types incorrectly
    df = pd.read_csv(input_file, parse_dates=['timestamp'], low_memory=False)
    
    initial_rows = len(df)
    print(f"Initial rows: {initial_rows:,}\n")
    
    # Track what we remove at each step and set default values to fill 
    cleaning_stats = {
        'initial': initial_rows,
        'duplicates_removed': 0,
        'missing_fields_removed': 0,
        'invalid_status_removed': 0,
        'byte_outliers_removed': 0,
        'final': 0
    }
    
    # STEP 1: Remove exact duplicates
    print("Step 1: Removing exact duplicates")
    before = len(df)
    # keep='first' means if we see the same row 3 times, keep the first, drop the other 2
    # First occurrence is likely the real request from raw data, others are retries
    df = df.drop_duplicates(keep='first')
    after = len(df)
    duplicates_removed = before - after
    cleaning_stats['duplicates_removed'] = duplicates_removed
    print(f"Removed {duplicates_removed:,} exact duplicates ({100*duplicates_removed/before:.2f}%)")
    print(f" Remaining: {after:,}\n")
    
    # STEP 2: Drop rows with missing critical fields
    print("Step 2: Dropping rows with missing critical fields")
    before = len(df)
    # Critical fields: timestamp, path, status
    # timestamp, path, status are essential for traffic analysis 
    df = df.dropna(subset=['timestamp', 'path', 'status'])
    after = len(df)
    missing_removed = before - after
    cleaning_stats['missing_fields_removed'] = missing_removed
    print(f"  Removed {missing_removed:,} rows with missing fields ({100*missing_removed/before:.2f}%)")
    print(f"  Remaining: {after:,}\n")
    
    # STEP 3: Filter invalid status codes
    print("Step 3: Filtering invalid HTTP status codes...")
    before = len(df)
    # Apply validation function to each status code
    # Keep only rows where validate_status_code returns True
    df = df[df['status'].apply(validate_status_code)]
    after = len(df)
    invalid_status_removed = before - after
    cleaning_stats['invalid_status_removed'] = invalid_status_removed
    print(f"  Removed {invalid_status_removed:,} rows with invalid status codes ({100*invalid_status_removed/before:.2f}%)")
    print(f"  Remaining: {after:,}\n")
    
    # STEP 4: Remove byte outliers
    print("Step 4: Removing byte size outliers...")
    before = len(df)
    df = df[df['bytes'].apply(validate_bytes)]
    after = len(df)
    outliers_removed = before - after
    cleaning_stats['byte_outliers_removed'] = outliers_removed
    print(f"  Removed {outliers_removed:,} rows with unrealistic byte sizes ({100*outliers_removed/before:.2f}%)")
    print(f"  Remaining: {after:,}\n")
    
    # STEP 5: Analyze timestamp quality
    print("Step 5: Analyzing timestamp quality...")
    ts_stats = detect_timestamp_issues(df)
    print(f"  Sorted: {ts_stats['is_sorted']}")
    print(f"  Duplicate timestamps: {ts_stats['duplicate_timestamps']:,}")
    print(f"  Max time gap: {ts_stats['max_gap_seconds']:.0f} seconds")
    print(f"  Out-of-order entries: {ts_stats['out_of_order_count']:,}")
    
    # Warning if major timestamp issues
    if not ts_stats['is_sorted']:
        print("Data not sorted by timestamp (will sort in next step)")
    if ts_stats['out_of_order_count'] > 0.01 * len(df):  # More than 1% out of order
        print(f"{ts_stats['out_of_order_count']:,} out-of-order entries")
    print()
    
    # STEP 6: Sort by timestamp
    print("Step 6: Sorting by timestamp...")
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # STEP 7: Final statistics
    cleaning_stats['final'] = len(df)
    total_removed = initial_rows - len(df)
    
    print(f"{'='*60}")
    print("CLEANING COMPLETE")
    print(f"{'='*60}")
    print(f"Initial rows:              {cleaning_stats['initial']:,}")
    print(f"Exact duplicates removed:  {cleaning_stats['duplicates_removed']:,}")
    print(f"Missing fields removed:    {cleaning_stats['missing_fields_removed']:,}")
    print(f"Invalid status removed:    {cleaning_stats['invalid_status_removed']:,}")
    print(f"Byte outliers removed:     {cleaning_stats['byte_outliers_removed']:,}")
    print(f"{'-'*60}")
    print(f"Total removed:             {total_removed:,} ({100*total_removed/initial_rows:.2f}%)")
    print(f"Final rows:                {cleaning_stats['final']:,} ({100*cleaning_stats['final']/initial_rows:.2f}%)")
    
    # Data quality check
    retention_rate = cleaning_stats['final'] / cleaning_stats['initial']
    if retention_rate < 0.95:
        print(f"High data loss ({100*(1-retention_rate):.1f}%)")
        print(" Expected: <5% loss. Investigate cleaning steps.")
    else:
        print(f"Data quality: good (retained {100*retention_rate:.2f}%)")
    
    # Save cleaned data
    print(f"\nSaving to: {output_file}")
    df.to_csv(output_file, index=False)
    print("Cleaned data saved successfully")
    
    # Return stats for potential further analysis
    return cleaning_stats


if __name__ == "__main__":
    # Setup paths
    # Path to the data directory where raw + extracted data is held
    data_dir = Path(__file__).resolve().parents[1] / "data"

    # Correct input and output paths
    input_file = data_dir / "Extracted_NASA_access_log_Jul95"
    output_file = data_dir / "cleaned_logs.csv"
    
    # Run cleaning
    stats = clean_data(input_file, output_file)