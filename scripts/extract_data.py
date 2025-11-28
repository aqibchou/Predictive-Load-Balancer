"""
DATA EXTRACTION SCRIPT
Purpose: Parse raw NASA Apache logs into structured CSV format

Input:  data/NASA_access_log_Aug95 (raw Apache logs, human readable format)
Output: data/extracted_logs.csv (timestamp, host, path, method, status, bytes)

Raw Format:
The logs are an ASCII file with one line per request, with the following columns:
1) host making the request. 
2) A hostname when possible, otherwise the Internet address if the name could not be looked up.
    timestamp in the format "DAY MON DD HH:MM:SS YYYY", where DAY is the day of the week, MON is the name of the month, 
    DD is the day of the month, HH:MM:SS is the time of day using a 24-hour clock, and YYYY is the year. The timezone is -0400.
3) request given in quotes.
4) HTTP reply code.
5) bytes in the reply.

"""


import re
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
import sys

#### TESTING A SAMPLE REGEX - MIGHT NEED TO REMOVE LATER 
# Apache Common Log Format regex pattern
# Groups: (host) (timestamp) (method) (path) (status) (bytes)

LOG_PATTERN = re.compile(
    r'(\S+) '          # host - any non-whitespace (IP or hostname)
    r'- - '            # remote logname and username (always "- -" in NASA logs, so hardcoded (easy fix))
    r'\[(.*?)\] '      # timestamp in brackets [01/Jul/1995:00:00:01 -0400]
    r'"(\S+) '         # method - GET, POST, etc
    r'(\S+) '          # path - /history/apollo/
    r'HTTP/\d\.\d" '   # protocol - HTTP/1.0 or HTTP/1.1 (we don't need this, so not captured)
    r'(\d+) '          # status - 200, 404, 500, etc
    r'(\S+)'           # bytes - response size (could be "-" for 0 bytes)
)


def parse_timestamp(timestamp_str: str) -> Optional[str]:
    """
    Convert Apache timestamp to ISO format
    
    Why ISO format?
    - Sortable as strings (chronological order)
    - Compatible with pandas, Prophet, and every ML library used in this project
    - Easy to parse back into datetime objects
    
    Args:
        timestamp_str: "01/Jul/1995:00:00:01 -0400"
    
    Returns:
        "1995-07-01 00:00:01" or None if invalid
    """
    try:
        # Parse: "01/Jul/1995:00:00:01 -0400"
        # Apache uses this insane format, so we need to specify it exactly
        dt = datetime.strptime(timestamp_str, "%d/%b/%Y:%H:%M:%S %z")
        
        # Convert to ISO format without timezone (we'll treat all as UTC)
        # Why remove timezone? NASA logs are all Eastern time, treating uniformly simplifies analysis
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        # Invalid timestamp format - corrupted log line
        return None


def parse_bytes(bytes_str: str) -> int:
    """
    Convert byte string to integer
    
    Why special handling?
    - Apache logs use "-" for 0 bytes (responses with no body)
    - Some entries might be corrupted (non-numeric values)
    
    Args:
        bytes_str: "6245" or "-"
    
    Returns:
        Integer bytes, or 0 if "-" or invalid
    """
    if bytes_str == "-":
        return 0
    try:
        return int(bytes_str)
    except ValueError:
        # Corrupted byte count, default to 0
        return 0


def parse_log_line(line: str) -> Optional[Dict[str, str]]:
    """
    Parse a single Apache log line into structured data
    
    Args:
        line: Raw log string
    
    Returns:
        Dict with {timestamp, host, path, method, status, bytes} or None if unparseable
    """
    match = LOG_PATTERN.match(line.strip())
    
    if not match:
        # Line doesn't match Apache format - could be:
        # - Corrupted log entry
        # - Header/footer lines
        # - Incomplete write to log file
        return None
    
    host, timestamp_str, method, path, status, bytes_str = match.groups()
    
    # Parse timestamp
    timestamp = parse_timestamp(timestamp_str)
    if timestamp is None:
        # Invalid timestamp means we can't use this entry for time-series analysis
        return None
    
    # Parse bytes
    bytes_val = parse_bytes(bytes_str)
    
    return {
        "timestamp": timestamp,
        "host": host,
        "path": path,
        "method": method,
        "status": status,
        "bytes": str(bytes_val)  # Keep as string for CSV writing
    }


def extract_logs(input_file: Path, output_file: Path):
    """
    Main extraction function used in data cleaning and feature engineering in this project
    """
    total_lines = 0
    parsed_lines = 0
    failed_lines = 0
    
    ##### Debug print statments used in training model -> will remove 
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    print("Parsing log entries...\n")
    
    with open(input_file, 'r', encoding='utf-8', errors='replace') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        # CSV writer with headers
        fieldnames = ["timestamp", "host", "path", "method", "status", "bytes"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for line_num, line in enumerate(infile, 1):
            total_lines += 1
            
            # Parse the line
            parsed = parse_log_line(line)
            
            if parsed:
                writer.writerow(parsed)
                parsed_lines += 1
            else:
                failed_lines += 1
                # Uncomment to see what failed (DEBBUGGING)
                # if failed_lines <= 10:  # Only show first 10 failures
                #     print(f"Failed line {line_num}: {line[:100]}")
            
            # Progress indicator every 100k lines
            if line_num % 100000 == 0:
                print(f"Processed {line_num:,} lines ({parsed_lines:,} parsed, {failed_lines:,} failed)")
    
    # Final statistics
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total lines read:    {total_lines:,}")
    print(f"Successfully parsed: {parsed_lines:,} ({100*parsed_lines/total_lines:.2f}%)")
    print(f"Failed to parse:     {failed_lines:,} ({100*failed_lines/total_lines:.2f}%)")
    print(f"\nOutput saved to: {output_file}")
    
    # Warning if failure rate is too high
    failure_rate = failed_lines / total_lines
    if failure_rate > 0.05:  # More than 5% failed
        print(f"\n⚠️  WARNING: High failure rate ({failure_rate*100:.1f}%)")
        print("   Check your input file format or regex pattern")


if __name__ == "__main__":
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    # Input/output: raw NASA logs
    input_file = r"C:\Users\moham\project-group-101\data\NASA_access_log_Jul95"
    output_file = r"C:\Users\moham\project-group-101\data\Extracted_NASA_access_log_Jul95"

    
    # Validate input exists
    '''if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        print(f"\nExpected file at: {input_file}")
        print("Download NASA logs from: http://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html")
        sys.exit(1)'''
    
    # Run extraction
    extract_logs(input_file, output_file)