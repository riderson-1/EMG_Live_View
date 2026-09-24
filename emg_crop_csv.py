#!/usr/bin/env python3
"""Crop the beginning and/or end of an EMG CSV file.

The CSV is expected to have a `sample` counter column (0-indexed). Time in
seconds is derived as sample / fs (default fs = 1000 Hz).

Usage:
    python emg_crop_csv.py <input.csv> [--start SEC] [--end SEC] [--outfile OUT]

By default --start and --end are the first and last sample of the file, so
running without them just copies the file. Provide them as timestamps in
seconds to crop.

Examples:
    # Keep only seconds 10..30
    python emg_crop_csv.py in.csv --start 10 --end 30

    # Drop the first 5 seconds, keep the rest
    python emg_crop_csv.py in.csv --start 5

    # Keep everything up to second 60
    python emg_crop_csv.py in.csv --end 60
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="Path to the source EMG CSV file")
    p.add_argument("--start", type=float, default=None,
                   help="Start timestamp in seconds (default: first sample)")
    p.add_argument("--end", type=float, default=None,
                   help="End timestamp in seconds (default: last sample)")
    p.add_argument("--outfile", type=Path, default=None,
                   help="Output path (default: <input>_cropped.csv)")
    p.add_argument("--fs", type=float, default=1000.0,
                   help="Sample rate in Hz used to convert sample->seconds (default: 1000)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.is_file():
        print(f"error: file not found: {args.input}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.input)

    if "sample" not in df.columns:
        print("error: no 'sample' column found in CSV", file=sys.stderr)
        return 1

    # Convert seconds -> sample index (0-indexed counter)
    start_sample = int(round(args.start * args.fs)) if args.start is not None else 0
    end_sample = int(round(args.end * args.fs)) if args.end is not None else df["sample"].iloc[-1]

    # Clamp to valid range
    start_sample = max(0, start_sample)
    end_sample = min(int(df["sample"].iloc[-1]), end_sample)

    if start_sample > end_sample:
        print("error: --start is after --end", file=sys.stderr)
        return 1

    cropped = df[(df["sample"] >= start_sample) & (df["sample"] <= end_sample)]

    out = args.outfile or args.input.with_name(f"{args.input.stem}_cropped.csv")
    cropped.to_csv(out, index=False)

    n = len(cropped)
    dur = n / args.fs
    print(f"wrote {out}")
    print(f"  samples {start_sample}..{end_sample}  ({n} rows, ~{dur:.2f} s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())