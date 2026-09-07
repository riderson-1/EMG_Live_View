#!/usr/bin/env python3
"""
Convert a Sokosti EMG capture CSV into an openhdemg-style .mat file matching
the structure of matlab_example_max/TD12_PF_Ramp25openhdemg.mat.

Output variables (all float64, matching the reference file):
    RAW_SIGNAL  : (N, n_channels) EMG in microvolts (electrode-surface units, IED=10 mm)
    REF_SIGNAL  : (N, 1)          reference/force channel (normalized 0-1 or volts)
    FSAMP       : (1, 1) uint16   sampling frequency in Hz
    IED         : (1, 1) uint8    inter-electrode distance in mm

Usage:
    python csv_to_mat.py input.csv [-o output.mat]
        [--force-col accel_z]      # column used as REF_SIGNAL
        [--vref 4.5] [--gain 24]   # ADS1299 conversion parameters
        [--fs 1000] [--ied 10]
        [--channels 1 2 3 ...]     # 1-based EMG channels to include (default: all 16)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.io import savemat


def ads1299_code_to_microvolts(code, vref=4.5, gain=24):
    """Convert signed ADS1299 ADC codes to input-referred microvolts."""
    code = np.asarray(code, dtype=np.float64)
    # Handle raw unsigned 24-bit codes if present (BLE path sends unsigned)
    return code * (vref / gain) / 8388607.0 * 1e6


def main():
    ap = argparse.ArgumentParser(description="CSV -> openhdemg-style .mat converter")
    ap.add_argument("csv", help="Input CSV file")
    ap.add_argument("-o", "--output", help="Output .mat file (default: <csv name>_openhdemg.mat next to the CSV)")
    ap.add_argument("--vref", type=float, default=4.5, help="ADS1299 reference voltage (V), default 4.5")
    ap.add_argument("--gain", type=float, default=1, help="ADS1299 PGA gain, default 24")
    ap.add_argument("--fs", type=int, default=1000, help="Sampling frequency (Hz), default 1000")
    ap.add_argument("--ied", type=int, default=10, help="Inter-electrode distance (mm), default 10")
    ap.add_argument("--channels", type=int, nargs="+", default=None,
                    help="1-based EMG channels to include (default: all present chN columns)")
    ap.add_argument("--force-col", default="accel_z",
                    help="CSV column stored as REF_SIGNAL (default: accel_z)")
    ap.add_argument("--ref-normalize", action="store_true",
                    help="Min-max normalize REF_SIGNAL to 0-1 (like the reference file)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    emg_cols = [c for c in df.columns if c.startswith("ch") and c[2:].isdigit()]
    emg_cols.sort(key=lambda c: int(c[2:]))
    if args.channels:
        emg_cols = [f"ch{c}" for c in args.channels]
    missing = [c for c in emg_cols if c not in df.columns]
    if missing:
        sys.exit(f"Missing EMG columns in CSV: {missing}")

    # Raw ADC codes -> signed 24-bit -> microvolts
    raw = df[emg_cols].to_numpy(dtype=np.float64)
    raw_signed = np.where(raw >= 8388608, raw - 16777216, raw)  # two's complement 24-bit
    raw_uV = ads1299_code_to_microvolts(raw_signed, vref=args.vref, gain=args.gain)

    # Reference signal
    if args.force_col not in df.columns:
        sys.exit(f"Column '{args.force_col}' not found in CSV. "
                 f"Available: {', '.join(df.columns)}")
    ref = df[args.force_col].to_numpy(dtype=np.float64)
    # drop rows where the reference is missing (BLE packets without IMU payload)
    valid = ~np.isnan(ref)
    ref = ref[valid].reshape(-1, 1)
    raw_uV = raw_uV[valid]
    if args.ref_normalize:
        span = ref.max() - ref.min()
        if span > 0:
            ref = (ref - ref.min()) / span

    n = raw_uV.shape[0]
    if raw_uV.shape[0] != ref.shape[0]:
        sys.exit("Row count mismatch between EMG and reference after cleaning.")

    out_path = args.output or os.path.splitext(args.csv)[0] + "_openhdemg.mat"
    savemat(out_path, {
        "RAW_SIGNAL": raw_uV,
        "REF_SIGNAL": ref,
        "FSAMP": np.array([[args.fs]], dtype=np.uint16),
        "IED": np.array([[args.ied]], dtype=np.uint8),
    })

    print(f"Wrote {out_path}")
    print(f"  RAW_SIGNAL : {raw_uV.shape[0]} samples x {raw_uV.shape[1]} channels (uV)")
    print(f"  REF_SIGNAL : {ref.shape[0]} samples x 1 ('{args.force_col}')")
    print(f"  FSAMP      : {args.fs} Hz")
    print(f"  IED        : {args.ied} mm")


if __name__ == "__main__":
    main()
