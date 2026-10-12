# Standalone gap statistics for a Sokosti EMG capture CSV.
# Same computation and markdown output as the gap section of
# emg_isometric_tools/emg_isometric.py, written to
# "<csv-name>-gap_statistic.log" next to the CSV file.

import argparse
import glob
import os
import shlex
import sys

import numpy as np
import pandas as pd

# ===== COMMAND LINE ARGUMENTS =====
parser = argparse.ArgumentParser(
    description="Gap statistics for an EMG capture CSV (sample-counter column), "
                "printed as markdown and saved to <csv-name>-gap_statistic.log "
                "next to the CSV file.")
parser.add_argument("csv", nargs="?",
                    help="Path to the CSV file (defaults to the most recent capture in captures/).")
parser.add_argument("--fs", type=int, default=980,
                    help="Sampling rate in Hz used for the ms gap lengths "
                         "(the Iso_Dor trials were recorded at 980 sps; default 980).")
args = parser.parse_args()

# ===== LOAD CSV =====
csv_path = args.csv
if csv_path is None:
    files = sorted(glob.glob(os.path.join("captures", "sokosti_capture_*.csv")))
    if not files:
        print("No CSV file given and none found in captures/.", file=sys.stderr)
        sys.exit(1)
    csv_path = files[-1]
    print(f"Using most recent capture: {csv_path}")

if not os.path.isfile(csv_path):
    sys.exit(f"CSV file not found: {csv_path}")

fs = args.fs  # Hz

idx = pd.read_csv(csv_path, usecols=[0]).iloc[:, 0].values.astype(np.int64)
if idx.size == 0:
    sys.exit(f"CSV has no data rows: {csv_path}")
if np.any(np.diff(idx) < 1):
    sys.exit(f"Sample counter in {csv_path} is not strictly increasing "
             "(duplicate or out-of-order indices) - cannot compute gap statistics.")

# ===== GAP DETECTION (sample counter column) =====
full_start, full_end = int(idx[0]), int(idx[-1])
n_full = full_end - full_start + 1

received_mask = np.zeros(n_full, dtype=bool)
received_mask[idx - full_start] = True
n_missing = int((~received_mask).sum())

# ===== GAP STATISTICS (gaps in the sample-counter index column) =====
d_idx = np.diff(idx)
gap_positions = np.where(d_idx != 1)[0]
gap_lengths = [int(d_idx[g] - 1) for g in gap_positions]

# capture the report so it goes to the terminal AND into the log file
import io
import contextlib

_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    # record the exact command so the analysis can be rerun immediately:
    # "python" prefix + shell-quoted arguments
    print("# Command: python " + " ".join(shlex.quote(a) for a in sys.argv[1:]) + "\n")

    print(f"File: {csv_path}")
    print(f"Samples: {len(idx)} received, {n_missing} missing ({100.0 * n_missing / n_full:.2f}%)")
    print(f"Gaps: {len(gap_lengths)} total")
    if gap_lengths:
        gl = np.array(gap_lengths, dtype=float)
        print("\n### Gap statistics\n")
        print("| Metric | Value |")
        print("|--------|-------|")
        print(f"| Number of gaps | {len(gap_lengths)} |")
        print(f"| Samples missing | {n_missing} ({100.0 * n_missing / n_full:.2f}%) |")
        print(f"| Gap length mean (samples) | {gl.mean():.1f} +/- {gl.std():.1f} (SD) |")
        print(f"| Gap length median (samples) | {np.median(gl):.0f} |")
        print(f"| Gap length min (samples) | {gl.min():.0f} |")
        print(f"| Gap length max (samples) | {gl.max():.0f} |")
        print(f"| Gap length mean (ms @{fs} Hz) | {gl.mean()/fs*1000:.1f} +/- {gl.std()/fs*1000:.1f} (SD) |")
        print(f"| Gap length max (ms @{fs} Hz) | {gl.max()/fs*1000:.0f} |")
        print("\n### Gap details\n")
        print("| Start index | Length (samples) | Length (ms) |")
        print("|-------------|------------------|-------------|")
        for g, length in zip(gap_positions, gap_lengths):
            start_idx = int(idx[g])
            print(f"| {start_idx} | {length} | {length/fs*1000:.0f} |")
    else:
        print("No gaps detected.")

report = _buf.getvalue()

# show the report in the terminal
sys.stdout.write(report)

# ===== WRITE LOG FILE next to the CSV =====
out_dir = os.path.dirname(os.path.abspath(csv_path))
base = os.path.splitext(os.path.basename(csv_path))[0]
log_path = os.path.join(out_dir, base + "-gap_statistic.log")
with open(log_path, "w") as f:
    f.write(report)
print(f"Log file: {log_path}")