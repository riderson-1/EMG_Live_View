#!/usr/bin/env python3
"""
Sit-to-stand trial plotter: reads the per-trial pipeline CSVs written by
sit_stand_analysis.py --save-trial-csv

    <stem>_signals.csv  - 50 Hz IMU/PCA grid + '# key: value' header comments
                          (PCA statistics, transition times, cycle metadata)
    <stem>_emg.csv      - 1000 Hz EMG envelope (µV) + header comments

and produces cycle-locked overlay figures (0-100 % of the sit-to-stand
cycle, individual traces + mean + 90 % CI) for the EMG envelope, the
dynamic-accel PC1 and the angular-rate PC1:

  - one overlay figure per trial  (<trial_id>_cycle_overlays.png)
  - one cross-trial figure with every trial's mean curve on one plot
    (sit_stand_cycle_means.png)
  - sit_stand_plot_data.csv with exactly the plotted numbers
    (trial, signal, cycle %, mean, CI low/high)

Usage:
    python plot_sit_stand_trials.py                 # all Sit_Stand trials
    python plot_sit_stand_trials.py --out-dir DIR   # custom output folder
    python plot_sit_stand_trials.py --source PC     # PC recordings only
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

try:
    import matplotlib
    if os.environ.get("DISPLAY", "") == "" and "--no-plot" not in sys.argv:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

BASE = "/home/karl/Documents/Master_Thesis/Testing/application_testing"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

# trial folder / file stem, e.g. BLE_1_Karl_Sit_Stand_2026-09-08
DIR_RE = re.compile(
    r"^(?P<connection>USB|BLE)_(?P<rep>\d+)_(?P<subject>[A-Za-z]+)_Sit_Stand_"
    r"(?P<date>\d{4}-\d{2}-\d{2})$"
)
STEM_RE = re.compile(
    r"^(?P<connection>USB|BLE)_(?P<rep>\d+)_(?P<subject>[A-Za-z]+)_Sit_Stand_"
    r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<source>PC|SD)$"
)

# signals plotted per trial, (column in _signals.csv, label, unit)
SIGNALS = [
    ("emg_envelope_uV", "EMG ch1 envelope", "µV"),
    ("acc_pc1_g", "dyn accel PC1", "g"),
    ("rate_pc1_dps", "ang. rate PC1", "deg/s"),
]

N_GRID = 201  # points on the 0-100 % cycle grid


def read_header_comments(path):
    """'# key: value' lines -> dict (first occurrence wins)."""
    meta = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                break
            m = re.match(r"#\s*([^:]+):\s*(.*)", line.strip())
            if m:
                meta.setdefault(m.group(1).strip(), m.group(2).strip())
    return meta


def parse_transition_times(meta):
    """'# transition_times_s: 641.6,653.1,...' -> list of floats, or None."""
    s = meta.get("transition_times_s")
    if not s:
        return None
    return [float(x) for x in s.split(",") if x.strip()]


def load_trial(signals_path, emg_path):
    """Load one trial's CSVs. Returns (meta, signals_df, emg_df) or None if
    the trial has no transition times (nothing to overlay)."""
    meta = read_header_comments(signals_path)
    times = parse_transition_times(meta)
    if not times or len(times) < 2:
        return None
    signals = pd.read_csv(signals_path, comment="#")
    emg = pd.read_csv(emg_path, comment="#")
    return meta, signals, emg


def cycle_curves(t, y, cycles, time_axis="percent"):
    """Interpolate a signal onto the cycle grid for every cycle.

    cycles: list of (start_s, end_s). Returns (grid, curves) where curves is
    (n_cycles, n_grid); cycles with <10 valid samples are skipped."""
    if time_axis == "percent":
        grid = np.linspace(0.0, 100.0, N_GRID)
    else:
        durs = np.array([b - a for a, b in cycles])
        grid = np.arange(0.0, durs.min(), 0.01)
    curves = []
    for (a, b) in cycles:
        if time_axis == "percent":
            x_cycle = (t - a) / (b - a) * 100.0
        else:
            x_cycle = t - a
        m = (~np.isnan(y)) & (t >= a) & (t <= b)
        if m.sum() < 10:
            continue
        curves.append(np.interp(grid, x_cycle[m], y[m]))
    if not curves:
        return grid, None
    return grid, np.stack(curves)


def make_trial_figure(trial_id, curves_by_signal, out_path):
    """3-subplot overlay figure for one trial: individual traces (grey
    dashed) + mean (black) + 90 % CI, one subplot per signal."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                             constrained_layout=True)
    fig.suptitle(f"Sit-to-stand-cycle overlays: {trial_id}")
    for ax, (col, label, unit) in zip(axes, SIGNALS):
        grid, curves = curves_by_signal[col]
        if curves is None:
            ax.text(0.5, 0.5, "No valid cycles in this signal", ha="center",
                    transform=ax.transAxes)
            continue
        n = curves.shape[0]
        for k in range(n):
            ax.plot(grid, curves[k], color="0.85", lw=0.7, ls="--", alpha=0.6,
                    label="Individual" if k == 0 else None)
        mean = curves.mean(axis=0)
        if n > 1:
            ci = 1.645 * curves.std(axis=0, ddof=1) / np.sqrt(n)
            ax.fill_between(grid, mean - ci, mean + ci, color="0.35",
                            alpha=0.35, lw=0, label="90% CI")
        ax.plot(grid, mean, color="black", lw=1.4, label=f"Mean (n={n})")
        ax.set_ylabel(f"{label} ({unit})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("% of sit-to-stand cycle (0 = start of standing up)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_means_figure(means_by_signal, out_path):
    """Cross-trial figure: every trial's mean curve on one subplot per
    signal (one colour per trial)."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                             constrained_layout=True)
    fig.suptitle("Sit-to-stand-cycle mean curves across trials")
    for ax, (col, label, unit) in zip(axes, SIGNALS):
        for (trial_id, grid, mean) in means_by_signal[col]:
            ax.plot(grid, mean, lw=1.1, label=trial_id)
        ax.set_ylabel(f"{label} ({unit})")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("% of sit-to-stand cycle (0 = start of standing up)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def find_trials(base, source=None):
    """All Sit_Stand trials with both pipeline CSVs. Returns a list of
    (trial_id, signals_path, emg_path)."""
    trials = []
    for dirpath, dirnames, filenames in os.walk(base):
        if not DIR_RE.match(os.path.basename(dirpath)):
            continue
        for fn in sorted(filenames):
            m = STEM_RE.match(fn[: -len("_signals.csv")]) if fn.endswith("_signals.csv") else None
            if not m:
                continue
            if source and m.group("source") != source:
                continue
            stem = m.group(0)
            signals_path = os.path.join(dirpath, stem + "_signals.csv")
            emg_path = os.path.join(dirpath, stem + "_emg.csv")
            if os.path.exists(emg_path):
                trials.append((stem, signals_path, emg_path))
    return sorted(trials)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=BASE,
                    help="Root folder to search for the trial CSVs")
    ap.add_argument("--out-dir", default=OUT_DIR,
                    help="Folder for the figures and the plot-data CSV")
    ap.add_argument("--source", choices=("PC", "SD"), default=None,
                    help="Only plot this recording source (default: all)")
    ap.add_argument("--no-plot", action="store_true",
                    help="Do not open an interactive plot window")
    args = ap.parse_args(argv)

    if plt is None:
        raise SystemExit("matplotlib is required for the plotter")

    trials = find_trials(args.base, args.source)
    if not trials:
        raise SystemExit(f"No *_signals.csv/*_emg.csv pairs found under {args.base}")
    print(f"Found {len(trials)} trials")

    plot_rows = []
    means_by_signal = {col: [] for col, _, _ in SIGNALS}
    for trial_id, signals_path, emg_path in trials:
        loaded = load_trial(signals_path, emg_path)
        if loaded is None:
            print(f"skipped (no transition times): {trial_id}")
            continue
        meta, signals, emg = loaded
        times = parse_transition_times(meta)
        cycles = list(zip(times[::2], times[1::2]))
        print(f"{trial_id}: {len(cycles)} cycles")

        # EMG time is absolute (sample/fs); IMU time is zero-based at the
        # first IMU row -> shift it into the EMG sample-time base using the
        # imu_t0_s header comment.
        t_imu0 = float(meta.get("imu_t0_s", "0"))

        curves_by_signal = {}
        for col, _, _ in SIGNALS:
            if col == "emg_envelope_uV":
                t, y = emg["t_s"].values, emg[col].values
            else:
                t, y = signals["t_s"].values + t_imu0, signals[col].values
            grid, curves = cycle_curves(t, y, cycles)
            curves_by_signal[col] = (grid, curves)
            if curves is not None:
                mean = curves.mean(axis=0)
                n = curves.shape[0]
                ci = (1.645 * curves.std(axis=0, ddof=1) / np.sqrt(n)
                      if n > 1 else np.zeros_like(mean))
                means_by_signal[col].append((trial_id, grid, mean))
                for x, m, lo, hi in zip(grid, mean, mean - ci, mean + ci):
                    plot_rows.append({
                        "trial_id": trial_id,
                        "signal": col,
                        "cycle_pct": x,
                        "mean": m,
                        "ci_low": lo,
                        "ci_high": hi,
                        "n_cycles": n,
                    })

        make_trial_figure(trial_id, curves_by_signal,
                          os.path.join(args.out_dir, trial_id + "_cycle_overlays.png"))

    means_path = os.path.join(args.out_dir, "sit_stand_cycle_means.png")
    make_means_figure(means_by_signal, means_path)

    plot_data = pd.DataFrame(plot_rows)
    csv_path = os.path.join(args.out_dir, "sit_stand_plot_data.csv")
    plot_data.to_csv(csv_path, index=False)

    print(f"written: {means_path}")
    print(f"written: {csv_path}  ({len(plot_data)} rows)")

    if not args.no_plot:
        plt.show()


if __name__ == "__main__":
    main()
