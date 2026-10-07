#!/usr/bin/env python3
"""
Sit-to-stand GROUP plotter: one figure comparing the four gain-1 PC
recordings, built from the same per-trial pipeline CSVs as
plot_sit_stand_trials.py:

    <stem>_signals.csv  - 50 Hz IMU/PCA grid + '# key: value' header comments
    <stem>_emg.csv      - 1000 Hz EMG envelope (µV) + header comments

Figure layout (2 x 2 quadrants, one trial per quadrant):

    top-left:     Karl  BLE (gain 1)      top-right:  Max BLE (gain 1)
    bottom-left:  Karl  USB (gain 1)      bottom-right: Max USB (gain 1)

Each quadrant has 4 stacked subplots of cycle-locked overlays
(0-100 % of the sit-to-stand cycle, individual traces + mean + 90 % CI):

    1. xyz Euler angles (roll / pitch / yaw, one colour per angle)
    2. EMG ch1 envelope (µV)
    3. dynamic accel PC1 (g)
    4. angular-rate PC1 (deg/s)

Also writes group_plot_data.csv with exactly the plotted numbers
(trial, signal, cycle %, mean, CI low/high, n_cycles).

Usage:
    python plot_sit_stand_group.py                 # default: PC, gain 1
    python plot_sit_stand_group.py --out-dir DIR   # custom output folder
"""
import argparse
import os
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

from plot_sit_stand_trials import (
    BASE,
    N_GRID,
    SIGNALS,
    cycle_curves,
    find_trials,
    load_trial,
    parse_transition_times,
    read_header_comments,
)

# quadrant -> (subject, connection); rep 1 == gain 1, PC source only
QUADRANTS = [
    ("Karl", "BLE"),
    ("Max", "BLE"),
    ("Karl", "USB"),
    ("Max", "USB"),
]

EULER_SIGNALS = [
    ("roll_deg", "roll", "deg"),
    ("pitch_deg", "pitch", "deg"),
    ("yaw_deg", "yaw", "deg"),
]
EULER_COLORS = {"roll_deg": "tab:red", "pitch_deg": "tab:green",
                "yaw_deg": "tab:blue"}


def find_group_trials(base, source="PC", gain=1):
    """The four group trials: (subject, connection) from QUADRANTS, rep 1,
    given source, and '# gain' matching in the EMG header.

    Returns {quadrant_title: (trial_id, signals_path, emg_path)}; missing
    trials are simply absent from the dict."""
    trials = find_trials(base, source)
    by_key = {}
    for trial_id, signals_path, emg_path in trials:
        m = trial_id.split("_")
        # BLE_1_Karl_Sit_Stand_2026-09-08_PC
        connection, rep, subject = m[0], m[1], m[2]
        if rep != str(gain):
            continue
        emg_meta = read_header_comments(emg_path)
        if emg_meta.get("gain") not in (str(gain), None):
            continue
        by_key[(subject, connection)] = (trial_id, signals_path, emg_path)
    return {f"{subj} {conn} (gain {gain})": by_key[(subj, conn)]
            for subj, conn in QUADRANTS if (subj, conn) in by_key}


def compute_curves(meta, signals, emg, cycles, columns):
    """Cycle-locked curves for the requested columns. Returns
    {column: (grid, curves)}; IMU columns are shifted by imu_t0_s into the
    EMG sample-time base, exactly like plot_sit_stand_trials.py."""
    t_imu0 = float(meta.get("imu_t0_s", "0"))
    out = {}
    for col in columns:
        if col == "emg_envelope_uV":
            t, y = emg["t_s"].values, emg[col].values
        else:
            t, y = signals["t_s"].values + t_imu0, signals[col].values
        out[col] = cycle_curves(t, y, cycles)
    return out


def plot_overlay(ax, grid, curves, color="black", label_mean="Mean",
                 show_traces=True, ci_label="90% CI"):
    """Individual traces (light dashed) + mean + 90 % CI on one axes."""
    if curves is None:
        ax.text(0.5, 0.5, "No valid cycles", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        return
    n = curves.shape[0]
    if show_traces:
        for k in range(n):
            ax.plot(grid, curves[k], color="0.85", lw=0.6, ls="--",
                    alpha=0.6, label="Individual" if k == 0 else None)
    mean = curves.mean(axis=0)
    if n > 1:
        ci = 1.645 * curves.std(axis=0, ddof=1) / np.sqrt(n)
        ax.fill_between(grid, mean - ci, mean + ci, color=color, alpha=0.25,
                        lw=0, label=ci_label if ci_label else None)
    ax.plot(grid, mean, color=color, lw=1.3, label=label_mean)
    ax.grid(True, alpha=0.3)


def plot_quadrant(axes, trial_id, curves_by_col):
    """Draw the 4 stacked subplots of one quadrant.

    axes: list of 4 Axes; top = Euler angles, then EMG, accel PC1, rate PC1.
    """
    # 1. Euler angles: one colour per angle, mean + CI (+ light traces)
    for col, label, unit in EULER_SIGNALS:
        grid, curves = curves_by_col[col]
        plot_overlay(axes[0], grid, curves, color=EULER_COLORS[col],
                     label_mean=label, show_traces=False, ci_label=None)
    axes[0].set_ylabel(f"Euler angle ({unit})")
    axes[0].legend(loc="upper right", fontsize=7, ncol=3)
    axes[0].set_title(trial_id, fontsize=10)

    # 2-4. EMG envelope, dyn accel PC1, ang rate PC1
    for ax, (col, label, unit) in zip(axes[1:], SIGNALS):
        grid, curves = curves_by_col[col]
        plot_overlay(ax, grid, curves)
        ax.set_ylabel(f"{label} ({unit})")
        ax.legend(loc="upper right", fontsize=7)


def make_group_figure(group, out_path):
    """One figure, 2 x 2 quadrants, 4 stacked subplots per quadrant."""
    fig = plt.figure(figsize=(16, 13), constrained_layout=True)
    fig.suptitle("Sit-to-stand cycle overlays: gain-1 PC recordings "
                 "(Karl/Max x BLE/USB)")
    outer = fig.add_gridspec(2, 2)
    for i, (title, trial) in enumerate(group.items()):
        row, col = divmod(i, 2)
        inner = outer[row, col].subgridspec(4, 1)
        axes = [fig.add_subplot(inner[j]) for j in range(4)]
        trial_id, signals_path, emg_path = trial
        loaded = load_trial(signals_path, emg_path)
        if loaded is None:
            for ax in axes:
                ax.text(0.5, 0.5, f"{trial_id}: no transition times",
                        ha="center", va="center", transform=ax.transAxes)
            continue
        meta, signals, emg = loaded
        times = parse_transition_times(meta)
        cycles = list(zip(times[::2], times[1::2]))
        columns = [c for c, _, _ in EULER_SIGNALS] + [c for c, _, _ in SIGNALS]
        curves_by_col = compute_curves(meta, signals, emg, cycles, columns)
        plot_quadrant(axes, trial_id, curves_by_col)
        print(f"{trial_id}: {len(cycles)} cycles")
    for ax in fig.get_axes():
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel("% of sit-to-stand cycle (0 = start of standing up)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=BASE,
                    help="Root folder to search for the trial CSVs")
    ap.add_argument("--out-dir",
                    default=os.path.dirname(os.path.abspath(__file__)),
                    help="Folder for the figure and the plot-data CSV")
    ap.add_argument("--source", choices=("PC", "SD"), default="PC",
                    help="Recording source (default: PC)")
    ap.add_argument("--gain", type=int, default=1,
                    help="Recording gain (default: 1)")
    ap.add_argument("--no-plot", action="store_true",
                    help="Do not open an interactive plot window")
    args = ap.parse_args(argv)

    if plt is None:
        raise SystemExit("matplotlib is required for the plotter")

    group = find_group_trials(args.base, args.source, args.gain)
    if not group:
        raise SystemExit(f"No matching gain-{args.gain} {args.source} "
                         f"Sit_Stand trials found under {args.base}")
    print(f"Found {len(group)} group trials: {', '.join(group)}")

    fig_path = os.path.join(args.out_dir, "sit_stand_group_overlays.png")
    plot_rows = []
    for title, (trial_id, signals_path, emg_path) in group.items():
        meta, signals, emg = load_trial(signals_path, emg_path)
        times = parse_transition_times(meta)
        cycles = list(zip(times[::2], times[1::2]))
        columns = [c for c, _, _ in EULER_SIGNALS] + [c for c, _, _ in SIGNALS]
        curves_by_col = compute_curves(meta, signals, emg, cycles, columns)
        for col in columns:
            grid, curves = curves_by_col[col]
            if curves is None:
                continue
            n = curves.shape[0]
            mean = curves.mean(axis=0)
            ci = (1.645 * curves.std(axis=0, ddof=1) / np.sqrt(n)
                  if n > 1 else np.zeros_like(mean))
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

    make_group_figure(group, fig_path)

    plot_data = pd.DataFrame(plot_rows)
    csv_path = os.path.join(args.out_dir, "group_plot_data.csv")
    plot_data.to_csv(csv_path, index=False)

    print(f"written: {fig_path}")
    print(f"written: {csv_path}  ({len(plot_data)} rows)")

    if not args.no_plot:
        plt.show()


if __name__ == "__main__":
    main()