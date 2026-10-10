#!/usr/bin/env python3
"""
Sit-to-stand GROUP overlay plotter: one figure comparing BLE and USB
directly, built from the same per-trial pipeline CSVs as
plot_sit_stand_trials.py:

    <stem>_signals.csv  - 50 Hz IMU/PCA grid + '# key: value' header comments
    <stem>_emg.csv      - 1000 Hz EMG envelope (µV) + header comments

Figure layout (2 x 2 quadrants). Every cell overlays the USB and BLE
gain-1 PC recordings of one subject, mean + 90 % CI only (no individual
cycle traces):

    top-left:     Karl  Euler angles (gain 1)    top-right:  Max  Euler angles (gain 1)
    bottom-left:  Karl                           bottom-right: Max
        2. EMG ch1 envelope (µV)                   2. EMG ch1 envelope (µV)
        3. dynamic accel PC1 (g)                   3. dynamic accel PC1 (g)
        4. angular-rate PC1 (deg/s)                4. angular-rate PC1 (deg/s)

Colour = connection everywhere (BLE tab:blue, USB tab:orange); the Euler
cell keeps the angles apart by line style (roll solid, pitch dashed, yaw
dotted); each of signals 2-4 keeps its own subplot with the USB and BLE
curves overlaid.

Also writes group_overlay_plot_data.csv with exactly the plotted numbers
(trial, signal, cycle %, mean, CI low/high, n_cycles).

Usage:
    python plot_sit_stand_group_overlay.py                 # default: PC, gain 1
    python plot_sit_stand_group_overlay.py --out-dir DIR   # custom output folder
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
    SIGNALS,
    cycle_curves,
    find_trials,
    load_trial,
    parse_transition_times,
    read_header_comments,
)

SUBJECTS = ("Karl", "Max")
CONNECTIONS = ("BLE", "USB")
CONNECTION_COLORS = {"BLE": "tab:blue", "USB": "tab:orange"}

EULER_SIGNALS = [
    ("roll_deg", "roll", "deg"),
    ("pitch_deg", "pitch", "deg"),
    ("yaw_deg", "yaw", "deg"),
]
EULER_STYLES = {"roll_deg": "-", "pitch_deg": "--", "yaw_deg": ":"}

CYCLE_XLABEL = "% of sit-to-stand cycle (0 = start of standing up)"


def find_subject_trials(base, source="PC", gain=1):
    """The gain-1 trials, keyed (subject, connection); rep 1 == gain 1,
    PC source only, '# gain' matching in the EMG header.

    Missing (subject, connection) combinations are simply absent."""
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
        if subject not in SUBJECTS or connection not in CONNECTIONS:
            continue
        by_key[(subject, connection)] = (trial_id, signals_path, emg_path)
    return by_key


def compute_curves(meta, signals, emg, cycles, columns):
    """Cycle-locked curves for the requested columns. Returns
    {column: (grid, curves)}; IMU columns are shifted by imu_t0_s into the
    EMG sample-time base, exactly like plot_sit_stand_group.py."""
    t_imu0 = float(meta.get("imu_t0_s", "0"))
    out = {}
    for col in columns:
        if col == "emg_envelope_uV":
            t, y = emg["t_s"].values, emg[col].values
        else:
            t, y = signals["t_s"].values + t_imu0, signals[col].values
        out[col] = cycle_curves(t, y, cycles)
    return out


def mean_and_ci(curves):
    """Mean curve, 90 % CI half-width, and cycle count."""
    n = curves.shape[0]
    mean = curves.mean(axis=0)
    ci = (1.645 * curves.std(axis=0, ddof=1) / np.sqrt(n)
          if n > 1 else np.zeros_like(mean))
    return mean, ci, n


def draw_euler_cell(ax, curves_by_conn, title):
    """Euler means + 90 % CI for BLE and USB on one axes: colour =
    connection, line style = angle; no individual cycle traces."""
    n_plotted = 0
    for conn in CONNECTIONS:
        data = curves_by_conn.get(conn)
        if not data:
            continue
        color = CONNECTION_COLORS[conn]
        ci_labeled = False
        for col, label, _unit in EULER_SIGNALS:
            grid, curves = data[col]
            if curves is None:
                continue
            n_plotted += 1
            mean, ci, n = mean_and_ci(curves)
            if ci.any():
                ax.fill_between(grid, mean - ci, mean + ci, color=color,
                                alpha=0.15, lw=0,
                                label=f"{conn} 90% CI"
                                if not ci_labeled else None)
                ci_labeled = True
            ax.plot(grid, mean, color=color, lw=1.3, ls=EULER_STYLES[col],
                    label=f"{conn} {label} (n={n})")
    if n_plotted == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
    else:
        ax.legend(loc="center right", fontsize=7, ncol=2)
    ax.set_ylabel("Euler angle (deg)")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(CYCLE_XLABEL)
    ax.grid(True, alpha=0.3)


def draw_signal_cell(axes, curves_by_conn, title):
    """3 stacked subplots (EMG, accel PC1, rate PC1); the USB and BLE
    means + 90 % CI are overlaid in each, colour = connection."""
    for ax, (col, label, unit) in zip(axes, SIGNALS):
        n_plotted = 0
        for conn in CONNECTIONS:
            data = curves_by_conn.get(conn)
            if not data:
                continue
            grid, curves = data[col]
            if curves is None:
                continue
            n_plotted += 1
            mean, ci, n = mean_and_ci(curves)
            color = CONNECTION_COLORS[conn]
            if ci.any():
                ax.fill_between(grid, mean - ci, mean + ci, color=color,
                                alpha=0.25, lw=0, label=f"{conn} 90% CI")
            ax.plot(grid, mean, color=color, lw=1.3,
                    label=f"{conn} mean (n={n})")
        if n_plotted == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
        else:
            ax.legend(loc="upper center", fontsize=7)
        ax.set_ylabel(f"{label} ({unit})")
        ax.grid(True, alpha=0.3)
    axes[0].set_title(title, fontsize=10)
    axes[-1].set_xlabel(CYCLE_XLABEL)


def make_overlay_figure(subject_curves, gain, suptitle, out_path):
    """One figure, 2 x 2 quadrants: euler cell on top, 3 signal
    subplots below; Karl left, Max right."""
    fig = plt.figure(figsize=(16, 13), constrained_layout=True)
    fig.suptitle(suptitle)
    outer = fig.add_gridspec(2, 2, height_ratios=[1.0, 2.75])
    for col, subject in enumerate(SUBJECTS):
        curves_by_conn = subject_curves.get(subject, {})
        ax = fig.add_subplot(outer[0, col])
        draw_euler_cell(ax, curves_by_conn,
                        f"{subject}: Euler angles (gain {gain})")
        inner = outer[1, col].subgridspec(3, 1)
        axes = [fig.add_subplot(inner[j]) for j in range(3)]
        draw_signal_cell(axes, curves_by_conn,
                         f"{subject}: BLE vs USB overlays (gain {gain})")
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

    group = find_subject_trials(args.base, args.source, args.gain)
    if not group:
        raise SystemExit(f"No matching gain-{args.gain} {args.source} "
                         f"Sit_Stand trials found under {args.base}")
    print(f"Found {len(group)} trials: "
          + ", ".join(t for t, _, _ in group.values()))

    subject_curves = {}
    plot_rows = []
    for (subject, connection), (trial_id, signals_path,
                                emg_path) in group.items():
        loaded = load_trial(signals_path, emg_path)
        if loaded is None:
            print(f"{trial_id}: no transition times, skipping")
            continue
        meta, signals, emg = loaded
        times = parse_transition_times(meta)
        cycles = list(zip(times[::2], times[1::2]))
        columns = [c for c, _, _ in EULER_SIGNALS] + [c for c, _, _ in SIGNALS]
        curves_by_col = compute_curves(meta, signals, emg, cycles, columns)
        subject_curves.setdefault(subject, {})[connection] = curves_by_col
        print(f"{trial_id}: {len(cycles)} cycles")
        for col in columns:
            grid, curves = curves_by_col[col]
            if curves is None:
                continue
            n = curves.shape[0]
            mean, ci, _n = mean_and_ci(curves)
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

    for subject in SUBJECTS:
        present = subject_curves.get(subject, {})
        missing = [c for c in CONNECTIONS if c not in present]
        if missing:
            print(f"WARNING: {subject}: missing {', '.join(missing)} data")

    fig_path = os.path.join(args.out_dir, "sit_stand_group_overlay.png")
    suptitle = (f"Sit-to-stand cycle overlays: gain-{args.gain} "
                f"{args.source} recordings - BLE vs USB per subject")
    make_overlay_figure(subject_curves, args.gain, suptitle, fig_path)

    plot_data = pd.DataFrame(plot_rows)
    csv_path = os.path.join(args.out_dir, "group_overlay_plot_data.csv")
    plot_data.to_csv(csv_path, index=False)

    print(f"written: {fig_path}")
    print(f"written: {csv_path}  ({len(plot_data)} rows)")

    if not args.no_plot:
        plt.show()


if __name__ == "__main__":
    main()
