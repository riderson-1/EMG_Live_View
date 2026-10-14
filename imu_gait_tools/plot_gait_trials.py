#!/usr/bin/env python3
"""
Gait trial plotter: reads the per-trial pipeline CSVs written by
gait_analysis.py --save-trial-csv

    <stem>_signals.csv  - 50 Hz IMU/PCA grid + '# key: value' header comments
                          (PCA statistics, heel strikes, cycle metadata)
    <stem>_emg.csv      - 980 Hz EMG envelope (µV) + header comments

and produces gait-cycle-locked overlay figures (0-100 % of the gait cycle,
individual traces + mean + 90 % CI) for the EMG envelope, the dynamic-accel
PC1 and the angular-rate PC1:

  - one overlay figure per trial  (<trial_id>_cycle_overlays.png)
  - one cross-trial figure with every trial's mean curve on one plot
    (gait_cycle_means.png)
  - gait_plot_data.csv with exactly the plotted numbers
    (trial, signal, cycle %, mean, CI low/high)

With --raw-emg the envelope subplot is replaced by the raw, UNFILTERED
EMG of --channel (read from the capture CSV named by the '# input:'
header comment, codes -> mV, DC offset kept) and one extra figure per
trial, <trial_id>_raw_ch<N>.png, shows that channel as a plain
time series with the heel strikes marked.

Usage:
    python plot_gait_trials.py                 # all Walk trials
    python plot_gait_trials.py --out-dir DIR   # custom output folder
    python plot_gait_trials.py --source PC     # PC recordings only
    python plot_gait_trials.py --walk-type Incline
    python plot_gait_trials.py --csv BLE_8_Max_Fast_Walk_2026-09-14_PC_signals.csv
    python plot_gait_trials.py --raw-emg --channel 3   # raw EMG, no filters
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

# trial folder / file stem, e.g. BLE_1_Karl_Fast_Walk_2026-09-08
DIR_RE = re.compile(
    r"^(?P<connection>USB|BLE)_(?P<rep>\d+)_(?P<subject>[A-Za-z]+)_"
    r"(?P<walk>Fast|Slow|Incline)_Walk_(?P<date>\d{4}-\d{2}-\d{2})$"
)
STEM_RE = re.compile(
    r"^(?P<connection>USB|BLE)_(?P<rep>\d+)_(?P<subject>[A-Za-z]+)_"
    r"(?P<walk>Fast|Slow|Incline)_Walk_(?P<date>\d{4}-\d{2}-\d{2})_"
    r"(?P<source>PC|SD)$"
)

# signals plotted per trial, (column in _signals.csv, label, unit)
SIGNALS = [
    ("emg_envelope_uV", "EMG envelope", "µV"),
    ("acc_pc1_g", "dyn accel PC1", "g"),
    ("rate_pc1_dps", "ang. rate PC1", "deg/s"),
]

N_GRID = 201  # points on the 0-100 % cycle grid
N_GRID_RAW = 2048  # raw EMG keeps its waveform: >= 980 Hz x ~2 s cycle

ADS1299_VREF = 4.5


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


def parse_heel_strikes(meta):
    """'# heel_strikes_s: 1120.5,1121.5,...' -> list of floats, or None."""
    s = meta.get("heel_strikes_s")
    if not s:
        return None
    return [float(x) for x in s.split(",") if x.strip()]


def load_trial(signals_path, emg_path):
    """Load one trial's CSVs. Returns (meta, signals_df, emg_df) or None if
    the trial has no heel strikes (nothing to overlay)."""
    meta = read_header_comments(signals_path)
    strikes = parse_heel_strikes(meta)
    if not strikes or len(strikes) < 2:
        return None
    signals = pd.read_csv(signals_path, comment="#")
    emg = pd.read_csv(emg_path, comment="#")
    return meta, signals, emg


def cycle_curves(t, y, cycles, time_axis="percent", n_grid=N_GRID):
    """Interpolate a signal onto the cycle grid for every cycle.

    cycles: list of (start_s, end_s). Returns (grid, curves) where curves is
    (n_cycles, n_grid); cycles with <10 valid samples are skipped.
    n_grid lets the raw-EMG path use a much denser grid so the waveform
    survives (default 201 keeps the other plotters' behaviour)."""
    if time_axis == "percent":
        grid = np.linspace(0.0, 100.0, n_grid)
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


def make_trial_figure(trial_id, curves_by_signal, out_path, signals=None):
    """Overlay figure for one trial: individual traces (grey dashed) +
    mean (black) + 90 % CI, one subplot per signal. signals defaults to
    the module-level SIGNALS list."""
    signals = SIGNALS if signals is None else signals
    fig, axes = plt.subplots(len(signals), 1, figsize=(11, 3 * len(signals)),
                             sharex=True, constrained_layout=True)
    fig.suptitle(f"Gait-cycle overlays: {trial_id}")
    for ax, (col, label, unit) in zip(axes, signals):
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
    axes[-1].set_xlabel("% of gait cycle (0 = heel strike)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_means_figure(means_by_signal, out_path, signals=None):
    """Cross-trial figure: every trial's mean curve on one subplot per
    signal (one colour per trial). signals defaults to SIGNALS."""
    signals = SIGNALS if signals is None else signals
    fig, axes = plt.subplots(len(signals), 1, figsize=(11, 3 * len(signals)),
                             sharex=True, constrained_layout=True)
    fig.suptitle("Gait-cycle mean curves across trials")
    for ax, (col, label, unit) in zip(axes, signals):
        for (trial_id, grid, mean) in means_by_signal[col]:
            ax.plot(grid, mean, lw=1.1, label=trial_id)
        ax.set_ylabel(f"{label} ({unit})")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("% of gait cycle (0 = heel strike)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def find_trials(base, source=None, walk_type=None):
    """All Walk trials with both pipeline CSVs. Returns a list of
    (trial_id, signals_path, emg_path)."""
    trials = []
    for dirpath, dirnames, filenames in os.walk(base):
        if not DIR_RE.match(os.path.basename(dirpath)):
            continue
        for fn in sorted(filenames):
            m = (STEM_RE.match(fn[: -len("_signals.csv")])
                 if fn.endswith("_signals.csv") else None)
            if not m:
                continue
            if source and m.group("source") != source:
                continue
            if walk_type and m.group("walk") != walk_type:
                continue
            stem = m.group(0)
            signals_path = os.path.join(dirpath, stem + "_signals.csv")
            emg_path = os.path.join(dirpath, stem + "_emg.csv")
            if os.path.exists(emg_path):
                trials.append((stem, signals_path, emg_path))
    return sorted(trials)


def resolve_signals_csv(path):
    """Map a user-supplied CSV onto its <stem>_signals.csv file.

    Accepts the _signals.csv itself, its _emg.csv sibling, or the raw
    capture CSV (<stem>.csv); a bare filename is searched under BASE.
    Returns (signals_path, emg_path)."""
    path = os.path.expanduser(path)
    candidates = []
    if path.endswith("_signals.csv"):
        candidates.append(path)
    elif path.endswith("_emg.csv"):
        candidates.append(path[: -len("_emg.csv")] + "_signals.csv")
    elif path.endswith(".csv"):
        candidates.append(path[: -len(".csv")] + "_signals.csv")
    else:
        candidates.append(path + "_signals.csv")
    # bare filename -> search BASE for it
    if not os.path.isabs(path):
        candidates += [os.path.join(dirpath, os.path.basename(c))
                       for dirpath, _, _ in os.walk(BASE)
                       for c in [os.path.basename(candidates[0])]]
    for cand in candidates:
        if os.path.exists(cand):
            emg = cand[: -len("_signals.csv")] + "_emg.csv"
            if not os.path.exists(emg):
                raise SystemExit(f"Missing companion EMG CSV: {emg}")
            stem = os.path.basename(cand)[: -len("_signals.csv")]
            if not STEM_RE.match(stem):
                raise SystemExit(
                    f"{cand}: filename does not look like a trial "
                    f"<stem>_signals.csv (e.g. BLE_8_Max_Fast_Walk_"
                    f"2026-09-14_PC_signals.csv)")
            return cand, emg
    raise SystemExit(f"No such trial CSV: {path}")


def load_raw_emg(meta, emg_path, channel, cycles):
    """Raw, UNFILTERED ch<channel> over the cycle time span, read from the
    capture CSV named by the '# input:' header comment (no filters, no
    envelope, no rectification; codes -> µV -> mV, DC offset kept).

    Returns {"t", "y", "gap_rows"} or None when the capture/channel is
    missing. gap_rows marks the first row after each sample-counter gap so
    the plotter can break the line instead of drawing across missing data."""
    capture = meta.get("input")
    emg_fs = float(meta.get("emg_fs_hz", "980"))
    if channel < 1:
        raise SystemExit("--channel must be >= 1")
    if not capture or not os.path.exists(capture):
        print(f"WARNING: raw EMG capture not found: {capture!r}")
        return None
    # scale from the _emg.csv header (gain/uv_per_code of this trial)
    emg_meta = read_header_comments(emg_path)
    if "uv_per_code" in emg_meta:
        uv_per_code = float(emg_meta["uv_per_code"])
    else:
        uv_per_code = (ADS1299_VREF / float(emg_meta.get("gain", "8"))
                       / (2 ** 23) * 1e6)

    ch_col = f"ch{channel}"
    try:
        df = pd.read_csv(capture, usecols=lambda c: c in ("sample", ch_col))
    except Exception as exc:
        print(f"WARNING: reading raw EMG from {capture} failed: {exc}")
        return None
    if "sample" not in df.columns or ch_col not in df.columns:
        print(f"WARNING: {capture}: no raw column {ch_col}")
        return None

    sample_all = df["sample"].to_numpy(np.int64)
    t_all = sample_all / emg_fs
    lo_t = min(a for a, _ in cycles)
    hi_t = max(b for _, b in cycles)
    inside = (t_all >= lo_t) & (t_all <= hi_t)
    if not inside.any():
        print(f"WARNING: {capture}: no ch{channel} samples inside the cycles")
        return None
    idx = np.flatnonzero(inside)
    lo, hi = int(idx[0]), int(idx[-1]) + 1

    sample = sample_all[lo:hi]
    gap_rows = np.flatnonzero(np.diff(sample) != 1) + 1
    y = df[ch_col].to_numpy(np.float64)[lo:hi] * uv_per_code / 1000.0
    return {"t": sample / emg_fs, "y": y, "gap_rows": gap_rows}


def make_raw_timeseries_figure(trial_id, raw, strikes, channel, out_path):
    """Plain raw-EMG time series (no cycle locking, no filters): the chosen
    channel over the full cycle time span, with the heel strikes marked."""
    t, y, gap_rows = raw["t"], raw["y"], raw["gap_rows"]
    y = y.copy()
    y[gap_rows] = np.nan  # break the line at sample-counter gaps
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    fig.suptitle(f"Raw EMG ch{channel} (unfiltered): {trial_id}")
    ax.plot(t, y, lw=0.5, color="tab:blue")
    for k, s in enumerate(strikes):
        ax.axvline(s, color="tab:red", lw=0.8, ls="--", alpha=0.6,
                   label="Heel strike" if k == 0 else None)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"EMG ch{channel} raw (mV)")
    ax.grid(True, alpha=0.3)
    if strikes:
        ax.legend(loc="upper right", fontsize=8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=BASE,
                    help="Root folder to search for the trial CSVs")
    ap.add_argument("--out-dir", default=OUT_DIR,
                    help="Folder for the figures and the plot-data CSV")
    ap.add_argument("--source", choices=("PC", "SD"), default=None,
                    help="Only plot this recording source (default: all)")
    ap.add_argument("--walk-type", choices=("Fast", "Slow", "Incline"),
                    default=None,
                    help="Only plot this walk type (default: all)")
    ap.add_argument("--csv", default=None, metavar="CSV",
                    help="Plot only this trial: path to its "
                         "<stem>_signals.csv (the sibling _emg.csv is "
                         "derived; the raw capture CSV and bare filenames "
                         "are also accepted)")
    ap.add_argument("--raw-emg", action="store_true",
                    help="Replace the EMG-envelope subplot with the raw, "
                         "UNFILTERED EMG of --channel (no bandpass, no "
                         "notch, no envelope) and also write a plain raw "
                         "time-series figure <trial>_raw_ch<N>.png")
    ap.add_argument("--channel", type=int, default=1,
                    help="1-based raw EMG channel (ch1, ch2, ...) used "
                         "with --raw-emg (default: 1)")
    ap.add_argument("--no-plot", action="store_true",
                    help="Do not open an interactive plot window")
    args = ap.parse_args(argv)

    if plt is None:
        raise SystemExit("matplotlib is required for the plotter")

    if args.csv:
        signals_path, emg_path = resolve_signals_csv(args.csv)
        trials = [(os.path.basename(signals_path)
                   [: -len("_signals.csv")], signals_path, emg_path)]
    else:
        trials = find_trials(args.base, args.source, args.walk_type)
    if not trials:
        raise SystemExit(f"No *_signals.csv/*_emg.csv pairs found under {args.base}")
    print(f"Found {len(trials)} trials")

    # signal list for this run: raw EMG replaces the envelope entry
    if args.raw_emg:
        raw_col = f"emg_raw_ch{args.channel}_mV"
        signals = [(raw_col, f"EMG ch{args.channel} raw", "mV"),
                   *SIGNALS[1:]]
    else:
        raw_col = None
        signals = SIGNALS

    plot_rows = []
    means_by_signal = {col: [] for col, _, _ in signals}
    for trial_id, signals_path, emg_path in trials:
        loaded = load_trial(signals_path, emg_path)
        if loaded is None:
            print(f"skipped (no heel strikes): {trial_id}")
            continue
        meta, signals_df, emg = loaded
        strikes = parse_heel_strikes(meta)
        cycles = list(zip(strikes[:-1], strikes[1:]))
        print(f"{trial_id}: {len(cycles)} cycles")

        # EMG time is absolute (sample/fs); IMU time is zero-based at the
        # first IMU row -> shift it into the EMG sample-time base using the
        # imu_t0_s header comment.
        t_imu0 = float(meta.get("imu_t0_s", "0"))

        raw = None
        if args.raw_emg:
            raw = load_raw_emg(meta, emg_path, args.channel, cycles)
            if raw is None:
                print(f"skipped (no raw EMG): {trial_id}")
                continue

        curves_by_signal = {}
        for col, _, _ in signals:
            if col == "emg_envelope_uV":
                t, y = emg["t_s"].values, emg[col].values
                grid, curves = cycle_curves(t, y, cycles)
            elif col == raw_col:
                grid, curves = cycle_curves(raw["t"], raw["y"], cycles,
                                            n_grid=N_GRID_RAW)
            else:
                t, y = signals_df["t_s"].values + t_imu0, signals_df[col].values
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
                          os.path.join(args.out_dir,
                                       trial_id + "_cycle_overlays.png"),
                          signals=signals)

        if args.raw_emg:
            raw_png = os.path.join(args.out_dir,
                                   f"{trial_id}_raw_ch{args.channel}.png")
            make_raw_timeseries_figure(trial_id, raw, strikes,
                                       args.channel, raw_png)
            print(f"written: {raw_png}")

    means_path = os.path.join(args.out_dir, "gait_cycle_means.png")
    make_means_figure(means_by_signal, means_path, signals=signals)

    plot_data = pd.DataFrame(plot_rows)
    csv_path = os.path.join(args.out_dir, "gait_plot_data.csv")
    plot_data.to_csv(csv_path, index=False)

    print(f"written: {means_path}")
    print(f"written: {csv_path}  ({len(plot_data)} rows)")

    if not args.no_plot:
        plt.show()


if __name__ == "__main__":
    main()