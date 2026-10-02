#!/usr/bin/env python3
"""
PCA-based sagittal-plane isolation for Sokosti IMU walking captures.

============================================================================
DISCLAIMER - accelerometer scaling in captures recorded before 2026-10-02
============================================================================
Captures recorded before the 2026-10-02 fix in sokosti/protocol.py stored
the BHI360 accelerometer with a WRONG conversion (raw/16 instead of the
correct raw/4096, per the Bosch BHY2-Sensor-API default scaling table).
In those files the accel_x/y/z columns are NOT g: 1 g reads as ~256.
The columns also include gravity: the firmware streams sensor ID 4, the
*corrected accelerometer*, not linear acceleration (that would be ID 31).

This script auto-detects the legacy scale (median |accel| >> 1 g) and
rescales by 1/256 so everything downstream is in g. Roll/pitch/yaw are
derived from the quaternion (Q14, /16384) and were always correct.
Override auto-detection with --accel-unit {auto,g,legacy}.

Pipeline
--------
1. Load (sample, roll, pitch, yaw, accel_x/y/z) from the capture CSV.
2. Resample the 50 Hz zero-order-held IMU streams onto a uniform grid.
3. Filter:
   - 4th-order zero-phase Butterworth low-pass (default 10 Hz) on angles.
   - Gravity removal: 4th-order zero-phase Butterworth high-pass
     (default 0.2 Hz) on each accel axis -> dynamic acceleration.
4. PCA:
   - Angular-rate PCA on [d(roll)/dt, d(pitch)/dt, d(yaw)/dt]: PC1 is the
     dominant rotation axis. For sagittal-dominant walking PC1 should align
     with the pitch axis and explain >~0.8 of the variance.
   - Dynamic-accel PCA: PC1/PC2 should span the sagittal (forward+vertical)
     plane; a small PC3 fraction means the motion is mostly planar.
5. Project, fix the arbitrary PCA sign (largest-magnitude peak positive),
   optionally plot and export the processed signals.

Usage
-----
python pca_sagittal_plane_isolation.py CAPTURE.csv
python pca_sagittal_plane_isolation.py CAPTURE.csv --fit-start 10 --fit-end 60
python pca_sagittal_plane_isolation.py CAPTURE.csv --no-plot --save-csv out.csv
"""

import argparse

import numpy as np
from scipy.signal import butter, filtfilt

try:
    import matplotlib.pyplot as plt
except ImportError:  # plotting is optional
    plt = None

IMU_COLS = ("roll", "pitch", "yaw", "accel_x", "accel_y", "accel_z")
ANGLE_NAMES = ("roll(x)", "pitch(y)", "yaw(z)")
# Legacy CSVs stored raw/16; correct is raw/4096 g -> stored/256 = g.
LEGACY_TO_G = 1.0 / 256.0


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
def pca(X):
    """
    X: (N, k) observation matrix (rows = samples), e.g. a 3-axis signal.

    Returns (components, explained):
      components: (k, k), PC1 in column 0, ordered by descending eigenvalue.
      explained:  variance fraction per component (sums to 1).
    """
    X = X - X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)   # ascending order
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    return eigvecs, eigvals / eigvals.sum()


def fix_sign(axis, signal):
    """Flip ``axis``/``signal`` so the largest-magnitude peak is positive."""
    if abs(signal.min()) > abs(signal.max()):
        return -axis, -signal
    return axis, signal


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def butter_filter(x, fs, fc, order, btype):
    """Zero-phase Butterworth filter (filtfilt). x: 1-D array."""
    nyq = fs / 2.0
    if not 0.0 < fc < nyq:
        raise SystemExit(f"Filter cutoff {fc} Hz must be between 0 and Nyquist {nyq} Hz")
    b, a = butter(order, fc / nyq, btype=btype)
    padlen = min(3 * max(len(a), len(b)), x.size - 1)
    if padlen < 1:
        raise SystemExit("Signal too short to filter")
    return filtfilt(b, a, x, padtype="odd", padlen=padlen)


def unwrap_deg(x_deg):
    """Unwrap an angle series given in degrees (handles +/-180 deg wraps)."""
    return np.rad2deg(np.unwrap(np.deg2rad(x_deg)))


# ---------------------------------------------------------------------------
# Loading / resampling
# ---------------------------------------------------------------------------
def load_capture(path):
    """Load the CSV and return (sample_idx, imu) with imu = (N, 6) columns
    roll, pitch, yaw, accel_x, accel_y, accel_z. Rows before the first IMU
    packet (all-NaN IMU cells) are dropped; isolated NaNs are interpolated."""
    raw = np.genfromtxt(path, delimiter=",", names=True)
    if raw.ndim == 0:
        raise SystemExit(f"No data rows found in {path}")
    names = raw.dtype.names or ()
    missing = [c for c in ("sample",) + IMU_COLS if c not in names]
    if missing:
        raise SystemExit(f"Missing columns in {path}: {missing}")

    sample = np.asarray(raw["sample"], dtype=float)
    imu = np.column_stack([np.asarray(raw[c], dtype=float) for c in IMU_COLS])

    valid = ~np.all(np.isnan(imu), axis=1)
    if not valid.any():
        raise SystemExit(f"No IMU rows found in {path}")
    sample, imu = sample[valid], imu[valid]

    for j in range(imu.shape[1]):  # fill any isolated gaps
        col = imu[:, j]
        bad = np.isnan(col)
        if bad.any() and (~bad).any():
            col[bad] = np.interp(sample[bad], sample[~bad], col[~bad])
    return sample, imu


def normalize_accel_units(imu, mode):
    """Convert legacy accel scaling to g if needed. Returns (imu, converted)."""
    mag = np.median(np.linalg.norm(imu[:, 3:6], axis=1))
    if mode == "auto":
        legacy = mag > 8.0  # real data is ~1 g; legacy 1 g reads as ~256
    else:
        legacy = mode == "legacy"

    if legacy:
        print("=" * 74)
        print("DISCLAIMER: legacy accelerometer scaling detected "
              f"(median |accel| = {mag:.1f}).")
        print("This capture was recorded before the 2026-10-02 protocol.py fix")
        print("(raw/16 instead of raw/4096). Converting accel_x/y/z to g by 1/256.")
        print("Roll/pitch/yaw are unaffected. New captures are already in g.")
        print("=" * 74)
        imu = imu.copy()
        imu[:, 3:6] *= LEGACY_TO_G
    else:
        print(f"Accelerometer scale: treating as g (median |accel| = {mag:.3f} g)")
    return imu, legacy


def resample_uniform(sample, imu, emg_fs, imu_fs):
    """Resample the zero-order-held IMU streams onto a uniform imu_fs grid.
    Time is zero-based at the first IMU row. Returns (t, imu_resampled)."""
    t_raw = sample / emg_fs
    t_raw = t_raw - t_raw[0]
    n = int(round(t_raw[-1] * imu_fs)) + 1
    t = np.arange(n) / imu_fs
    out = np.column_stack([np.interp(t, t_raw, imu[:, j]) for j in range(imu.shape[1])])
    return t, out


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze(t, imu, args):
    """Run the full sagittal-plane analysis. Returns a dict of signals/stats."""
    fs = args.imu_fs
    dt = 1.0 / fs

    roll_f = butter_filter(imu[:, 0], fs, args.lp_fc, args.order, "low")
    pitch_f = butter_filter(imu[:, 1], fs, args.lp_fc, args.order, "low")
    yaw_f = butter_filter(imu[:, 2], fs, args.lp_fc, args.order, "low")

    # Gravity estimate (before high-pass): should be ~1 g for a static-ish sensor.
    g_vec = imu[:, 3:6].mean(axis=0)
    print(f"Gravity estimate (mean accel): "
          f"[{g_vec[0]:+.3f}, {g_vec[1]:+.3f}, {g_vec[2]:+.3f}] g, "
          f"|g| = {np.linalg.norm(g_vec):.3f} g")

    # Dynamic acceleration: high-pass removes gravity + DC offset.
    acc_dyn = np.column_stack([
        butter_filter(imu[:, 3 + j], fs, args.hp_fc, args.order, "high")
        for j in range(3)
    ])

    # Angular rates from filtered, unwrapped Euler angles (deg/s).
    rate = np.column_stack([
        np.gradient(unwrap_deg(a), dt) for a in (roll_f, pitch_f, yaw_f)
    ])

    # PCA fit window (default: whole file). Use --fit-start/--fit-end to
    # exclude standing still / turns from the PCA fit.
    fit = np.ones(t.size, dtype=bool)
    if args.fit_start is not None:
        fit &= t >= args.fit_start
    if args.fit_end is not None:
        fit &= t <= args.fit_end
    if fit.sum() < 10:
        raise SystemExit("PCA fit window has fewer than 10 samples")

    rate_vecs, rate_expl = pca(rate[fit])
    acc_vecs, acc_expl = pca(acc_dyn[fit])

    rate_axis = rate_vecs[:, 0]
    rate_pc1 = rate @ rate_axis
    rate_axis, rate_pc1 = fix_sign(rate_axis, rate_pc1)

    acc_pc1 = acc_dyn @ acc_vecs[:, 0]
    acc_pc2 = acc_dyn @ acc_vecs[:, 1]
    acc_pc3 = acc_dyn @ acc_vecs[:, 2]
    acc_axis1, acc_pc1 = fix_sign(acc_vecs[:, 0], acc_pc1)

    dom_rate = int(np.argmax(np.abs(rate_axis)))
    corr1 = np.corrcoef(rate_pc1, acc_pc1)[0, 1]
    corr2 = np.corrcoef(rate_pc1, acc_pc2)[0, 1]

    print(f"\nAngular-rate PCA (fit window {fit.sum()} samples, "
          f"{t[fit][-1] - t[fit][0]:.1f} s):")
    print(f"  explained variance: {np.round(rate_expl, 3)}")
    print(f"  PC1 axis: {np.round(rate_axis, 3)} -> dominant Euler axis: "
          f"{ANGLE_NAMES[dom_rate]}")
    print(f"  (sagittal-dominant walking: PC1 ~ pitch(y), explained[0] > ~0.8)")

    print(f"\nDynamic-accel PCA:")
    print(f"  explained variance: {np.round(acc_expl, 3)}")
    print(f"  PC1 axis: {np.round(acc_axis1, 3)}")
    print(f"  PC1+PC2 span the dominant movement plane "
          f"({acc_expl[0] + acc_expl[1]:.1%} of variance);")
    print(f"  a small PC3 fraction ({acc_expl[2]:.1%}) means mostly planar "
          f"(sagittal) motion.")

    print(f"\nCorrelation of rate-PC1 with accel-PC1: {corr1:+.3f}, "
          f"with accel-PC2: {corr2:+.3f}")
    print("(accel leads angular rate by ~90 deg in a sinusoidal sense, so a")
    print(" strong |corr| against either PC is expected; check the plot.)")

    return {
        "t": t,
        "roll_f": roll_f, "pitch_f": pitch_f, "yaw_f": yaw_f,
        "rate": rate, "rate_pc1": rate_pc1, "rate_axis": rate_axis,
        "rate_expl": rate_expl,
        "acc_dyn": acc_dyn,
        "acc_pc1": acc_pc1, "acc_pc2": acc_pc2, "acc_pc3": acc_pc3,
        "acc_axis1": acc_axis1, "acc_expl": acc_expl,
        "fit": fit,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save_csv(path, res):
    header = ",".join([
        "t_s", "roll_deg", "pitch_deg", "yaw_deg",
        "droll_dps", "dpitch_dps", "dyaw_dps", "rate_pc1_dps",
        "acc_dyn_x_g", "acc_dyn_y_g", "acc_dyn_z_g",
        "acc_pc1_g", "acc_pc2_g", "acc_pc3_g",
    ])
    data = np.column_stack([
        res["t"], res["roll_f"], res["pitch_f"], res["yaw_f"],
        res["rate"][:, 0], res["rate"][:, 1], res["rate"][:, 2], res["rate_pc1"],
        res["acc_dyn"][:, 0], res["acc_dyn"][:, 1], res["acc_dyn"][:, 2],
        res["acc_pc1"], res["acc_pc2"], res["acc_pc3"],
    ])
    np.savetxt(path, data, delimiter=",", header=header, comments="")
    print(f"Saved processed signals to {path}")


def plot(res, args):
    if plt is None:
        print("matplotlib not available - skipping plot")
        return
    t = res["t"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True,
                             constrained_layout=True)
    fig.suptitle(f"Sagittal-plane isolation: {args.input}")

    ax = axes[0]
    ax.plot(t, res["roll_f"], lw=0.8, label="roll")
    ax.plot(t, res["pitch_f"], lw=0.8, label="pitch")
    ax.plot(t, res["yaw_f"], lw=0.8, label="yaw")
    ax.set_ylabel("angle (deg)")
    ax.legend(loc="upper right", fontsize=8, ncol=3)

    ax = axes[1]
    for j, name in enumerate("xyz"):
        ax.plot(t, res["acc_dyn"][:, j], lw=0.5, alpha=0.5, label=f"dyn a{name}")
    ax.plot(t, res["acc_pc1"], lw=1.2, color="k", label="accel PC1")
    ax.plot(t, res["acc_pc2"], lw=1.2, color="tab:red", label="accel PC2")
    ax.set_ylabel("dyn accel (g)")
    ax.legend(loc="upper right", fontsize=8, ncol=5)

    ax = axes[2]
    for j, name in enumerate(ANGLE_NAMES):
        ax.plot(t, res["rate"][:, j], lw=0.5, alpha=0.5, label=f"d{name}")
    ax.plot(t, res["rate_pc1"], lw=1.2, color="k", label="rate PC1 (sagittal)")
    ax.set_ylabel("ang. rate (deg/s)")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", fontsize=8, ncol=4)

    if args.save_plot:
        fig.savefig(args.save_plot, dpi=150)
        print(f"Saved plot to {args.save_plot}")
    if not args.no_plot:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="PCA-based sagittal-plane isolation from a Sokosti capture CSV "
                    "(roll/pitch/yaw + accel). See module docstring for the "
                    "legacy accelerometer-scaling disclaimer.",
    )
    p.add_argument("input", help="Capture CSV with columns: sample, roll, pitch, "
                                 "yaw, accel_x, accel_y, accel_z")
    p.add_argument("--emg-fs", type=float, default=1000.0,
                   help="CSV row rate in Hz used for the time base (default: 1000)")
    p.add_argument("--imu-fs", type=float, default=50.0,
                   help="IMU update rate in Hz to resample to (default: 50)")
    p.add_argument("--lp-fc", type=float, default=10.0,
                   help="Low-pass cutoff for angles in Hz (default: 10)")
    p.add_argument("--hp-fc", type=float, default=0.2,
                   help="High-pass cutoff for gravity removal in Hz (default: 0.2)")
    p.add_argument("--order", type=int, default=4,
                   help="Butterworth filter order (default: 4)")
    p.add_argument("--accel-unit", choices=("auto", "g", "legacy"), default="auto",
                   help="Accelerometer scaling of the input: auto-detect, already "
                        "in g, or legacy pre-2026-10-02 scale (default: auto)")
    p.add_argument("--fit-start", type=float, default=None,
                   help="PCA fit window start in s (exclude standing/turns)")
    p.add_argument("--fit-end", type=float, default=None,
                   help="PCA fit window end in s")
    p.add_argument("--save-csv", default=None,
                   help="Optional path to save the processed signals")
    p.add_argument("--save-plot", default=None,
                   help="Optional path to save the plot as an image")
    p.add_argument("--no-plot", action="store_true",
                   help="Do not open an interactive plot window")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sample, imu = load_capture(args.input)
    print(f"Loaded {sample.size} IMU-bearing rows from {args.input}")

    imu, _ = normalize_accel_units(imu, args.accel_unit)
    t, imu = resample_uniform(sample, imu, args.emg_fs, args.imu_fs)
    print(f"Resampled to {args.imu_fs} Hz, {t.size} samples, "
          f"{t[-1]:.1f} s duration")

    res = analyze(t, imu, args)
    if args.save_csv:
        save_csv(args.save_csv, res)
    plot(res, args)


if __name__ == "__main__":
    main()

