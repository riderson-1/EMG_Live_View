#!/usr/bin/env python3
"""
Crosstalk and baseline-noise analysis pipeline for a 16-channel ADS1299-based
EMG/EEG acquisition system.

Produces publication-quality figures and tables for two benchtop tests:
  - Baseline test (all channels shorted): noise floor, SNR helper
  - Crosstalk test (one channel driven at a time): crosstalk matrix, distance dependence

Usage:
    python3 crosstalk_analysis.py

All paths and parameters are defined as constants at the top of the file.
Edit only the relevant function if changes are requested later.
"""

import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal as sp_signal
from scipy.optimize import least_squares
from scipy.integrate import trapezoid

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================
# CONFIGURATION CONSTANTS (edit these to match your setup)
# ============================================================================

BASELINE_PATH = "/home/karl/Documents/Master_Thesis/Testing/benchtop_testing/breakout_pcb_baseline"
CROSSTALK_PATH = "/home/karl/Documents/Master_Thesis/Testing/benchtop_testing/crosstalk_test"
OUTPUT_DIR = "/home/karl/repos/Sokosti_tools/benchtop_testing_tools/crosstalk_test/outputs"
LAYOUT_PATH = "/home/karl/repos/Sokosti_tools/benchtop_testing_tools/crosstalk_test"

# System parameters
FS = 1000          # Sampling rate [Hz]
GAIN = 1           # PGA gain (1G in folder name)
VREF = 4.5         # Reference voltage [V]
N_CHANNELS = 16    # Total channels
# LSB formula: LSB = 2 * Vref / (gain * 2^24) for ADS1299 (24-bit ADC, differential)
LSB_V = 2.0 * VREF / (GAIN * 2**24)  # [V per count]
LSB_uV = LSB_V * 1e6                         # [uV per count]

# Analysis parameters
SEGMENT_DURATION = 3.0   # [s] – length of each analysis segment (parameter)
SINE_F0 = [10, 100]      # Test frequencies [Hz]
AMPLITUDES_MVPP = [2, 10]  # Test amplitudes [mVpp]

# Crosstalk / noise thresholds
NOISE_CORR_THRESHOLD_DB = 6.0  # [dB] above noise to enable noise correction
SIGNIFICANT_LOSS_FRAC = 0.01   # Flag files losing >1% of samples

# Channel-to-chip mapping: ch 1-8 -> ads1299-1, ch 9-16 -> ads1299-2
CHIP_OF = {ch: ("ads1299-1" if 1 <= ch <= 8 else "ads1299-2") for ch in range(1, 17)}

# Physical layout (left-to-right): 16,15,...,9 | 8,7,...,1
# Physical position index (0-based from left): pos(ch) = 16 - ch
PHYS_POS = {ch: 16 - ch for ch in range(1, 17)}

# Plotting style
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_pc_csv(filepath):
    """Load a PC.csv recording, detect BLE packet gaps via sample counter,
    and return the longest contiguous gap-free segment.

    Returns
    -------
    df_seg : pd.DataFrame – rows of the longest gap-free segment
    gap_info : dict – {n_gaps, max_gap_samples, total_lost_samples, flagged,
                        n_samples_full, n_samples_seg, start_idx, end_idx}
    df_full : pd.DataFrame – original full dataframe
    """
    df_full = pd.read_csv(filepath)
    samples = df_full["sample"].values.astype(np.int64)
    n = len(df_full)

    diffs = np.diff(samples)
    gaps = diffs - 1
    gap_indices = np.where(gaps > 0)[0]
    n_gaps = len(gap_indices)
    total_lost = int(gaps.sum())
    max_gap = int(gaps.max()) if n_gaps > 0 else 0

    if n_gaps == 0:
        start_idx, end_idx = 0, n
    else:
        starts = np.concatenate([[0], gap_indices + 1])
        ends = np.concatenate([gap_indices + 1, [n]])
        lengths = ends - starts
        best = np.argmax(lengths)
        start_idx, end_idx = int(starts[best]), int(ends[best])

    df_seg = df_full.iloc[start_idx:end_idx].reset_index(drop=True)
    flagged = total_lost > SIGNIFICANT_LOSS_FRAC * (samples[-1] - samples[0] + 1)

    gap_info = {
        "n_gaps": n_gaps,
        "max_gap_samples": max_gap,
        "total_lost_samples": total_lost,
        "flagged": flagged,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "n_samples_full": n,
        "n_samples_seg": len(df_seg),
    }
    return df_seg, gap_info, df_full


def parse_filename(filepath):
    """Extract driven channel, frequency, amplitude from PC.csv filename."""
    fname = os.path.basename(filepath)
    m = re.match(r"CH(\d+)_1G_Sine_(\d+)mVpp_(\d+)Hz_PC\.csv", fname)
    if not m:
        raise ValueError(f"Cannot parse filename: {fname}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


# ============================================================================
# SIGNAL CONDITIONING
# ============================================================================

def counts_to_uv(counts):
    """Convert raw ADC counts to microvolts using LSB and gain."""
    return counts * LSB_uV


def remove_dc(signal_uv):
    """Remove the mean (DC offset) from a signal in uV."""
    return signal_uv - np.mean(signal_uv)


def split_into_segments(signal_uv, fs, seg_dur):
    """Split a 1-D signal into equal-length segments of seg_dur seconds.
    Drops any remainder shorter than seg_dur.
    Returns list of (segment_array, start_sample_index) tuples.
    """
    n_seg_samples = int(fs * seg_dur)
    n_full = (len(signal_uv) // n_seg_samples) * n_seg_samples
    if n_full == 0:
        return []
    return [(signal_uv[i:i + n_seg_samples], i) for i in range(0, n_full, n_seg_samples)]


# ============================================================================
# AMPLITUDE EXTRACTION
# ============================================================================

def amplitude_ls_sine_fit(t, y, fs, f0, f_bound_hz=2.0):
    """Primary: 4-parameter LS sine fit with free frequency.

    Model: y(t) = A*sin(2*pi*f*t + phi) + C
    Frequency fit tightly around f0 (+/- f_bound_hz).
    Returns dict with amp_peak_uV, freq_hz, phase_rad, offset_uV.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)

    if np.std(y) < 1e-12:
        return {"amp_peak_uV": np.nan, "freq_hz": np.nan,
                "phase_rad": np.nan, "offset_uV": np.nan}

    def residuals(p):
        A, f, phi, C = p
        if A <= 0 or not (f0 - f_bound_hz <= f <= f0 + f_bound_hz):
            return np.full_like(y, 1e12)
        return y - (A * np.sin(2 * np.pi * f * t + phi) + C)

    A0 = np.std(y) * np.sqrt(2)
    p0 = [A0, f0, 0.0, np.mean(y)]
    bounds = ([0, f0 - f_bound_hz, -np.pi, -np.inf],
              [np.inf, f0 + f_bound_hz, np.pi, np.inf])

    try:
        res = least_squares(residuals, p0, bounds=bounds, method="trf", max_nfev=500,
                            xtol=1e-6, ftol=1e-6, gtol=1e-6)
        return {"amp_peak_uV": float(res.x[0]), "freq_hz": float(res.x[1]),
                "phase_rad": float(res.x[2]), "offset_uV": float(res.x[3])}
    except Exception:
        return {"amp_peak_uV": np.nan, "freq_hz": np.nan,
                "phase_rad": np.nan, "offset_uV": np.nan}


def amplitude_fft_hann(y, fs, f0):
    """Cross-check: Hann-windowed FFT amplitude at bin nearest f0.
    A = 2 * |X[k]| / sum(w). Returns peak amplitude in uV.
    """
    y = np.asarray(y, dtype=float)
    N = len(y)
    if N == 0:
        return np.nan
    w = np.hanning(N)
    spectrum = np.fft.rfft(y * w)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    k = np.argmin(np.abs(freqs - f0))
    sum_w = np.sum(w)
    if sum_w == 0:
        return np.nan
    return float(2.0 * np.abs(spectrum[k]) / sum_w)


def amplitude_rms_sqrt2(y):
    """Sanity check: RMS * sqrt(2) as peak amplitude estimate."""
    y = np.asarray(y, dtype=float)
    return float(np.sqrt(2.0) * np.sqrt(np.mean(y**2)))


# ============================================================================
# NOISE REFERENCE FROM BASELINE
# ============================================================================

def compute_noise_reference(baseline_seg_df, fs, freqs, rms_bands):
    """Compute noise reference from baseline recording.

    For each channel:
      - Hann-FFT amplitude at each test frequency (noise in narrow band)
      - Band-limited RMS noise via PSD integration
      - Noise density via Welch (nV/sqrt(Hz))
    """
    n_seg_samples = int(fs * SEGMENT_DURATION)
    n_full = (len(baseline_seg_df) // n_seg_samples) * n_seg_samples

    noise_ref = {}
    for ch in range(1, N_CHANNELS + 1):
        ch_uV = remove_dc(counts_to_uv(baseline_seg_df[f"ch{ch}"].values.astype(float)))
        segs = split_into_segments(ch_uV, fs, SEGMENT_DURATION)
        if not segs:
            continue

        # Noise amplitude at test frequencies (Hann FFT, median across segments)
        fft_amps = {}
        for f0 in freqs:
            amps = [amplitude_fft_hann(seg, fs, f0) for seg, _ in segs]
            amps = [a for a in amps if np.isfinite(a)]
            fft_amps[f0] = float(np.median(amps)) if amps else np.nan

        # Band-limited RMS via PSD integration
        rms_vals = {}
        for band_name, (f_low, f_high) in rms_bands.items():
            nperseg = min(2048, len(ch_uV))
            freqs_psd, psd = sp_signal.welch(
                ch_uV, fs=fs, window="hann", nperseg=nperseg,
                noverlap=nperseg // 2, scaling="density",
            )
            mask = (freqs_psd >= f_low) & (freqs_psd <= f_high)
            if mask.sum() > 0:
                variance = trapezoid(psd[mask], freqs_psd[mask])
                rms_vals[band_name] = float(np.sqrt(variance))
            else:
                rms_vals[band_name] = np.nan

        # Noise density: Welch scaling='density', convert to nV/sqrt(Hz)
        nperseg = min(2048, len(ch_uV))
        freqs_welch, psd_welch = sp_signal.welch(
            ch_uV, fs=fs, window="hann", nperseg=nperseg,
            noverlap=nperseg // 2, scaling="density",
        )
        noise_density_nv = np.sqrt(psd_welch) * 1e9

        noise_ref[ch] = {
            "fft_amps_uV": fft_amps,
            "rms_bands_uV": rms_vals,
            "welch_freqs_hz": freqs_welch,
            "welch_density_nv": noise_density_nv,
        }
    return noise_ref


# ============================================================================
# CROSSTALK METRICS
# ============================================================================

def compute_crosstalk_matrix(driven_amp_uV, victim_amps_uV_array, noise_amp_uV, threshold_db=6.0):
    """Compute 16x16 crosstalk matrix.

    driven_amp_uV: scalar, amplitude of driven channel (uV peak)
    victim_amps_uV_array: length-16 array, amplitude per victim channel (uV peak)
    noise_amp_uV: scalar, baseline noise amplitude at test frequency (uV peak)

    Returns dict with xt_db, xt_corrected, censored (16x16), detection_limit_db.
    """
    if driven_amp_uV <= 0 or not np.isfinite(driven_amp_uV):
        return {
            "xt_db": np.full((16, 16), np.nan),
            "xt_corrected": np.full((16, 16), np.nan),
            "censored": np.ones((16, 16), dtype=bool),
            "detection_limit_db": np.nan,
        }

    xt_db = 20.0 * np.log10(victim_amps_uV_array / driven_amp_uV)
    xt_db = np.where(np.isfinite(xt_db), xt_db, np.nan)

    detection_limit = 20.0 * np.log10(noise_amp_uV / driven_amp_uV)

    threshold_linear = 10 ** (threshold_db / 20.0)
    can_correct = victim_amps_uV_array >= noise_amp_uV * threshold_linear
    A_ct = np.where(can_correct, np.sqrt(np.maximum(victim_amps_uV_array**2 - noise_amp_uV**2, 0)), np.nan)
    xt_corrected = 20.0 * np.log10(A_ct / driven_amp_uV)
    xt_corrected = np.where(np.isfinite(xt_corrected), xt_corrected, np.nan)

    diag_mask = np.eye(16, dtype=bool)
    censored = (~can_correct) | diag_mask

    return {
        "xt_db": xt_db,
        "xt_corrected": xt_corrected,
        "censored": censored,
        "detection_limit_db": detection_limit,
    }


# ============================================================================
# FIGURE GENERATION
# ============================================================================

def _fig_size(width_in, ratio=0.618):
    return (width_in, width_in * ratio)


def plot_parameter_overview(conditions_info, baseline_duration_s, output_path):
    """Figure 1: Setup and parameter overview table."""
    fig, ax = plt.subplots(figsize=_fig_size(7.0))
    ax.axis("off")

    sys_rows = [
        ["Parameter", "Value"],
        ["Sampling rate fs", f"{FS} Hz"],
        ["PGA gain", f"{GAIN} (1G)"],
        ["Vref", f"{VREF} V"],
        ["LSB formula", "LSB = 2*Vref / (gain * 2^24)"],
        ["LSB value", f"{LSB_V:.4e} V = {LSB_uV:.4f} uV/count"],
        ["Channel count", f"{N_CHANNELS}"],
        ["Channel mapping", "ch 1-8 -> ads1299-1, ch 9-16 -> ads1299-2"],
        ["Segment duration", f"{SEGMENT_DURATION} s"],
        ["File format", "PC.csv (16 EMG channels + sample/status)"],
        ["Baseline duration", f"{baseline_duration_s:.2f} s"],
    ]

    test_rows = [["Condition", "Frequency", "Amplitude", "Files"]]
    for (f0, amp), info in sorted(conditions_info.items()):
        test_rows.append([f"CH{info.get('ch', '?')}", f"{f0} Hz", f"{amp} mVpp", str(info.get("n_files", 0))])

    table_data = sys_rows + [["", ""]] + test_rows
    table = ax.table(cellText=table_data, loc="upper left", cellLoc="left",
                     colWidths=[0.35, 0.25, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    for j in range(len(table_data[0])):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    if len(sys_rows) > 0:
        for j in range(len(table_data[0])):
            table[1, j].set_facecolor("#D9E2F3")
            table[1, j].set_text_props(fontweight="bold")

    ax.set_title("System Parameters and Test Conditions", fontsize=10, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_psd_overlay(noise_ref, crosstalk_results, output_path):
    """Figure 2: PSD overlay for driven, adjacent victim, far victim, baseline."""
    fig, axes = plt.subplots(2, 2, figsize=_fig_size(7.0, 0.85))
    axes = axes.flatten()

    conditions = [(10, 2), (10, 10), (100, 2), (100, 10)]
    for idx, (f0, amp) in enumerate(conditions):
        ax = axes[idx]
        key = (f0, amp)
        if key not in crosstalk_results:
            continue
        res = crosstalk_results[key]
        driven_ch = res["driven_channel"]

        # Adjacent victim: next channel index
        adj_ch = driven_ch + 1 if driven_ch < 16 else driven_ch - 1
        # Far victim: farthest channel on the other chip
        if CHIP_OF[driven_ch] == "ads1299-1":
            far_ch = 16
        else:
            far_ch = 1

        baseline_ch = driven_ch

        labels = {
            driven_ch: f"Driven CH{driven_ch}",
            adj_ch: f"Adjacent CH{adj_ch}",
            far_ch: f"Far CH{far_ch}",
            baseline_ch: f"Baseline CH{baseline_ch}",
        }
        colors = {driven_ch: "red", adj_ch: "blue", far_ch: "green", baseline_ch: "gray"}

        for ch, label in labels.items():
            if ch in noise_ref:
                nr = noise_ref[ch]
                ax.plot(nr["welch_freqs_hz"], nr["welch_density_nv"],
                        color=colors[ch], label=label, linewidth=0.8)

        ax.axvline(f0, color="red", linestyle="--", alpha=0.5, label=f"{f0} Hz")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (nV$^2$/Hz)")
        ax.set_title(f"{f0} Hz, {amp} mVpp – CH{driven_ch} driven", fontsize=8)
        ax.set_xlim([0, 500])
        ax.legend(fontsize=6, loc="upper right")

    plt.suptitle("PSD Overlay: Driven, Victim, and Baseline Channels", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_crosstalk_matrix(crosstalk_results, output_path):
    """Figure 3: Crosstalk matrix heatmap (4 panels, shared color scale)."""
    conditions = [(10, 2), (10, 10), (100, 2), (100, 10)]

    all_vals = []
    for key in conditions:
        if key in crosstalk_results:
            vals = crosstalk_results[key]["xt_corrected"]
            mask = ~crosstalk_results[key]["censored"]
            all_vals.extend(vals[mask].flatten())
    vmin = min(np.min(all_vals), -60) if all_vals else -60
    vmax = max(np.max(all_vals), 0) if all_vals else 0

    fig, axes = plt.subplots(2, 2, figsize=_fig_size(7.0, 0.85))
    axes = axes.flatten()

    for idx, (f0, amp) in enumerate(conditions):
        ax = axes[idx]
        key = (f0, amp)
        if key not in crosstalk_results:
            continue
        res = crosstalk_results[key]
        xt = res["xt_corrected"]
        censored = res["censored"]

        im = ax.imshow(xt, cmap="RdYlGn_r", vmin=vmin, vmax=vmax, aspect="auto")
        # Mask diagonal
        diag = np.eye(16, dtype=bool)
        ax.imshow(np.where(diag, 1, 0), cmap="gray", alpha=0.3, vmin=0, vmax=1, aspect="auto")
        # Hatch censored cells
        for i in range(16):
            for j in range(16):
                if censored[i, j]:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               fill=False, edgecolor="black",
                                               hatch="//", linewidth=0.5))
        # Annotate corner cells
        for i in [0, 7, 15]:
            for j in [0, 7, 15]:
                if not censored[i, j] and np.isfinite(xt[i, j]):
                    ax.text(j, i, f"{xt[i, j]:.1f}", ha="center", va="center", fontsize=6,
                            color="black" if xt[i, j] > (vmin + vmax) / 2 else "white")

        ax.set_xlabel("Victim channel")
        ax.set_ylabel("Driven channel")
        ax.set_title(f"{f0} Hz, {amp} mVpp – XT (dB, corrected)", fontsize=8)
        ax.set_xticks(range(0, 16, 4))
        ax.set_yticks(range(0, 16, 4))
        dl = res["detection_limit_db"]
        if np.isfinite(dl):
            ax.text(0.02, 0.98, f"DL: {dl:.1f} dB", transform=ax.transAxes,
                    fontsize=7, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle("Crosstalk Matrix (noise-corrected, hatched = censored)", fontsize=10, fontweight="bold")
    if all_vals:
        cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.03, pad=0.02)
        cbar.set_label("XT (dB)", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_crosstalk_vs_distance(crosstalk_results, output_path):
    """Figure 4: Crosstalk vs channel distance scatter."""
    conditions = [(10, 2), (10, 10), (100, 2), (100, 10)]
    fig, axes = plt.subplots(1, 2, figsize=_fig_size(7.0, 0.65))

    for col, f0 in enumerate([10, 100]):
        ax = axes[col]
        for amp in AMPLITUDES_MVPP:
            key = (f0, amp)
            if key not in crosstalk_results:
                continue
            res = crosstalk_results[key]
            xt = res["xt_corrected"]
            censored = res["censored"]
            xt_raw = res["xt_db"]

            distances, xt_vals, is_censored, pair_types = [], [], [], []
            for i in range(16):
                for j in range(16):
                    if i == j:
                        continue
                    d = abs((i + 1) - (j + 1))
                    same_chip = CHIP_OF[i + 1] == CHIP_OF[j + 1]
                    if d == 1:
                        pt = "adjacent"
                    elif same_chip:
                        pt = "same-ADS"
                    else:
                        pt = "other-ADS"

                    if censored[i, j]:
                        distances.append(d)
                        xt_vals.append(xt_raw[i, j] if np.isfinite(xt_raw[i, j]) else np.nan)
                        is_censored.append(True)
                    else:
                        distances.append(d)
                        xt_vals.append(xt[i, j])
                        is_censored.append(False)
                    pair_types.append(pt)

            markers = {"adjacent": "o", "same-ADS": "s", "other-ADS": "D"}
            for pt in ["adjacent", "same-ADS", "other-ADS"]:
                idx_pts = [k for k, t in enumerate(pair_types) if t == pt and not is_censored[k] and np.isfinite(xt_vals[k])]
                if idx_pts:
                    ax.scatter([distances[k] for k in idx_pts], [xt_vals[k] for k in idx_pts],
                               marker=markers[pt], s=15, alpha=0.5,
                               label=f"{amp} mVpp, {pt}")

            # Censored points (open markers)
            idx_cens = [k for k, t in enumerate(pair_types) if t == "adjacent" and is_censored[k] and np.isfinite(xt_vals[k])]
            if idx_cens:
                ax.scatter([distances[k] for k in idx_cens], [xt_vals[k] for k in idx_cens],
                           marker="o", s=20, facecolors="none", edgecolors="red", linewidths=0.8,
                           label=f"{amp} mVpp, censored")

            # Mean and max per distance
            for d in range(1, 16):
                vals = [xt_vals[k] for k in range(len(distances))
                        if distances[k] == d and not is_censored[k] and np.isfinite(xt_vals[k])]
                if vals:
                    ax.scatter(d, np.mean(vals), marker="D", s=40, c="black", zorder=5)
                    ax.scatter(d, np.max(vals), marker="v", s=40, c="black", zorder=5, facecolors="none")

            # Detection limit lines
            for a in AMPLITUDES_MVPP:
                k2 = (f0, a)
                if k2 in crosstalk_results:
                    dl = crosstalk_results[k2]["detection_limit_db"]
                    if np.isfinite(dl):
                        ax.axhline(dl, linestyle="--", alpha=0.5, label=f"DL ({a} mVpp): {dl:.1f} dB")

        ax.set_xlabel("Channel distance")
        ax.set_ylabel("Crosstalk (dB)")
        ax.set_title(f"{f0} Hz", fontsize=9)
        ax.set_xticks(range(1, 16))
        ax.legend(fontsize=5.5, loc="lower left")

    plt.suptitle("Crosstalk vs Channel Distance", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_freq_amp_dependence(crosstalk_results, output_path):
    """Figure 5: Frequency and amplitude dependence of crosstalk."""
    conditions = [(10, 2), (10, 10), (100, 2), (100, 10)]
    fig, axes = plt.subplots(1, 2, figsize=_fig_size(7.0, 0.6))

    cond_means = {}
    for key in conditions:
        if key not in crosstalk_results:
            continue
        res = crosstalk_results[key]
        vals = res["xt_corrected"][~res["censored"]]
        vals = vals[np.isfinite(vals)]
        cond_means[key] = np.mean(vals) if len(vals) > 0 else np.nan

    # Frequency dependence
    ax = axes[0]
    for amp in AMPLITUDES_MVPP:
        y = [cond_means.get((10, amp), np.nan), cond_means.get((100, amp), np.nan)]
        x = np.array([10, 100])
        ax.plot(x, y, "o-", label=f"{amp} mVpp")
        if np.isfinite(y[0]) and np.isfinite(y[1]):
            ax.annotate(f"{y[1]-y[0]:+.1f} dB", (x[1], y[1]), textcoords="offset points",
                        xytext=(5, 5), fontsize=7)
    ax.set_xscale("log")
    ax.set_xticks([10, 100])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean crosstalk (dB)")
    ax.set_title("Frequency dependence", fontsize=9)
    ax.legend(fontsize=7)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)

    # Amplitude dependence
    ax = axes[1]
    for f0 in [10, 100]:
        y = [cond_means.get((f0, 2), np.nan), cond_means.get((f0, 10), np.nan)]
        x = np.array([2, 10])
        ax.plot(x, y, "o-", label=f"{f0} Hz")
        if np.isfinite(y[0]) and np.isfinite(y[1]):
            ax.annotate(f"{y[1]-y[0]:+.1f} dB", (x[1], y[1]), textcoords="offset points",
                        xytext=(5, 5), fontsize=7)
    ax.set_xscale("log")
    ax.set_xticks([2, 10])
    ax.set_xlabel("Amplitude (mVpp)")
    ax.set_ylabel("Mean crosstalk (dB)")
    ax.set_title("Amplitude dependence", fontsize=9)
    ax.legend(fontsize=7)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)

    plt.suptitle("Frequency and Amplitude Dependence", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def export_summary_table(crosstalk_results, output_path):
    """Figure 6: Summary table per freq/amp (also exported as CSV + LaTeX)."""
    conditions = [(10, 2), (10, 10), (100, 2), (100, 10)]
    rows = []
    for f0, amp in conditions:
        key = (f0, amp)
        if key not in crosstalk_results:
            continue
        res = crosstalk_results[key]
        xt = res["xt_corrected"]
        censored = res["censored"]
        vals = xt[~censored]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        worst_idx = np.unravel_index(np.argmax(vals), xt.shape)
        worst_pair = f"CH{worst_idx[0]+1}->CH{worst_idx[1]+1}"
        n_censored = int(censored.sum())
        n_total = 16 * 16 - 16
        dl = res["detection_limit_db"]

        adj_vals, non_adj_vals = [], []
        for i in range(16):
            for j in range(16):
                if i == j or censored[i, j]:
                    continue
                d = abs((i + 1) - (j + 1))
                if d == 1:
                    adj_vals.append(xt[i, j])
                else:
                    non_adj_vals.append(xt[i, j])

        rows.append({
            "Freq (Hz)": f0,
            "Amplitude (mVpp)": amp,
            "Worst pair": worst_pair,
            "Worst XT (dB)": round(float(vals.max()), 2),
            "Median XT (dB)": round(float(np.median(vals)), 2),
            "Mean adj (dB)": round(float(np.mean(adj_vals)), 2) if adj_vals else "N/A",
            "Mean non-adj (dB)": round(float(np.mean(non_adj_vals)), 2) if non_adj_vals else "N/A",
            "Censored pairs": f"{n_censored}/{n_total}",
            "Detection limit (dB)": round(float(dl), 2) if np.isfinite(dl) else "N/A",
        })

    df = pd.DataFrame(rows)

    # Plot as table figure
    fig, ax = plt.subplots(figsize=_fig_size(7.0, 0.5))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.5)
    for j in range(len(df.columns)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Summary Table: Crosstalk per Frequency/Amplitude Condition", fontsize=10, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "fig6_summary_table.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(os.path.join(FIGURE_DIR, "fig6_summary_table.pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: fig6_summary_table.png/pdf")

    # CSV
    csv_path = os.path.join(TABLE_DIR, "summary_crosstalk.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # LaTeX (booktabs)
    tex_path = os.path.join(TABLE_DIR, "summary_crosstalk.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Crosstalk summary per frequency/amplitude condition.}\n")
        f.write("\\begin{tabular}{" + "l" * len(df.columns) + "}\n\\toprule\n")
        f.write(" & ".join(df.columns) + " \\\\\n\\midrule\n")
        for _, row in df.iterrows():
            f.write(" & ".join(str(v) for v in row.values) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  Saved: {tex_path}")

    return df


def plot_baseline_noise(noise_ref, output_path):
    """Figure 7: Per-channel noise bar chart (uV RMS in 0.5-100Hz and 20-450Hz)."""
    channels = list(range(1, 17))
    rms_05_100 = [noise_ref[ch]["rms_bands_uV"].get("0.5-100Hz", np.nan) for ch in channels]
    rms_20_450 = [noise_ref[ch]["rms_bands_uV"].get("20-450Hz", np.nan) for ch in channels]

    fig, ax = plt.subplots(figsize=_fig_size(7.0, 0.5))
    x = np.arange(len(channels))
    width = 0.35
    ax.bar(x - width / 2, rms_05_100, width, label="0.5-100 Hz", color="#4472C4")
    ax.bar(x + width / 2, rms_20_450, width, label="20-450 Hz", color="#ED7D31")
    ax.axvline(8.5, color="black", linestyle="--", linewidth=1, label="ADS1299 boundary")
    ax.set_xlabel("Channel")
    ax.set_ylabel("RMS noise (uV)")
    ax.set_title("Per-Channel Noise Floor (Baseline)", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in channels])
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_baseline_psd(noise_ref, output_path):
    """Figure 8: PSD overlay of all 16 channels (nV/sqrt(Hz)), 50Hz harmonics marked."""
    fig, ax = plt.subplots(figsize=_fig_size(7.0, 0.6))
    colors = plt.cm.tab20(np.linspace(0, 1, 16))
    for ch in range(1, 17):
        nr = noise_ref[ch]
        ax.plot(nr["welch_freqs_hz"], nr["welch_density_nv"],
                color=colors[ch - 1], label=f"CH{ch}", linewidth=0.7)
    for h in range(1, 11):
        f_harm = 50 * h
        ax.axvline(f_harm, color="red", linestyle=":", alpha=0.3, linewidth=0.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (nV$^2$/Hz)")
    ax.set_title("Baseline PSD – All 16 Channels", fontsize=10, fontweight="bold")
    ax.set_xlim([0, 500])
    ax.legend(fontsize=5.5, ncol=4, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def export_baseline_table(noise_ref, baseline_seg_df, output_path):
    """Figure 9: Per-channel table (mean, RMS, Vpp, band-limited noise)."""
    channels = list(range(1, 17))
    rows = []
    for ch in channels:
        ch_uV = remove_dc(counts_to_uv(baseline_seg_df[f"ch{ch}"].values.astype(float)))
        nr = noise_ref[ch]
        rows.append({
            "Channel": ch,
            "Mean (uV)": round(float(np.mean(ch_uV)), 2),
            "RMS (uV)": round(float(np.sqrt(np.mean(ch_uV**2))), 2),
            "Vpp (uV)": round(float(np.max(ch_uV) - np.min(ch_uV)), 2),
            "Noise 0.5-100Hz (uV)": round(float(nr["rms_bands_uV"].get("0.5-100Hz", np.nan)), 2) if np.isfinite(nr["rms_bands_uV"].get("0.5-100Hz", np.nan)) else "N/A",
            "Noise 20-450Hz (uV)": round(float(nr["rms_bands_uV"].get("20-450Hz", np.nan)), 2) if np.isfinite(nr["rms_bands_uV"].get("20-450Hz", np.nan)) else "N/A",
        })

    df = pd.DataFrame(rows)
    medians = {
        "Channel": "Median",
        "Mean (uV)": round(df["Mean (uV)"].astype(float).median(), 2),
        "RMS (uV)": round(df["RMS (uV)"].astype(float).median(), 2),
        "Vpp (uV)": round(df["Vpp (uV)"].astype(float).median(), 2),
        "Noise 0.5-100Hz (uV)": round(df["Noise 0.5-100Hz (uV)"].astype(float).median(), 2),
        "Noise 20-450Hz (uV)": round(df["Noise 20-450Hz (uV)"].astype(float).median(), 2),
    }
    df = pd.concat([df, pd.DataFrame([medians])], ignore_index=True)

    fig, ax = plt.subplots(figsize=_fig_size(7.0, 0.5))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.5)
    for j in range(len(df.columns)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for j in range(len(df.columns)):
        table[len(df), j].set_facecolor("#D9E2F3")
        table[len(df), j].set_text_props(fontweight="bold")
    ax.set_title("Baseline Noise per Channel", fontsize=10, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "fig9_baseline_table.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(os.path.join(FIGURE_DIR, "fig9_baseline_table.pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: fig9_baseline_table.png/pdf")

    csv_path = os.path.join(TABLE_DIR, "baseline_channels.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    tex_path = os.path.join(TABLE_DIR, "baseline_channels.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Baseline noise per channel.}\n")
        f.write("\\begin{tabular}{" + "l" * len(df.columns) + "}\n\\toprule\n")
        f.write(" & ".join(df.columns) + " \\\\\n\\midrule\n")
        for _, row in df.iterrows():
            f.write(" & ".join(str(v) for v in row.values) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  Saved: {tex_path}")

    return df


def plot_snr_helper(noise_ref, crosstalk_results, output_path):
    """Figure 10: SNR helper for each driven channel (band: 0.5-100 Hz)."""
    conditions = [(10, 2), (10, 10), (100, 2), (100, 10)]
    fig, axes = plt.subplots(2, 2, figsize=_fig_size(7.0, 0.8))
    axes = axes.flatten()

    for idx, (f0, amp) in enumerate(conditions):
        ax = axes[idx]
        key = (f0, amp)
        if key not in crosstalk_results:
            continue
        res = crosstalk_results[key]
        driven_ch = res["driven_channel"]
        A_sig = res["driven_amp_uV"]

        snr_vals = []
        for ch in range(1, 17):
            noise_rms = noise_ref[ch]["rms_bands_uV"].get("0.5-100Hz", np.nan)
            if ch == driven_ch and np.isfinite(A_sig) and np.isfinite(noise_rms) and noise_rms > 0:
                snr = 20 * np.log10(A_sig / noise_rms)
            else:
                snr = np.nan
            snr_vals.append(snr)

        channels = list(range(1, 17))
        ax.bar(channels, snr_vals, color="#4472C4")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Channel")
        ax.set_ylabel("SNR (dB)")
        ax.set_title(f"{f0} Hz, {amp} mVpp – SNR (band: 0.5-100 Hz)", fontsize=8)
        ax.set_xticks(channels)

    plt.suptitle("SNR Helper: 20*log10(A_sig / noise_RMS_0.5-100Hz)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def main():
    print("=" * 70)
    print("Crosstalk and Baseline-Noise Analysis Pipeline")
    print("=" * 70)

    # --- Load baseline ---
    print("\n[1/6] Loading baseline...")
    baseline_file = os.path.join(BASELINE_PATH, "sokosti_ble_capture_20260915_151745_rec01.csv")
    baseline_seg, baseline_gaps, baseline_full = load_pc_csv(baseline_file)
    baseline_duration = baseline_gaps["n_samples_seg"] / FS
    print(f"  Baseline: {baseline_gaps['n_samples_full']:,} samples, "
          f"{baseline_gaps['n_samples_seg']:,} in longest gap-free segment, "
          f"{baseline_duration:.2f} s")
    print(f"  Gaps: {baseline_gaps['n_gaps']}, max gap: {baseline_gaps['max_gap_samples']} samples, "
          f"lost: {baseline_gaps['total_lost_samples']} samples")

    # --- Compute noise reference from baseline ---
    print("\n[2/6] Computing noise reference from baseline...")
    rms_bands = {"0.5-100Hz": (0.5, 100), "20-450Hz": (20, 450)}
    noise_ref = compute_noise_reference(baseline_seg, FS, SINE_F0, rms_bands)
    print(f"  Noise reference computed for {len(noise_ref)} channels")

    # --- Load and process crosstalk files ---
    print("\n[3/6] Loading and processing crosstalk files...")
    pc_files = sorted(glob.glob(os.path.join(CROSSTALK_PATH, "*", "*_PC.csv")))
    print(f"  Found {len(pc_files)} PC.csv files")

    # Per-condition storage: key = (f0, amp) -> {driven_ch, per-channel amplitudes, ...}
    condition_data = {}
    log_entries = []
    files_with_significant_loss = []
    failed_files = []

    for filepath in pc_files:
        try:
            ch, amp, f0 = parse_filename(filepath)
            df_seg, gaps, df_full = load_pc_csv(filepath)
            duration = len(df_seg) / FS

            # Convert to uV, remove DC per channel
            ch_data_uV = {}
            for c in range(1, 17):
                ch_data_uV[c] = remove_dc(counts_to_uv(df_seg[f"ch{c}"].values.astype(float)))

            # Extract amplitudes for each segment, then average across segments
            n_seg_samples = int(FS * SEGMENT_DURATION)
            n_full = (len(ch_data_uV[1]) // n_seg_samples) * n_seg_samples
            n_segments = n_full // n_seg_samples

            # Limit segments for speed (median across 3 is sufficient)
            max_segs = 3
            if n_segments > max_segs:
                n_segments = max_segs

            seg_ls = {c: [] for c in range(1, 17)}
            seg_fft = {c: [] for c in range(1, 17)}
            seg_rms = {c: [] for c in range(1, 17)}
            for seg_idx in range(n_segments):
                start = seg_idx * n_seg_samples
                end = start + n_seg_samples
                t = np.arange(n_seg_samples) / FS
                for c in range(1, 17):
                    seg = ch_data_uV[c][start:end]
                    ls = amplitude_ls_sine_fit(t, seg, FS, f0)
                    seg_ls[c].append(ls["amp_peak_uV"])
                    seg_fft[c].append(amplitude_fft_hann(seg, FS, f0))
                    seg_rms[c].append(amplitude_rms_sqrt2(seg))

            # Median amplitude across segments (uV peak)
            amp_per_channel = {}
            for c in range(1, 17):
                ls_vals = [v for v in seg_ls[c] if np.isfinite(v)]
                amp_per_channel[c] = float(np.median(ls_vals)) if ls_vals else np.nan

            key = (f0, amp)
            if key not in condition_data:
                condition_data[key] = {
                    "driven_channel": ch,
                    "per_channel_amps": {},  # ch -> amplitude uV
                    "driven_amp": np.nan,
                    "n_segments": n_segments,
                    "duration_s": duration,
                }
            condition_data[key]["per_channel_amps"][ch] = amp_per_channel

            log_entries.append({
                "file": os.path.basename(filepath),
                "channel": ch,
                "freq": f0,
                "amp": amp,
                "n_samples": len(df_seg),
                "duration_s": duration,
                "n_gaps": gaps["n_gaps"],
                "max_gap": gaps["max_gap_samples"],
                "lost": gaps["total_lost_samples"],
                "flagged": gaps["flagged"],
            })

            if gaps["flagged"]:
                files_with_significant_loss.append(os.path.basename(filepath))

        except Exception as e:
            failed_files.append((filepath, str(e)))
            print(f"  FAILED: {os.path.basename(filepath)}: {e}")

        done = len(log_entries) + len(failed_files)
        if done % 10 == 0 or done == len(pc_files):
            print(f"  Processed {done}/{len(pc_files)} files...")

    # For each condition, set the driven channel amplitude and compute crosstalk matrix
    print("\n[4/6] Computing crosstalk matrices...")
    crosstalk_results = {}
    for key, data in condition_data.items():
        f0, amp = key
        driven_ch = data["driven_channel"]

        # Driven amplitude: use the amplitude from the driven channel itself
        A_driven = data["per_channel_amps"].get(driven_ch, np.nan)

        # Victim amplitudes: all 16 channels
        victim_amps = np.array([data["per_channel_amps"].get(c, np.nan) for c in range(1, 17)])

        # Noise amplitude at test frequency from baseline (driven channel's noise)
        A_noise = noise_ref[driven_ch]["fft_amps_uV"].get(f0, np.nan)

        # Compute crosstalk matrix
        matrix = compute_crosstalk_matrix(A_driven, victim_amps, A_noise)

        crosstalk_results[key] = {
            "driven_channel": driven_ch,
            "frequency": f0,
            "amplitude": amp,
            "driven_amp_uV": A_driven,
            "victim_amps_uV": victim_amps,
            "A_noise_uV": A_noise,
            "xt_db": matrix["xt_db"],
            "xt_corrected": matrix["xt_corrected"],
            "censored": matrix["censored"],
            "detection_limit_db": matrix["detection_limit_db"],
            "n_segments": data["n_segments"],
            "duration_s": data["duration_s"],
        }

    # --- Generate figures ---
    print("\n[5/6] Generating figures and tables...")

    # Conditions info for parameter overview
    conditions_info = {}
    for key, res in crosstalk_results.items():
        cond_key = (res["frequency"], res["amplitude"])
        if cond_key not in conditions_info:
            conditions_info[cond_key] = {"ch": res["driven_channel"], "n_files": 0}
        conditions_info[cond_key]["n_files"] += 1

    plot_parameter_overview(conditions_info, baseline_duration,
                            os.path.join(FIGURE_DIR, "fig1_parameter_overview.png"))
    plot_psd_overlay(noise_ref, crosstalk_results,
                     os.path.join(FIGURE_DIR, "fig2_psd_overlay.png"))
    plot_crosstalk_matrix(crosstalk_results,
                          os.path.join(FIGURE_DIR, "fig3_crosstalk_matrix.png"))
    plot_crosstalk_vs_distance(crosstalk_results,
                               os.path.join(FIGURE_DIR, "fig4_crosstalk_vs_distance.png"))
    plot_freq_amp_dependence(crosstalk_results,
                             os.path.join(FIGURE_DIR, "fig5_freq_amp_dependence.png"))
    export_summary_table(crosstalk_results,
                         os.path.join(TABLE_DIR, "summary_crosstalk.csv"))
    plot_baseline_noise(noise_ref,
                        os.path.join(FIGURE_DIR, "fig7_baseline_noise.png"))
    plot_baseline_psd(noise_ref,
                      os.path.join(FIGURE_DIR, "fig8_baseline_psd.png"))
    export_baseline_table(noise_ref, baseline_seg,
                          os.path.join(TABLE_DIR, "baseline_channels.csv"))
    plot_snr_helper(noise_ref, crosstalk_results,
                    os.path.join(FIGURE_DIR, "fig10_snr_helper.png"))

    # --- Print log ---
    print("\n" + "=" * 70)
    print("ANALYSIS LOG")
    print("=" * 70)
    print(f"Files processed: {len(log_entries)}")
    print(f"Failed files: {len(failed_files)}")
    if failed_files:
        for f, e in failed_files:
            print(f"  - {os.path.basename(f)}: {e}")
    print(f"Files with significant packet loss (>1%): {len(files_with_significant_loss)}")
    for f in files_with_significant_loss:
        print(f"  - {f}")
    print(f"\nSample counts and gap statistics:")
    for entry in log_entries:
        flag = " *** FLAGGED ***" if entry["flagged"] else ""
        print(f"  {entry['file']}: {entry['n_samples']:,} samples, "
              f"{entry['duration_s']:.2f} s, gaps={entry['n_gaps']}, "
              f"max_gap={entry['max_gap']}, lost={entry['lost']}{flag}")
    print(f"\nBaseline: {baseline_gaps['n_samples_full']:,} samples, "
          f"{baseline_duration:.2f} s, gaps={baseline_gaps['n_gaps']}")

    print(f"\nOutput directories:")
    print(f"  Figures: {FIGURE_DIR}")
    print(f"  Tables: {TABLE_DIR}")
    print("=" * 70)
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
