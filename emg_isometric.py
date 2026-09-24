# EMG isometric contraction analysis:
#   1) bandpass 20-400 Hz + notch 49-51 Hz -> magnitude plot
#   2) linear envelope (rectify + low-pass Butterworth)
#   3) auto-detect strong (~5 s) / weak (~30 s) contractions (or manual override)
#   4) overlay time-normalized (0-100 %) envelopes in separate strong/weak subplots

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

# ===== COMMAND LINE ARGUMENTS =====
parser = argparse.ArgumentParser(
    description="EMG isometric analysis: filtered magnitude, linear envelope, "
                "and overlaid contraction envelopes (strong vs weak).")
parser.add_argument("csv", nargs="?", help="Path to the CSV file (defaults to the most recent capture).")
parser.add_argument("--channels", type=int, nargs="+", default=[1],
                    help="Channels to analyze (1-based). Default: 1")
parser.add_argument("--thresh-strong", type=float, default=20.0,
                    help="Strong-contraction detection threshold as %% of the GLOBAL envelope max (default 20).")
parser.add_argument("--thresh-weak", type=float, default=5.0,
                    help="Weak-contraction detection threshold as %% of the GLOBAL envelope max (default 5).")
parser.add_argument("--env-cutoff", type=float, default=6.0,
                    help="Low-pass cutoff (Hz) for the linear envelope (default 6).")
parser.add_argument("--strong-min", type=float, default=2.0,
                    help="Minimum duration (s) for a contraction to be classified strong (default 2).")
parser.add_argument("--weak-min", type=float, default=10.0,
                    help="Minimum duration (s) for a contraction to be classified weak (default 10).")
parser.add_argument("--merge-gap", type=float, default=0.5,
                    help="Merge above-threshold runs separated by gaps shorter than this (s, default 0.5).")
parser.add_argument("--contractions", type=str, default=None,
                    help='Manual contraction windows in seconds, e.g. "10-15,20-50,60-65". '
                         "Overrides auto-detection. Classified by duration unless --labels given.")
parser.add_argument("--labels", type=str, default=None,
                    help='Comma-separated labels for manual contractions: "s" or "w" per contraction, '
                         'e.g. "s,s,w,w". Only used with --contractions.')
parser.add_argument("--strong-windows", type=str, default=None,
                    help='Manually set STRONG contraction windows in seconds, e.g. "5-11,21-27". '
                         "Overrides auto-detection for strong contractions.")
parser.add_argument("--weak-windows", type=str, default=None,
                    help='Manually set WEAK contraction windows in seconds, e.g. "77-97,117-147". '
                         "Overrides auto-detection for weak contractions.")
parser.add_argument("--save", type=str, default=None, help="Save figures to PNG files with this prefix.")
parser.add_argument("--save-results", action="store_true",
                    help="Write the PNG figures and a terminal-output log file into the same "
                         "folder as the CSV file (named after the CSV).")
parser.add_argument("--gain", type=float, default=1.0,
                    help="ADS1299 PGA gain used during recording, for converting ADC codes to "
                         "microvolts (Vref = 4.5 V). Default 1.")
parser.add_argument("--unit", type=str, default=None,
                    help="Y-axis unit for the magnitude plot: 'uV', 'mV' or 'V'. "
                         "Default: auto (uV if max < 1 mV, else mV).")
parser.add_argument("--compare-channels", action="store_true",
                    help="Compare all channels: print signal, noise and SNR for every "
                         "contraction of every channel, plus best/worst channels.")
parser.add_argument("--psd", choices=["fft", "welch"], default="fft",
                    help="PSD method for the frequency plot below the magnitude: "
                         "'fft' (classic FFT magnitude) or 'welch' (Welch PSD). Default: fft.")
args = parser.parse_args()

# ===== LOG CAPTURE (--save-results) =====
# When saving results, tee all terminal output into a log file next to the CSV.
if args.save_results:
    import contextlib
    import io
    _log_buf = io.StringIO()
    _log_ctx = contextlib.redirect_stdout(_log_buf)
    _log_ctx.__enter__()
    # record the exact command so the analysis can be rerun
    _log_buf.write("# Command: " + " ".join(sys.argv) + "\n\n")

# ===== LOAD CSV =====
csv_path = args.csv
if csv_path is None:
    files = sorted(glob.glob(os.path.join("captures", "sokosti_capture_*.csv")))
    if not files:
        print("No CSV file given and none found in captures/.", file=sys.stderr)
        sys.exit(1)
    csv_path = files[-1]
    print(f"Using most recent capture: {csv_path}")

df = pd.read_csv(csv_path)

fs = 1000  # Hz

# ADS1299 scaling: raw ADC codes -> volts (Vref = 4.5 V, 24-bit signed)
VREF = 4.5
VOLTS_PER_CODE = VREF / args.gain / (2 ** 23)

# EMG channels present in the file (ch1, ch2, ... sorted numerically)
emg_channels = [c for c in df.columns if isinstance(c, str) and c.lower().startswith("ch")
                and c.lower()[2:].isdigit()]
emg_channels.sort(key=lambda c: int(c[2:]))
emg_data = df[emg_channels].values.astype(float)

# ===== GAP DETECTION (sample counter column) =====
idx = df.iloc[:, 0].values.astype(np.int64)
full_start, full_end = idx[0], idx[-1]
n_full = full_end - full_start + 1

received_mask = np.zeros(n_full, dtype=bool)
received_mask[idx - full_start] = True

recon = np.full((n_full, emg_data.shape[1]), np.nan)
recon[received_mask] = emg_data

n_missing = int((~received_mask).sum())
print(f"File: {csv_path}")
print(f"Samples: {len(idx)} received, {n_missing} missing ({100.0 * n_missing / n_full:.2f}%)")

# ===== FILTER DESIGN =====
nyquist = fs / 2
b_bandpass, a_bandpass = signal.butter(4, [20 / nyquist, 400 / nyquist], btype="bandpass")
b_notch, a_notch = signal.butter(2, [48 / nyquist, 52 / nyquist], btype="bandstop")
b_env, a_env = signal.butter(4, args.env_cutoff / nyquist, btype="lowpass")


def valid_segments(mask, min_len=1):
    """Return (start, end) index pairs of contiguous True runs."""
    d = np.diff(mask.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, len(mask)]
    return [(s, e) for s, e in zip(starts, ends) if e - s >= min_len]


# ===== FILTERING: per contiguous valid segment =====
long_segments = valid_segments(received_mask, min_len=64)
emg_filtered = np.full_like(recon, np.nan)
for (s, e) in long_segments:
    for i in range(recon.shape[1]):
        seg = recon[s:e, i]
        seg = signal.filtfilt(b_bandpass, a_bandpass, seg)
        seg = signal.filtfilt(b_notch, a_notch, seg)
        emg_filtered[s:e, i] = seg

# ===== LINEAR ENVELOPE: rectify + low-pass, per valid segment =====
envelopes = np.full_like(recon, np.nan)
for (s, e) in long_segments:
    for i in range(recon.shape[1]):
        seg = np.abs(emg_filtered[s:e, i])
        envelopes[s:e, i] = signal.filtfilt(b_env, a_env, seg)

t = np.arange(n_full) / fs

# ===== FIGURE 1: magnitude + envelope per channel =====
plot_indices = [c - 1 for c in args.channels]
n_channels = len(plot_indices)

fig1, axes1 = plt.subplots(n_channels, 1, figsize=(15, 2.8 * n_channels), squeeze=False)
for row, i in enumerate(plot_indices):
    ax = axes1[row][0]
    ax.plot(t, emg_filtered[:, i], color="0.7", linewidth=0.4, label="Filtered EMG")
    ax.plot(t, envelopes[:, i], "r-", linewidth=1.2, label="Envelope")
    ax.axhline(0, color="k", linewidth=0.3)
    ax.set_ylabel(f"Ch{i+1}")
    ax.grid(True, alpha=0.3)
    if row == 0:
        ax.legend(loc="upper right")
    if row == n_channels - 1:
        ax.set_xlabel("Time (s)")
fig1.suptitle(f"Filtered EMG magnitude + linear envelope - {os.path.basename(csv_path)}", fontsize=12)
fig1.tight_layout()
plt.close(fig1)  # don't pop up a window; only fig2 is shown

# ===== CONTRACTION DETECTION =====
def parse_manual(spec):
    """Parse '10-15,20-50' into [(s0, e0), ...] in seconds."""
    out = []
    for part in spec.split(","):
        a, b = part.split("-")
        out.append((float(a), float(b)))
    return out


def detect_contractions(env, thresh_strong_pct, thresh_weak_pct, merge_gap_s,
                        strong_min_s, weak_min_s):
    """Two-threshold detection, both relative to the GLOBAL envelope max.

    Strong: runs above thresh_strong with duration >= strong_min_s.
    Weak:   runs above thresh_weak with duration >= weak_min_s
            (strong contractions also cross this threshold, so runs that
            reach the strong threshold are excluded from weak).
    Returns (strong, weak) lists of (start_s, end_s).
    """
    valid = ~np.isnan(env)
    vmax = np.nanmax(env)
    t_strong = thresh_strong_pct / 100.0 * vmax
    t_weak = thresh_weak_pct / 100.0 * vmax

    def runs_above(thresh):
        above = valid & (env > thresh)
        runs = valid_segments(above, min_len=1)
        # merge runs separated by short gaps
        merged = []
        gap_samples = int(merge_gap_s * fs)
        for (s, e) in runs:
            if merged and s - merged[-1][1] <= gap_samples:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        return merged

    strong, weak = [], []
    for (s, e) in runs_above(t_strong):
        if (e - s) / fs >= strong_min_s:
            strong.append((s / fs, e / fs))
    for (s, e) in runs_above(t_weak):
        dur = (e - s) / fs
        if dur < weak_min_s:
            continue
        # exclude runs that belong to a strong contraction (overlap)
        if any(not (e / fs <= sa or s / fs >= sb) for (sa, sb) in strong):
            continue
        weak.append((s / fs, e / fs))
    return strong, weak


if args.strong_windows or args.weak_windows:
    # manual strong/weak windows set explicitly (either or both)
    strong = parse_manual(args.strong_windows) if args.strong_windows else []
    weak = parse_manual(args.weak_windows) if args.weak_windows else []
elif args.contractions:
    manual = parse_manual(args.contractions)
    if args.labels:
        labels = [x.strip().lower() for x in args.labels.split(",")]
        if len(labels) != len(manual):
            print("Error: --labels count must match --contractions count.", file=sys.stderr)
            sys.exit(1)
        strong = [m for m, l in zip(manual, labels) if l.startswith("s")]
        weak = [m for m, l in zip(manual, labels) if l.startswith("w")]
    else:
        strong, weak = [], []
        for (a, b) in manual:
            (weak if (b - a) >= args.weak_min else strong).append((a, b))
else:
    strong, weak = detect_contractions(
        envelopes[:, plot_indices[0]], args.thresh_strong, args.thresh_weak,
        args.merge_gap, args.strong_min, args.weak_min)

print("=" * 60)
print("CONTRACTIONS DETECTED")
print(f"Strong ({len(strong)}): " + ", ".join(f"{a:.1f}-{b:.1f}s" for a, b in strong))
print(f"Weak   ({len(weak)}): " + ", ".join(f"{a:.1f}-{b:.1f}s" for a, b in weak))
print(f"Total: {len(strong) + len(weak)}")
print("=" * 60)

# ===== CHANNEL COMPARISON (--compare-channels) =====
if args.compare_channels:
    all_contractions = strong + weak
    print("\n" + "=" * 70)
    print("CHANNEL COMPARISON (signal / noise / SNR per contraction)")
    print("=" * 70)
    print(f"Contractions used: {len(all_contractions)} "
          f"({len(strong)} strong, {len(weak)} weak)")
    print(f"Signal = mean envelope during contraction; "
          f"Noise = mean envelope in the window [a-2*pad, a-pad] before onset, "
          f"leaving the pad seconds right before onset (the ramp) out.")
    print("-" * 70)

    # per-channel aggregate metrics
    ch_metrics = []
    for ci in range(envelopes.shape[1]):
        env = envelopes[:, ci]
        sigs, noises = [], []
        for (a, b) in all_contractions:
            pad = 0.2 * (b - a)
            i_sig0, i_sig1 = int(a * fs), int(b * fs)
            i_noi0 = max(0, int((a - 2 * pad) * fs))
            i_noi1 = max(0, int((a - pad) * fs))
            sig = env[i_sig0:i_sig1]
            noi = env[i_noi0:i_noi1]
            sig = sig[~np.isnan(sig)]
            noi = noi[~np.isnan(noi)]
            if len(sig) < 10 or len(noi) < 10:
                continue
            sigs.append(np.mean(sig))
            noises.append(np.mean(noi))
        if not sigs:
            ch_metrics.append((ci, 0.0, 0.0, 0.0, 0))
            continue
        sig_mean = np.mean(sigs)
        noise_mean = np.mean(noises)
        snr = sig_mean / noise_mean if noise_mean > 0 else float("inf")
        ch_metrics.append((ci, sig_mean, noise_mean, snr, len(sigs)))

    # print per-contraction detail + per-channel summary
    for ci, sig_mean, noise_mean, snr, n in ch_metrics:
        print(f"\nCh{ci+1}: mean signal {sig_mean:.3g}, mean noise {noise_mean:.3g}, "
              f"SNR {snr:.2f} (over {n} contractions)")
        for (a, b) in all_contractions:
            pad = 0.2 * (b - a)
            i_sig0, i_sig1 = int(a * fs), int(b * fs)
            i_noi0 = max(0, int((a - 2 * pad) * fs))
            i_noi1 = max(0, int((a - pad) * fs))
            sig = envelopes[i_sig0:i_sig1, ci]
            noi = envelopes[i_noi0:i_noi1, ci]
            sig = sig[~np.isnan(sig)]
            noi = noi[~np.isnan(noi)]
            if len(sig) < 10 or len(noi) < 10:
                continue
            s, n = np.mean(sig), np.mean(noi)
            r = s / n if n > 0 else float("inf")
            print(f"    {a:6.1f}-{b:6.1f}s  signal {s:9.3g}  noise {n:9.3g}  SNR {r:7.2f}")

    # best / worst by SNR
    valid = [m for m in ch_metrics if m[3] != float("inf")]
    if valid:
        best = max(valid, key=lambda m: m[3])
        worst = min(valid, key=lambda m: m[3])
        print("\n" + "-" * 70)
        print(f"BEST channel:  Ch{best[0]+1}  (SNR {best[3]:.2f}, "
              f"signal {best[1]:.3g}, noise {best[2]:.3g})")
        print(f"WORST channel: Ch{worst[0]+1}  (SNR {worst[3]:.2f}, "
              f"signal {worst[1]:.3g}, noise {worst[2]:.3g})")
        print("=" * 70)

# ===== FIGURE 2: magnitude (top) + strong/weak overlays (bottom row) =====
NORM_PTS = 101  # 0..100 %


def contraction_segments(env, contractions):
    """Extract each contraction envelope with 20 % before/after context,
    on its REAL time axis (seconds relative to its own onset).
    No time rescaling: longer contractions simply last longer.
    Returns list of (x_rel, y) arrays."""
    out = []
    for (a, b) in contractions:
        pad = 0.2 * (b - a)
        i0 = max(0, int((a - pad) * fs))
        i1 = min(n_full, int((b + pad) * fs))
        seg = env[i0:i1]
        t_seg = np.arange(i0, i1) / fs - a  # seconds relative to onset
        valid = ~np.isnan(seg)
        if valid.sum() < 10:
            continue
        out.append((t_seg[valid], seg[valid]))
    return out


# unit selection: explicit --unit wins, otherwise auto (uV if max < 1 mV, else mV)
env_v = envelopes[:, plot_indices[0]] * VOLTS_PER_CODE
max_v = np.nanmax(env_v)
if args.unit:
    u = args.unit.lower().strip()
    unit_scale = {"uv": 1e6, "µv": 1e6, "mv": 1e3, "v": 1.0}[u]
    unit = {"uv": "µV", "µv": "µV", "mv": "mV", "v": "V"}[u]
else:
    if max_v < 1e-3:
        unit_scale, unit = 1e6, "µV"
    else:
        unit_scale, unit = 1e3, "mV"

env_ch = env_v * unit_scale
emg_filtered_uv = emg_filtered * VOLTS_PER_CODE * unit_scale

# ===== PSD helpers (gap-aware, per contiguous segment) =====
def welch_segment_average(x, fs, min_seg=512, nperseg=1024):
    """Average Welch PSD over contiguous non-NaN segments of length >= min_seg."""
    valid = ~np.isnan(x)
    psds, weights = [], []
    f_ref = None
    for (s, e) in valid_segments(valid, min_len=max(min_seg, nperseg)):
        seg = signal.detrend(x[s:e])
        f, pxx = signal.welch(seg, fs=fs, nperseg=nperseg, nfft=nperseg)
        if f_ref is None:
            f_ref = f
        psds.append(pxx)
        weights.append(e - s)
    if not psds:
        return None, None, 0
    pxx_avg = np.average(np.stack(psds), axis=0, weights=weights)
    return f_ref, pxx_avg, len(psds)


def fft_segment_average(x, fs, min_seg=512, nfft=1024):
    """Average FFT magnitude over contiguous non-NaN segments of length >= min_seg."""
    valid = ~np.isnan(x)
    f = np.fft.rfftfreq(nfft, 1 / fs)
    mags, weights = [], []
    for (s, e) in valid_segments(valid, min_len=max(min_seg, nfft)):
        seg = signal.detrend(x[s:e])
        fft_mag = np.abs(np.fft.rfft(seg, n=nfft))
        mags.append(fft_mag)
        weights.append(e - s)
    if not mags:
        return None, None, 0
    mag_avg = np.average(np.stack(mags), axis=0, weights=weights)
    return f, mag_avg, len(mags)


fig2 = plt.figure(figsize=(15, 10))
fig2.suptitle(f"Isometric contraction analysis (Ch{args.channels[0]}) - "
              f"{os.path.basename(csv_path)}", fontsize=12)

# 3 rows: magnitude (full width), PSD (full width), strong/weak (side by side)
gs = fig2.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 1.0])
# magnitude plot spans the full top row (both columns)
ax_mag = fig2.add_subplot(gs[0, :])
ax_mag.plot(t, emg_filtered_uv[:, plot_indices[0]], color="0.7", linewidth=0.4,
            label="Filtered EMG")
ax_mag.plot(t, env_ch, "r-", linewidth=1.2, label="Envelope")
ax_mag.set_ylabel(f"Ch{args.channels[0]} ({unit})")
ax_mag.set_xlabel("Time (s)")
ax_mag.legend(loc="upper right", fontsize=8)
ax_mag.grid(True, alpha=0.3)
ax_mag.set_title("Filtered EMG magnitude + linear envelope")

# PSD plot spans the middle row (both columns)
ax_psd = fig2.add_subplot(gs[1, :])
psd_ch = emg_filtered_uv[:, plot_indices[0]]
if args.psd == "welch":
    f_psd, pxx, n_segs = welch_segment_average(psd_ch, fs)
    if pxx is not None:
        ax_psd.semilogy(f_psd, pxx, "r-", linewidth=0.8,
                        label=f"Welch PSD ({n_segs} segment(s))")
        print(f"Ch{args.channels[0]}: Welch PSD from {n_segs} segment(s)")
    else:
        ax_psd.text(0.5, 0.5, "No segment long enough for Welch", ha="center",
                    transform=ax_psd.transAxes)
else:
    f_psd, mag, n_segs = fft_segment_average(psd_ch, fs)
    if mag is not None:
        ax_psd.plot(f_psd, mag, "r-", linewidth=0.8,
                    label=f"FFT magnitude ({n_segs} segment(s))")
        print(f"Ch{args.channels[0]}: FFT from {n_segs} segment(s)")
    else:
        ax_psd.text(0.5, 0.5, "No segment long enough for FFT", ha="center",
                    transform=ax_psd.transAxes)
ax_psd.set_ylabel(f"({unit})")
ax_psd.set_xlabel("Frequency (Hz)")
ax_psd.set_xlim(0, 500)
ax_psd.legend(loc="upper right", fontsize=8)
ax_psd.grid(True, alpha=0.3)
ax_psd.set_title(f"{'Welch PSD' if args.psd == 'welch' else 'FFT'} (contiguous segments)")

cmap_strong = plt.cm.viridis(np.linspace(0, 0.85, max(len(strong), 1)))
cmap_weak = plt.cm.viridis(np.linspace(0, 0.85, max(len(weak), 1)))

bottom_axes = []
for ax, contractions, cmap, name in [
        (fig2.add_subplot(gs[2, 0]), strong, cmap_strong, "Strong"),
        (fig2.add_subplot(gs[2, 1]), weak, cmap_weak, "Weak")]:
    bottom_axes.append(ax)
    segs = contraction_segments(env_ch, contractions)
    # x-axis in seconds: 0 = contraction onset (threshold crossing), real time.
    # Contractions are aligned at onset; ends are NOT aligned (durations differ).
    if segs:
        x_min = min(s[0][0] for s in segs)
        x_max = max(s[0][-1] for s in segs)
    else:
        x_min, x_max = -1.0, 1.0
    global_max = np.nanmax(env_ch)
    # Each contraction is plotted only over its own real time range
    # (-20 % before onset to +20 % after its end). No holding/extrapolation.
    for k, (x_seg, y_seg) in enumerate(segs):
        ax.plot(x_seg, 100.0 * y_seg / global_max, color="0.85", linewidth=0.8,
                alpha=0.6, linestyle="--",
                label="Individual" if k == 0 else None)  # light grey dashed individual
    if segs:
        # mean + 90 % CI only where ALL contractions are active. As soon as
        # one trace ends, the mean stops (no partial-n averaging).
        n_all = len(segs)
        t_start = min(s[0][0] for s in segs)
        t_end = min(s[0][-1] for s in segs)  # stop at the earliest end
        t_common = np.arange(t_start, t_end, 0.01)
        interp = np.array([np.interp(t_common, s[0], s[1]) for s in segs])
        mean_curve = np.mean(interp, axis=0)
        ci = 1.645 * np.std(interp, axis=0, ddof=1) / np.sqrt(n_all) if n_all > 1 \
            else np.zeros_like(mean_curve)
        ax.fill_between(t_common, 100.0 * (mean_curve - ci) / global_max,
                        100.0 * (mean_curve + ci) / global_max,
                        color="0.35", alpha=0.35, linewidth=0, label="90% CI")
        ax.plot(t_common, 100.0 * mean_curve / global_max, color="black",
                linewidth=2.5, label=f"Mean (n={n_all})")
    ax.set_ylabel("Envelope (% of global max)")
    ax.set_xlabel("Time from contraction onset (s)")
    ax.set_xlim(x_min, x_max)
    ax.set_title(f"{name} contractions (n={len(segs)})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
# auto-scale each subplot independently so weak contractions fill their window,
# but the y-axis label shows "% of global max" so the scale is comparable
fig2.suptitle(f"Isometric contraction analysis (Ch{args.channels[0]}) - "
              f"{os.path.basename(csv_path)}", fontsize=12)
fig2.tight_layout()

if args.save:
    fig1.savefig(args.save + "_magnitude.png", dpi=150)
    fig2.savefig(args.save + "_overlays.png", dpi=150)
    print(f"Saved: {args.save}_magnitude.png, {args.save}_overlays.png")

if args.save_results:
    # write PNGs + a log of the terminal output into the CSV's folder
    out_dir = os.path.dirname(os.path.abspath(csv_path))
    base = os.path.splitext(os.path.basename(csv_path))[0]
    png_mag = os.path.join(out_dir, base + "_magnitude.png")
    png_ovl = os.path.join(out_dir, base + "_overlays.png")
    log_path = os.path.join(out_dir, base + "_analysis.log")
    fig1.savefig(png_mag, dpi=150)
    fig2.savefig(png_ovl, dpi=150)
    # flush captured stdout to the log file
    _log_ctx.__exit__(None, None, None)
    with open(log_path, "w") as f:
        f.write(_log_buf.getvalue())
    print(f"Saved figures: {png_mag}, {png_ovl}")
    print(f"Log file: {log_path}")

plt.show()
