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
parser.add_argument("--save", type=str, default=None, help="Save figures to PNG files with this prefix.")
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

df = pd.read_csv(csv_path)

fs = 1000  # Hz

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
b_notch, a_notch = signal.butter(2, [49 / nyquist, 51 / nyquist], btype="bandstop")
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


if args.contractions:
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

# ===== FIGURE 2: magnitude (top) + strong/weak overlays (bottom row) =====
NORM_PTS = 101  # 0..100 %


def normalized_envelopes(env, contractions, global_max):
    """Resample each contraction envelope to 0-100 % of its duration.
    Amplitude is expressed as % of the GLOBAL envelope max so that strong
    and weak contractions are directly comparable across subplots."""
    curves = []
    for (a, b) in contractions:
        i0, i1 = int(a * fs), int(b * fs)
        seg = env[i0:i1]
        seg = seg[~np.isnan(seg)]
        if len(seg) < 10:
            continue
        x_old = np.linspace(0, 100, len(seg))
        x_new = np.linspace(0, 100, NORM_PTS)
        curves.append(np.interp(x_new, x_old, seg))
    return np.array(curves)


env_ch = envelopes[:, plot_indices[0]]
global_max = np.nanmax(env_ch)

fig2 = plt.figure(figsize=(15, 8))
gs = fig2.add_gridspec(2, 2, height_ratios=[1.2, 1])
# magnitude plot spans the full top row (both columns)
ax_mag = fig2.add_subplot(gs[0, :])
ax_mag.plot(t, emg_filtered[:, plot_indices[0]], color="0.7", linewidth=0.4,
            label="Filtered EMG")
ax_mag.plot(t, env_ch, "r-", linewidth=1.2, label="Envelope")
ax_mag.set_ylabel(f"Ch{args.channels[0]}")
ax_mag.legend(loc="upper right", fontsize=8)
ax_mag.grid(True, alpha=0.3)
ax_mag.set_title("Filtered EMG magnitude + linear envelope")

cmap_strong = plt.cm.viridis(np.linspace(0, 0.85, max(len(strong), 1)))
cmap_weak = plt.cm.viridis(np.linspace(0, 0.85, max(len(weak), 1)))

bottom_axes = []
for ax, contractions, cmap, name in [
        (fig2.add_subplot(gs[1, 0]), strong, cmap_strong, "Strong"),
        (fig2.add_subplot(gs[1, 1]), weak, cmap_weak, "Weak")]:
    bottom_axes.append(ax)
    curves = normalized_envelopes(env_ch, contractions, global_max)
    for k, c in enumerate(curves):
        ax.plot(np.linspace(0, 100, NORM_PTS), 100.0 * c / global_max,
                color=cmap[k], linewidth=1.2, label=f"{name} {k+1}")
    if len(curves):
        mean_curve = 100.0 * np.mean(curves, axis=0) / global_max
        ax.plot(np.linspace(0, 100, NORM_PTS), mean_curve, "k-", linewidth=2.5, label="Mean")
    ax.set_ylabel("Envelope (% of global max)")
    ax.set_xlabel("Contraction duration (%)")
    ax.set_title(f"{name} contractions (n={len(curves)})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
# auto-scale each subplot independently so weak contractions fill their window,
# but the y-axis label shows "% of global max" so the scale is comparable
fig2.suptitle(f"Isometric contraction analysis (Ch{args.channels[0]}) - "
              f"{os.path.basename(csv_path)}", fontsize=12)
fig2.tight_layout()

if args.save:
    fig1.savefig(args.save + "_magnitude.png", dpi=150)
    fig2.savefig(args.save + "_overlays.png", dpi=150)
    print(f"Saved: {args.save}_magnitude.png, {args.save}_overlays.png")

plt.show()
