# apply 20 - 400 Hz bandpass
# apply 50 Hz bandstop

import argparse
import sys
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

# ===== COMMAND LINE ARGUMENTS =====
parser = argparse.ArgumentParser(description="EMG signal processing with bandpass and bandstop filtering.")
parser.add_argument("csv", nargs="?", help="Path to the CSV file (defaults to the most recent capture).")
parser.add_argument("--channels", type=int, nargs="+", help="Channels to plot (1-based). Example: --channels 12 or --channels 1 5 12")
parser.add_argument("--ylim_mag", type=float, nargs=2, help="Y-axis limits for all channels (min max). Example: --ylim -500 500")
parser.add_argument("--ylim_fft", type=float, nargs=2, help="Y-axis limits for all channels (min max). Example: --ylim -500 500")
parser.add_argument("--min_seg_fft", type=int, default=512, help="Minimum contiguous segment length (samples) used in Welch FFT (default 512).")
args = parser.parse_args()

# dataframe
csv_path = args.csv
if csv_path is None:
    import glob
    import os
    files = sorted(glob.glob(os.path.join("captures", "sokosti_capture_*.csv")))
    if not files:
        print("No CSV file given and none found in captures/.", file=sys.stderr)
        sys.exit(1)
    csv_path = files[-1]
    print(f"Using most recent capture: {csv_path}")

df = pd.read_csv(csv_path)

# channels to plot (1-based)
channels_to_plot = args.channels  # None means all channels

# sampling frequency
fs = 1000  # Hz

# get EMG channels (ch1-ch16)
emg_channels = [f'ch{i}' for i in range(1, 17)]
emg_data = df[emg_channels].values.astype(float)

# ===== GAP DETECTION (based on the counter index column) =====
idx = df.iloc[:, 0].values.astype(np.int64)  # first column = sample counter
full_start, full_end = idx[0], idx[-1]
n_full = full_end - full_start + 1

received_mask = np.zeros(n_full, dtype=bool)
received_mask[idx - full_start] = True

# reconstructed timeline with NaN where samples are missing (all channels)
recon = np.full((n_full, emg_data.shape[1]), np.nan)
recon[received_mask] = emg_data

n_missing = int((~received_mask).sum())
gap_lengths = []
d_idx = np.diff(idx)
gap_positions = np.where(d_idx != 1)[0]
for g in gap_positions:
    gap_lengths.append(int(d_idx[g] - 1))

print("=" * 60)
print("GAP STATISTICS")
print("=" * 60)
print(f"File: {csv_path}")
print(f"Counter range: {full_start} .. {full_end}  (expected {n_full} samples)")
print(f"Samples received: {len(idx)}")
print(f"Samples missing: {n_missing}  ({100.0 * n_missing / n_full:.2f}%)")
print(f"Number of gaps: {len(gap_lengths)}")
if gap_lengths:
    gl = np.array(gap_lengths, dtype=float)
    print(f"Gap length: mean {gl.mean():.1f}, median {np.median(gl):.0f}, "
          f"min {gl.min():.0f}, max {gl.max():.0f} samples")
    print(f"Gap length: mean {gl.mean()/fs*1000:.1f}, max {gl.max()/fs*1000:.0f} ms @{fs} Hz")
else:
    print("No gaps detected.")
print("=" * 60)

# select channels to plot (convert to 0-based indices)
if channels_to_plot is None:
    plot_indices = list(range(emg_data.shape[1]))
else:
    plot_indices = [c - 1 for c in channels_to_plot]

# time vector over the reconstructed (gap-aware) timeline
t = np.arange(n_full) / fs

# design bandpass filter (20-400 Hz)
nyquist = fs / 2
b_bandpass, a_bandpass = signal.butter(4, [20/nyquist, 400/nyquist], btype='bandpass')

# design bandstop filter (50 Hz notch)
b_stop, a_stop = signal.butter(2, [48/nyquist, 52/nyquist], btype='bandstop')

# ===== HELPERS: contiguous valid segments =====
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

segments = valid_segments(received_mask, min_len=1)
# filtfilt with a 4th-order Butterworth needs padlen = 3*(2*4+1)-1 = 26 samples,
# so only filter segments comfortably longer than that.
long_segments = valid_segments(received_mask, min_len=64)

# ===== FILTERING: per contiguous valid segment only =====
emg_filtered = np.full_like(recon, np.nan)
for (s, e) in long_segments:
    for i in range(recon.shape[1]):
        seg = recon[s:e, i]
        filtered = signal.filtfilt(b_bandpass, a_bandpass, seg)
        filtered = signal.filtfilt(b_stop, a_stop, filtered)
        emg_filtered[s:e, i] = filtered

# plot after filters (selected channels)
n_channels = len(plot_indices)

plt.figure(figsize=(15, 2.5 * n_channels))
for row, i in enumerate(plot_indices):
    plt.subplot(n_channels, 1, row+1)
    plt.plot(t, emg_filtered[:, i], 'r-', label='Filtered', linewidth=0.8)
    plt.ylabel(f'Ch{i+1}')
    if row == 0:
        plt.legend()
    if row == n_channels - 1:
        plt.xlabel('Time (s)')
    plt.grid(True, alpha=0.3)
    if args.ylim_mag:
        plt.ylim(args.ylim_mag)
plt.suptitle(f"Filtered EMG (gaps shown as gaps) - {csv_path}", fontsize=12)
plt.tight_layout()
plt.show()

# ===== FFT: Welch PSD computed per contiguous segment, then averaged =====
def welch_segment_average(x, fs, min_seg):
    """Average Welch PSD over contiguous non-NaN segments of length >= min_seg."""
    valid = ~np.isnan(x)
    nperseg = 1024
    psds = []
    weights = []
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
    # length-weighted average
    pxx_avg = np.average(np.stack(psds), axis=0, weights=weights)
    return f_ref, pxx_avg, len(psds)

# FFT of selected channels (gaps handled by segmenting)
plt.figure(figsize=(15, 2.5 * n_channels))
for row, i in enumerate(plot_indices):
    plt.subplot(n_channels, 1, row+1)

    f, pxx, n_segs = welch_segment_average(emg_filtered[:, i], fs, args.min_seg_fft)
    if pxx is not None:
        plt.semilogy(f, pxx, 'r-', label='Welch (contiguous segments)', linewidth=0.8)
        print(f"Ch{i+1}: FFT computed from {n_segs} segment(s) "
              f"(>= {args.min_seg_fft} samples, total {len(emg_filtered) - int(np.isnan(emg_filtered[:, i]).sum())} valid samples)")
    else:
        print(f"Ch{i+1}: no segment >= {args.min_seg_fft} samples; FFT skipped.")
    plt.ylabel(f'Ch{i+1}')
    if row == 0:
        plt.legend()
    if row == n_channels - 1:
        plt.xlabel('Frequency (Hz)')
    plt.xlim([0, 500])
    plt.grid(True, alpha=0.3)
    if args.ylim_fft:
        plt.ylim(args.ylim_fft)
plt.suptitle(f"Welch PSD (contiguous segments only) - {csv_path}", fontsize=12)
plt.tight_layout()
plt.show()