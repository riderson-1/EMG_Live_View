# apply 20 - 400 Hz bandpass
# apply 48-52 Hz notch
# RMS EMG envelope: 100 sample window, 90 sample overlap (hop = 10)
# Plot 1 selected EMG channel (RMS envelope + raw filtered) plus all 6 IMU signals
# (accel_x/y/z and roll/pitch/yaw) in the same figure

import argparse
import sys
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import transforms as mtransforms

# ===== COMMAND LINE ARGUMENTS =====
parser = argparse.ArgumentParser(description="EMG RMS envelope + IMU plotting.")
parser.add_argument("csv", nargs="?", help="Path to the CSV file (defaults to the most recent capture).")
parser.add_argument("--channel", type=int, default=1, help="EMG channel to plot (1-based, default 1).")
parser.add_argument("--ylim_mag", type=float, nargs=2, help="Y-axis limits for the EMG plot (min max). Example: --ylim_mag -500 500")
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

# selected EMG channel (1-based)
ch_idx = args.channel - 1
if not 0 <= ch_idx <= 15:
    print(f"Invalid channel {args.channel}: must be 1-16.", file=sys.stderr)
    sys.exit(1)

# sampling frequency
fs = 1000  # Hz

# ===== FILTERING =====
emg = df[f'ch{args.channel}'].values.astype(float)

# design bandpass filter (20-400 Hz)
nyquist = fs / 2
b_bandpass, a_bandpass = signal.butter(4, [20/nyquist, 400/nyquist], btype='bandpass')

# design bandstop filter (48-52 Hz notch)
b_stop, a_stop = signal.butter(2, [48/nyquist, 52/nyquist], btype='bandstop')

# apply filters
emg_filtered = signal.filtfilt(b_bandpass, a_bandpass, emg)
emg_filtered = signal.filtfilt(b_stop, a_stop, emg_filtered)

# ===== RMS ENVELOPE =====
# 100 sample window, 90 sample overlap -> hop of 10 samples
win = 100
hop = 10
n_frames = 1 + (len(emg_filtered) - win) // hop
rms_env = np.empty(n_frames)
for k in range(n_frames):
    seg = emg_filtered[k*hop : k*hop + win]
    rms_env[k] = np.sqrt(np.mean(seg**2))
# time axis of frame centers
t_env = (np.arange(n_frames) * hop + win/2) / fs

# ===== IMU DATA =====
imu_channels = ['accel_x', 'accel_y', 'accel_z', 'roll', 'pitch', 'yaw']
imu_labels = ['Accel X', 'Accel Y', 'Accel Z', 'Roll', 'Pitch', 'Yaw']

# time vector
t = np.arange(len(emg_filtered)) / fs

# ===== PLOTTING =====
# Everything in one axes, with separate y-axes for the different scales:
#   left  -> EMG (filtered + RMS envelope)
#   middle-> accelerations (g or m/s^2 depending on device config)
#   right -> orientation angles (deg)
fig, ax_emg = plt.subplots(figsize=(15, 7))

# EMG (left axis)
# l_filt, = ax_emg.plot(t, emg_filtered, 'r-', linewidth=0.6, label=f'Ch{args.channel} filtered')
l_env, = ax_emg.plot(t_env, rms_env, 'b-', linewidth=1.2, label=f'Ch{args.channel} RMS (100/10)')
ax_emg.set_ylabel(f'EMG Ch{args.channel} (raw units)')
ax_emg.grid(True, alpha=0.3)
if args.ylim_mag:
    ax_emg.set_ylim(args.ylim_mag)

# Accelerations (middle axis, offset right)
ax_acc = ax_emg.twinx()
acc_colors = ['tab:orange', 'tab:green', 'tab:red']
acc_lines = []
for col, label, c in zip(imu_channels[:3], imu_labels[:3], acc_colors):
    vals = df[col].values.astype(float)
    (ln,) = ax_acc.plot(t, vals, color=c, linewidth=0.8, label=label)
    acc_lines.append(ln)
ax_acc.set_ylabel('Acceleration')
ax_acc.yaxis.label.set_color('tab:orange')

# Orientation angles (right axis, further offset)
ax_ang = ax_emg.twinx()
ax_ang.spines['right'].set_position(('outward', 70))
ang_colors = ['tab:purple', 'tab:brown', 'tab:pink']
for col, label, c in zip(imu_channels[3:], imu_labels[3:], ang_colors):
    vals = df[col].values.astype(float)
    (ln,) = ax_ang.plot(t, vals, color=c, linewidth=0.8, label=label)
    acc_lines.append(ln)
ax_ang.set_ylabel('Angle (deg)')
ax_ang.yaxis.label.set_color('tab:purple')

ax_emg.set_xlabel('Time (s)')
# lines = [l_filt, l_env] + acc_lines
lines = [l_env] + acc_lines
ax_emg.legend(lines, [l.get_label() for l in lines], loc='upper right', fontsize=8, ncol=2)
plt.title(f"EMG RMS + IMU - {csv_path}")
fig.tight_layout()

# ===== INTERACTIVE GUI: per-signal scale/offset sliders in a separate control window =====
# store raw IMU data (one entry per signal: accel_x/y/z, roll/pitch/yaw)
acc_raw = [df[c].values.astype(float) for c in imu_channels[:3]]
ang_raw = [df[c].values.astype(float) for c in imu_channels[3:]]
all_raw = acc_raw + ang_raw
all_labels = imu_labels

# per-signal scale/offset
scales = [1.0] * 6
offsets = [0.0] * 6

def update(_=None):
    for ln, raw, s, o in zip(acc_lines, all_raw, scales, offsets):
        ln.set_ydata(raw * s + o)
    fig.canvas.draw_idle()

# Axis limits stay FIXED to the raw data range (plus a small margin),
# completely independent of the slider ranges below. When you scale or
# offset a curve beyond the limits it simply clips at the top/bottom of
# the plot, so increasing the scale visibly grows the IMU curves
# relative to the EMG while the axes stay small.
def data_range(raw_list, margin=0.1):
    allv = np.concatenate(raw_list)
    allv = allv[np.isfinite(allv)]
    lo, hi = allv.min(), allv.max()
    pad = (hi - lo) * margin or 1.0
    return lo - pad, hi + pad

ax_acc.set_ylim(*data_range(acc_raw))
ax_ang.set_ylim(*data_range(ang_raw))

# Slider ranges (independent of the axis limits above; tune freely).
# NOTE: a very large offset range makes the offset sliders coarse to drag;
# lower acc_span/ang_span if you need finer offset control.
acc_span = 10000
ang_span = 10000

# control window: 12 sliders (scale + offset for each of the 6 signals)
fig_ctrl, ctrl_axes = plt.subplots(figsize=(6, 8))
fig_ctrl.canvas.manager.set_window_title('IMU Scale/Offset Controls')
ctrl_axes.axis('off')
ctrl_axes.text(0.5, 0.98, 'IMU Scale / Offset (per signal)', ha='center', va='top', fontsize=11)

def add_slider(y, label, vmin, vmax, vinit):
    sax = fig_ctrl.add_axes([0.32, y, 0.55, 0.02])
    return Slider(sax, label, vmin, vmax, valinit=vinit)

def make_pair(y, name, vmax_scale, vmax_offset, idx):
    s_scale = add_slider(y, f'{name} scale', 0.1, vmax_scale, 1.0)
    s_offset = add_slider(y - 0.04, f'{name} offset', -vmax_offset, vmax_offset, 0.0)
    s_scale.on_changed(lambda v, i=idx: (scales.__setitem__(i, v), update()))
    s_offset.on_changed(lambda v, i=idx: (offsets.__setitem__(i, v), update()))
    slider_refs.append(s_scale)
    slider_refs.append(s_offset)

# accel sliders (top block), angle sliders (bottom block)
slider_refs = []
top, row_h = 0.88, 0.11
for i, label in enumerate(imu_labels[:3]):
    make_pair(top - i * row_h, label, 100.0, acc_span, i)
base = top - 3 * row_h - 0.06
for j, label in enumerate(imu_labels[3:]):
    make_pair(base - j * row_h, label, 100.0, ang_span, 3 + j)

# initialize limits with default scaling
update()

plt.show()
