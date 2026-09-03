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

# ===== INTERACTIVE GUI: scale/offset sliders =====
# store raw IMU data
acc_raw = [df[c].values.astype(float) for c in imu_channels[:3]]
ang_raw = [df[c].values.astype(float) for c in imu_channels[3:]]

state = {
    'scale_acc': 1.0, 'offset_acc': 0.0,
    'scale_ang': 1.0, 'offset_ang': 0.0,
}

def update(_=None):
    sa, oa = state['scale_acc'], state['offset_acc']
    sg, og = state['scale_ang'], state['offset_ang']
    for ln, raw in zip(acc_lines[:3], acc_raw):
        ln.set_ydata(raw * sa + oa)
    for ln, raw in zip(acc_lines[3:], ang_raw):
        ln.set_ydata(raw * sg + og)
    fig.canvas.draw_idle()

# FIXED axis limits so slider changes are actually visible.
# Computed from the raw data range, expanded to cover the full slider
# scale/offset ranges (scale up to 10x, offset up to +/- data span).
def fixed_ylim(raw_list, max_scale, max_offset):
    allv = np.concatenate(raw_list)
    allv = allv[np.isfinite(allv)]
    lo, hi = allv.min(), allv.max()
    cand = [lo * max_scale + max_offset, lo * max_scale - max_offset,
            hi * max_scale + max_offset, hi * max_scale - max_offset,
            lo, hi]
    return min(cand), max(cand)

acc_span = float(np.nanmax(np.abs(np.concatenate(acc_raw))))
ang_span = float(np.nanmax(np.abs(np.concatenate(ang_raw))))
ax_acc.set_ylim(*fixed_ylim(acc_raw, 10.0, acc_span))
ax_ang.set_ylim(*fixed_ylim(ang_raw, 10.0, ang_span))

# slider axes at the bottom of the figure
plt.subplots_adjust(bottom=0.28)
def add_slider(y, label, vmin, vmax, vinit):
    sax = plt.axes([0.15, y, 0.6, 0.02])
    return Slider(sax, label, vmin, vmax, valinit=vinit)

s_scale_acc = add_slider(0.19, 'acc scale', 0.1, 50.0, 1.0)
s_offset_acc = add_slider(0.15, 'acc offset', -5000, 5000, 0.0)
s_scale_ang = add_slider(0.11, 'ang scale', 0.1, 50.0, 1.0)
s_offset_ang = add_slider(0.07, 'ang offset', -5000, 5000, 0.0)

def on_scale_acc(v): state['scale_acc'] = v; update()
def on_offset_acc(v): state['offset_acc'] = v; update()
def on_scale_ang(v): state['scale_ang'] = v; update()
def on_offset_ang(v): state['offset_ang'] = v; update()

s_scale_acc.on_changed(on_scale_acc)
s_offset_acc.on_changed(on_offset_acc)
s_scale_ang.on_changed(on_scale_ang)
s_offset_ang.on_changed(on_offset_ang)

# initialize limits with default scaling
update()

plt.tight_layout()
plt.show()
