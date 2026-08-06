# apply 20 - 400 Hz bandpass
# apply 50 Hz bandstop
# Source file captures/sokosti_capture_20260806_110557.csv

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

# ===== PARAMETERS =====
# channels to plot (1-based). Examples:
#   [12]        -> only channel 12
#   [1, 5, 12]  -> channels 1, 5 and 12
#   None        -> all 16 channels
channels_to_plot = [12]
# ======================

# dataframe
df = pd.read_csv("captures/sokosti_capture_20260806_110557.csv")

# sampling frequency
fs = 1000  # Hz

# get EMG channels (ch1-ch16)
emg_channels = [f'ch{i}' for i in range(1, 17)]
emg_data = df[emg_channels].values

# select channels to plot (convert to 0-based indices)
if channels_to_plot is None:
    plot_indices = list(range(emg_data.shape[1]))
else:
    plot_indices = [c - 1 for c in channels_to_plot]

# time vector
t = np.arange(len(emg_data)) / fs

# design bandpass filter (20-400 Hz)
nyquist = fs / 2
b_bandpass, a_bandpass = signal.butter(4, [20/nyquist, 400/nyquist], btype='bandpass')

# design bandstop filter (50 Hz notch)
b_stop, a_stop = signal.butter(2, [45/nyquist, 55/nyquist], btype='bandstop')

# apply filters
emg_filtered = np.zeros_like(emg_data)
for i in range(emg_data.shape[1]):
    # apply bandpass
    filtered = signal.filtfilt(b_bandpass, a_bandpass, emg_data[:, i])
    # apply bandstop
    filtered = signal.filtfilt(b_stop, a_stop, filtered)
    emg_filtered[:, i] = filtered

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
plt.tight_layout()
plt.show()

# apply fft to check frequencies (selected channels)
# remove DC offset before FFT so the 0 Hz spike doesn't hide the spectrum
plt.figure(figsize=(15, 2.5 * n_channels))
for row, i in enumerate(plot_indices):
    plt.subplot(n_channels, 1, row+1)

    # FFT of original (DC removed)
    orig_detrended = emg_data[:, i] - np.mean(emg_data[:, i])
    fft_orig = np.fft.fft(orig_detrended)
    # FFT of filtered
    fft_filt = np.fft.fft(emg_filtered[:, i])
    # frequency axis (positive frequencies only)
    freq = np.fft.fftfreq(len(emg_data[:, i]), 1/fs)
    pos_freqs = freq >= 0
    plt.plot(freq[pos_freqs], np.abs(fft_orig[pos_freqs]), 'b-', alpha=0.5, label='Original', linewidth=0.5)
    plt.plot(freq[pos_freqs], np.abs(fft_filt[pos_freqs]), 'r-', label='Filtered', linewidth=0.8)
    plt.ylabel(f'Ch{i+1}')
    if row == 0:
        plt.legend()
    if row == n_channels - 1:
        plt.xlabel('Frequency (Hz)')
    plt.xlim([0, 500])
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()