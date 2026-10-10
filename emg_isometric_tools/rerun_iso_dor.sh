#!/bin/bash
# Rerun the 12 Iso_Dor analyses with corrected command lines
# (python prefix, quoted window lists, fixed flag typos).
set -u
# Non-interactive matplotlib backend: no figure windows pop up
# (plt.show() becomes a no-op, figures are still saved to disk).
export MPLBACKEND=Agg
TOOLS_DIR=$(cd "$(dirname "$0")" && pwd)
SCRIPT=$TOOLS_DIR/emg_isometric.py
VENV=$TOOLS_DIR/../.venv/bin/python
BASE=/home/karl/Documents/Master_Thesis/Testing/application_testing

# Sample rate is 980 sps (NOT 1000): --fs 980 is appended by run().
# All time values below (window lists, --noise-offset, and the USB
# auto-detection thresholds) were scaled by 1000/980 so that every
# signal/noise window selects the SAME samples as the old fs=1000 runs.
run() {
  echo "=== RUN: $1"
  "$VENV" "$SCRIPT" "$2" "${@:3}" --fs 980 --save-results || echo "!!! FAILED: $1"
}

D0209=$BASE/EMG_Testing_02092026
D0809=$BASE/EMG_Testing_08092026
D1409=$BASE/EMG_Testing_14092026

# --- 02092026 USB_1_Max (ch2, thresh-strong 40, gain 1) ---
run USB_1_Max_PC \
  $D0209/USB_1_Max_Iso_Dor_2026-09-02/USB_1_Max_Iso_Dor_2026-09-02_PC.csv \
  --channels 2 --thresh-strong 40 --gain 1 --merge-gap 0.511 --strong-min 2.0405 --weak-min 10.2035 --compare-channels --psd fft \
  --show-windows --signal-trim 0.05 --noise-offset 1.530612 --noise-window 50
run USB_1_Max_SD_cropped \
  $D0209/USB_1_Max_Iso_Dor_2026-09-02/USB_1_Max_Iso_Dor_2026-09-02_SD_cropped.csv \
  --channels 2 --thresh-strong 40 --gain 1 --merge-gap 0.511 --strong-min 2.0405 --weak-min 10.2035 --compare-channels --psd fft \
  --show-windows --signal-trim 0.05 --noise-offset 1.530612 --noise-window 50

# --- 02092026 USB_1_Karl (ch13, thresh-strong 30, gain 1) ---
run USB_1_Karl_PC \
  $D0209/USB_1_Karl_Iso_Dor_2026-09-02/USB_1_Karl_Iso_Dor_2026-09-02_PC.csv \
  --channels 13 --thresh-strong 30 --gain 1 --merge-gap 0.511 --strong-min 2.0405 --weak-min 10.2035 --compare-channels --psd fft \
  --signal-trim 0.05 --noise-offset 1.530612 --noise-window 50 --show-windows
run USB_1_Karl_SD_cropped \
  $D0209/USB_1_Karl_Iso_Dor_2026-09-02/USB_1_Karl_Iso_Dor_2026-09-02_SD_cropped.csv \
  --channels 13 --thresh-strong 30 --gain 1 --merge-gap 0.511 --strong-min 2.0405 --weak-min 10.2035 --compare-channels --psd fft \
  --signal-trim 0.05 --noise-offset 1.530612 --noise-window 50 --show-windows

# --- 08092026 BLE_1_Karl (gain 1) ---
run BLE_1_Karl_PC \
  $D0809/BLE_1_Karl_Iso_Dor_2026-09-08/BLE_1_Karl_Iso_Dor_2026-09-08_PC.csv \
  --thresh-strong 30 --gain 1 --channels 15 --compare-channels \
  --strong-windows "15.306123-20.408164, 30.102041-35.204082, 45.918368-51.020409, 62.755103-67.857143" \
  --weak-windows "78.571429-108.67347, 128.571429-159.183674, 178.571429-209.183674, 228.061225-258.67347" \
  --psd fft --show-windows --signal-trim 0.05 --noise-offset 1.530612 --noise-window 50
run BLE_1_Karl_SD_shortened \
  $D0809/BLE_1_Karl_Iso_Dor_2026-09-08/BLE_1_Karl_Iso_Dor_2026-09-08_SD_shortened.csv \
  --thresh-strong 30 --gain 1 --channels 16 --compare-channels \
  --strong-windows "21.938776-27.040817, 36.734694-41.836735, 52.551021-57.653062, 69.387756-74.489796" \
  --weak-windows "85.204082-115.306123, 135.204082-165.816327, 185.204082-215.816327, 234.693878-265.306123" \
  --psd fft --show-windows --signal-trim 0.05 --noise-offset 1.530612 --noise-window 50

# --- 14092026 BLE_8_Max (gain 8, ch13, thresh-strong 30) ---
run BLE_8_Max_PC \
  $D1409/BLE_8_Max_Iso_Dor_2026-09-14/BLE_8_Max_Iso_Dor_2026-09-14_PC.csv \
  --thresh-strong 30 --gain 8 --channels 13 --compare-channels \
  --strong-windows "22.040817-28.163266, 47.44898-53.571429, 72.755103-78.877552, 97.857143-103.979592" \
  --weak-windows "123.061225-154.081633, 174.489796-204.693878, 223.367347-254.693878, 273.67347-305.714286" \
  --psd fft --show-windows --signal-trim 0.05 --noise-offset 1.530612 --noise-window 50
run BLE_8_Max_SD \
  $D1409/BLE_8_Max_Iso_Dor_2026-09-14/BLE_8_Max_Iso_Dor_2026-09-14_SD.csv \
  --thresh-strong 30 --gain 8 --channels 13 --compare-channels \
  --strong-windows "26.938776-33.061225, 52.346939-58.469388, 77.653062-83.775511, 102.755103-108.877552" \
  --weak-windows "127.959184-158.979592, 179.387756-209.591837, 228.265307-259.591837, 278.571429-310.612245" \
  --psd fft --show-windows --signal-trim 0.05 --noise-offset 1.530612 --noise-window 50

# --- 14092026 BLE_8_Karl (gain 8, ch11, thresh-strong 40) ---
# (original logs had --strong-window/--weak-window singular: fixed to plural)
run BLE_8_Karl_PC \
  $D1409/BLE_8_Karl_Iso_Dor_2026-09-14/BLE_8_Karl_Iso_Dor_2026-09-14_PC.csv \
  --thresh-strong 40 --thresh-weak 5 --gain 8 --channels 11 --compare-channels \
  --strong-windows "9.591837-14.89796, 35.204082-40.408164, 60.612245-65.816327, 87.040817-92.44898" \
  --weak-windows "113.367347-144.89796, 163.979592-195.714286, 214.489796-245.408164, 265.102041-295.816327" \
  --psd fft --show-windows --signal-trim 0.05 --noise-offset 1.530612 --noise-window 50
run BLE_8_Karl_SD_cropped \
  $D1409/BLE_8_Karl_Iso_Dor_2026-09-14/BLE_8_Karl_Iso_Dor_2026-09-14_SD_cropped.csv \
  --thresh-strong 40 --thresh-weak 5 --gain 8 --channels 11 --compare-channels \
  --strong-windows "9.591837-14.89796, 35.204082-40.408164, 60.612245-65.816327, 87.040817-92.44898" \
  --weak-windows "113.367347-144.89796, 163.979592-195.714286, 214.489796-245.408164, 265.102041-295.816327" \
  --psd fft --show-windows --signal-trim 0.05 --noise-offset 1.530612 --noise-window 50

# --- 14092026 BLE_1_Max (gain 1, ch3, thresh-strong 40) ---
run BLE_1_Max_PC \
  $D1409/BLE_1_Max_Iso_Dor_2026-09-14/BLE_1_Max_Iso_Dor_2026-09-14_PC.csv \
  --thresh-strong 40 --thresh-weak 5 --gain 1 --channels 3 --compare-channels \
  --strong-windows "17.142858-22.653062, 42.346939-47.857143, 67.857143-73.265307, 93.877552-98.877552" \
  --weak-windows "117.959184-151.734694, 170.408164-201.734694, 220.714286-252.346939, 271.22449-302.857143" \
  --psd fft --show-windows --signal-trim 0.05 --noise-offset 1.530612 --noise-window 50
run BLE_1_Max_SD \
  $D1409/BLE_1_Max_Iso_Dor_2026-09-14/BLE_1_Max_Iso_Dor_2026-09-14_SD.csv \
  --thresh-strong 40 --thresh-weak 5 --gain 1 --channels 3 --compare-channels \
  --strong-windows "22.755103-28.265307, 47.959184-53.469388, 73.469388-78.877552, 99.489796-104.489796" \
  --weak-windows "123.571429-157.346939, 176.020409-207.346939, 226.326531-257.959184, 276.836735-308.469388" \
  --psd fft --show-windows --signal-trim 0.05 --noise-offset 1.530612 --noise-window 50

# --- Stage 2: parse the fresh logs into the summary workbook ---
echo "=== PARSE: emg_isometric_summary.ods"
"$VENV" "$TOOLS_DIR/parse_emg_logs.py" --out "$TOOLS_DIR/emg_isometric_summary.ods" \
  || echo "!!! FAILED: parse_emg_logs"

# --- Stage 3: pooled all-channel plots (SNR + MDF) ---
echo "=== PLOT: SNR all-channel figures"
"$VENV" "$TOOLS_DIR/plot_isometric_trials_allch.py" \
  "$TOOLS_DIR/emg_isometric_summary.ods" --out-dir "$TOOLS_DIR" --no-show \
  || echo "!!! FAILED: plot_isometric_trials_allch"
echo "=== PLOT: MDF all-channel figures"
"$VENV" "$TOOLS_DIR/plot_isometric_mdf_allch.py" \
  "$TOOLS_DIR/emg_isometric_summary.ods" --out-dir "$TOOLS_DIR" --no-show \
  || echo "!!! FAILED: plot_isometric_mdf_allch"

echo "=== ALL DONE ==="
