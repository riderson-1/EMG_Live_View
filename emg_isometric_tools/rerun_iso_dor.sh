#!/bin/bash
# Rerun the 12 Iso_Dor analyses with corrected command lines
# (python prefix, quoted window lists, fixed flag typos).
set -u
TOOLS_DIR=$(cd "$(dirname "$0")" && pwd)
SCRIPT=$TOOLS_DIR/emg_isometric.py
VENV=$TOOLS_DIR/../.venv/bin/python
BASE=/home/karl/Documents/Master_Thesis/Testing/application_testing

run() {
  echo "=== RUN: $1"
  "$VENV" "$SCRIPT" "$2" "${@:3}" --save-results || echo "!!! FAILED: $1"
}

D0209=$BASE/EMG_Testing_02092026
D0809=$BASE/EMG_Testing_08092026
D1409=$BASE/EMG_Testing_14092026

# --- 02092026 USB_1_Max (ch2, thresh-strong 40, gain 1) ---
run USB_1_Max_PC \
  $D0209/USB_1_Max_Iso_Dor_2026-09-02/USB_1_Max_Iso_Dor_2026-09-02_PC.csv \
  --channels 2 --thresh-strong 40 --gain 1 --compare-channels --psd fft \
  --show-windows --signal-trim 0.05 --noise-offset 1.5 --noise-window 50
run USB_1_Max_SD_cropped \
  $D0209/USB_1_Max_Iso_Dor_2026-09-02/USB_1_Max_Iso_Dor_2026-09-02_SD_cropped.csv \
  --channels 2 --thresh-strong 40 --gain 1 --compare-channels --psd fft \
  --show-windows --signal-trim 0.05 --noise-offset 1.5 --noise-window 50

# --- 02092026 USB_1_Karl (ch13, thresh-strong 30, gain 1) ---
run USB_1_Karl_PC \
  $D0209/USB_1_Karl_Iso_Dor_2026-09-02/USB_1_Karl_Iso_Dor_2026-09-02_PC.csv \
  --channels 13 --thresh-strong 30 --gain 1 --compare-channels --psd fft \
  --signal-trim 0.05 --noise-offset 1.5 --noise-window 50 --show-windows
run USB_1_Karl_SD_cropped \
  $D0209/USB_1_Karl_Iso_Dor_2026-09-02/USB_1_Karl_Iso_Dor_2026-09-02_SD_cropped.csv \
  --channels 13 --thresh-strong 30 --gain 1 --compare-channels --psd fft \
  --signal-trim 0.05 --noise-offset 1.5 --noise-window 50 --show-windows

# --- 08092026 BLE_1_Karl (gain 1) ---
run BLE_1_Karl_PC \
  $D0809/BLE_1_Karl_Iso_Dor_2026-09-08/BLE_1_Karl_Iso_Dor_2026-09-08_PC.csv \
  --thresh-strong 30 --gain 1 --channels 15 --compare-channels \
  --strong-windows "15-20, 29.5-34.5, 45-50, 61.5-66.5" \
  --weak-windows "77-106.5, 126-156, 175-205, 223.5-253.5" \
  --psd fft --show-windows --signal-trim 0.05 --noise-offset 1.5 --noise-window 50
run BLE_1_Karl_SD_shortened \
  $D0809/BLE_1_Karl_Iso_Dor_2026-09-08/BLE_1_Karl_Iso_Dor_2026-09-08_SD_shortened.csv \
  --thresh-strong 30 --gain 1 --channels 16 --compare-channels \
  --strong-windows "21.5-26.5, 36-41, 51.5-56.5, 68-73" \
  --weak-windows "83.5-113, 132.5-162.5, 181.5-211.5,230-260" \
  --psd fft --show-windows --signal-trim 0.05 --noise-offset 1.5 --noise-window 50

# --- 14092026 BLE_8_Max (gain 8, ch13, thresh-strong 30) ---
run BLE_8_Max_PC \
  $D1409/BLE_8_Max_Iso_Dor_2026-09-14/BLE_8_Max_Iso_Dor_2026-09-14_PC.csv \
  --thresh-strong 30 --gain 8 --channels 13 --compare-channels \
  --strong-windows "21.6-27.6, 46.5-52.5, 71.3-77.3, 95.9-101.9" \
  --weak-windows "120.6-151.0, 171.0-200.6, 218.9-249.6, 268.2-299.6" \
  --psd fft --show-windows --signal-trim 0.05 --noise-offset 1.5 --noise-window 50
run BLE_8_Max_SD \
  $D1409/BLE_8_Max_Iso_Dor_2026-09-14/BLE_8_Max_Iso_Dor_2026-09-14_SD.csv \
  --thresh-strong 30 --gain 8 --channels 13 --compare-channels \
  --strong-windows "26.4-32.4, 51.3-57.3, 76.1-82.1, 100.7-106.7" \
  --weak-windows "125.4-155.8, 175.8-205.4, 223.7-254.4, 273.0-304.4" \
  --psd fft --show-windows --signal-trim 0.05 --noise-offset 1.5 --noise-window 50

# --- 14092026 BLE_8_Karl (gain 8, ch11, thresh-strong 40) ---
# (original logs had --strong-window/--weak-window singular: fixed to plural)
run BLE_8_Karl_PC \
  $D1409/BLE_8_Karl_Iso_Dor_2026-09-14/BLE_8_Karl_Iso_Dor_2026-09-14_PC.csv \
  --thresh-strong 40 --thresh-weak 5 --gain 8 --channels 11 --compare-channels \
  --strong-windows "9.4-14.6, 34.5-39.6, 59.4-64.5, 85.3-90.6" \
  --weak-windows "111.1-142.0, 160.7-191.8, 210.2-240.5, 259.8-289.9" \
  --psd fft --show-windows --signal-trim 0.05 --noise-offset 1.5 --noise-window 50
run BLE_8_Karl_SD_cropped \
  $D1409/BLE_8_Karl_Iso_Dor_2026-09-14/BLE_8_Karl_Iso_Dor_2026-09-14_SD_cropped.csv \
  --thresh-strong 40 --thresh-weak 5 --gain 8 --channels 11 --compare-channels \
  --strong-windows "9.4-14.6, 34.5-39.6, 59.4-64.5, 85.3-90.6" \
  --weak-windows "111.1-142.0, 160.7-191.8, 210.2-240.5, 259.8-289.9" \
  --psd fft --show-windows --signal-trim 0.05 --noise-offset 1.5 --noise-window 50

# --- 14092026 BLE_1_Max (gain 1, ch3, thresh-strong 40) ---
run BLE_1_Max_PC \
  $D1409/BLE_1_Max_Iso_Dor_2026-09-14/BLE_1_Max_Iso_Dor_2026-09-14_PC.csv \
  --thresh-strong 40 --thresh-weak 5 --gain 1 --channels 3 --compare-channels \
  --strong-windows "16.8-22.2, 41.5-46.9, 66.5-71.8, 92.0-96.9" \
  --weak-windows "115.6-148.7, 167.0-197.7, 216.3-247.3, 265.8-296.8" \
  --psd fft --show-windows --signal-trim 0.05 --noise-offset 1.5 --noise-window 50
run BLE_1_Max_SD \
  $D1409/BLE_1_Max_Iso_Dor_2026-09-14/BLE_1_Max_Iso_Dor_2026-09-14_SD.csv \
  --thresh-strong 40 --thresh-weak 5 --gain 1 --channels 3 --compare-channels \
  --strong-windows "22.3-27.7, 47.0-52.4, 72.0-77.3, 97.5-102.4" \
  --weak-windows "121.1-154.2, 172.5-203.2, 221.8-252.8, 271.3-302.3" \
  --psd fft --show-windows --signal-trim 0.05 --noise-offset 1.5 --noise-window 50

echo "=== ALL DONE ==="
