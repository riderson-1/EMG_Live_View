#!/bin/bash
# Rerun the 12 Sit_Stand analyses with the exact settings from the existing
# *_imu_analysis.log files (gains, transition times, default ch1, percent
# axis), regenerating logs + PNGs + the per-trial pipeline CSVs
# (<stem>_signals.csv and <stem>_emg.csv via --save-trial-csv).
#
# Sample rate is 980 sps (NOT 1000): --emg-fs 980 is appended by run().
# All transition times were scaled by 1000/980 so they select the SAME
# samples as the old fs=1000 runs (same convention as rerun_iso_dor.sh).
set -u
TOOLS_DIR=$(cd "$(dirname "$0")" && pwd)
SCRIPT=$TOOLS_DIR/sit_stand_analysis.py
VENV=$TOOLS_DIR/../.venv/bin/python
BASE=/home/karl/Documents/Master_Thesis/Testing/application_testing

run() {
  echo "=== RUN: $1"
  "$VENV" "$SCRIPT" "$2" "${@:3}" --emg-fs 980 --channel 1 --time-axis percent \
      --no-plot --save-results --save-trial-csv || echo "!!! FAILED: $1"
}

D0209=$BASE/EMG_Testing_02092026
D0809=$BASE/EMG_Testing_08092026
D1409=$BASE/EMG_Testing_14092026

# --- 02092026 USB_1_Karl (gain 1) ---
run USB_1_Karl_PC \
  $D0209/USB_1_Karl_Sit_Stand_2026-09-02/USB_1_Karl_Sit_Stand_2026-09-02_PC.csv \
  --gain 1 --transition-times '27.0408, 39.2857, 42.7551, 54.7959, 58.3673, 70.8163, 76.3265, 87.7551'
run USB_1_Karl_SD \
  $D0209/USB_1_Karl_Sit_Stand_2026-09-02/USB_1_Karl_Sit_Stand_2026-09-02_SD.csv \
  --gain 1 --transition-times '27.0408, 39.2857, 42.7551, 54.7959, 58.3673, 70.8163, 76.3265, 87.7551'

# --- 02092026 USB_1_Max (gain 1) ---
run USB_1_Max_PC \
  $D0209/USB_1_Max_Sit_Stand_2026-09-02/USB_1_Max_Sit_Stand_2026-09-02_PC.csv \
  --gain 1 --transition-times '161.2245, 174.0816, 177.2449, 191.1224, 194.0816, 206.5306, 209.0816, 221.3265'
run USB_1_Max_SD \
  $D0209/USB_1_Max_Sit_Stand_2026-09-02/USB_1_Max_Sit_Stand_2026-09-02_SD.csv \
  --gain 1 --transition-times '161.2245, 174.0816, 177.2449, 191.1224, 194.0816, 206.5306, 209.0816, 221.3265'

# --- 08092026 BLE_1_Karl (gain 1) ---
run BLE_1_Karl_PC \
  $D0809/BLE_1_Karl_Sit_Stand_2026-09-08/BLE_1_Karl_Sit_Stand_2026-09-08_PC.csv \
  --gain 1 --transition-times '654.6939, 666.4286, 670.102, 683.1633, 686.2245, 697.9592, 702.3469, 714.1837'
run BLE_1_Karl_SD \
  $D0809/BLE_1_Karl_Sit_Stand_2026-09-08/BLE_1_Karl_Sit_Stand_2026-09-08_SD.csv \
  --gain 1 --transition-times '654.6939, 666.4286, 670.102, 683.1633, 686.2245, 697.9592, 702.3469, 714.1837'

# --- 14092026 BLE_8_Max (gain 8) ---
run BLE_8_Max_PC \
  $D1409/BLE_8_Max_Sit_Stand_2026-09-14/BLE_8_Max_Sit_Stand_2026-09-14_PC.csv \
  --gain 8 --transition-times '148.5714, 160.8163, 163.1633, 176.0204, 177.7551, 190.9184, 193.5714, 206.0204'
run BLE_8_Max_SD \
  $D1409/BLE_8_Max_Sit_Stand_2026-09-14/BLE_8_Max_Sit_Stand_2026-09-14_SD.csv \
  --gain 8 --transition-times '148.5714, 160.8163, 163.1633, 176.0204, 177.7551, 190.9184, 193.5714, 206.0204'

# --- 14092026 BLE_1_Max (gain 1) ---
run BLE_1_Max_PC \
  $D1409/BLE_1_Max_Sit_Stand_2026-09-14/BLE_1_Max_Sit_Stand_2026-09-14_PC.csv \
  --gain 1 --transition-times '521.7347, 534.4898, 537.0408, 550.7143, 553.2653, 565.102, 568.7755, 581.6327'
run BLE_1_Max_SD \
  $D1409/BLE_1_Max_Sit_Stand_2026-09-14/BLE_1_Max_Sit_Stand_2026-09-14_SD.csv \
  --gain 1 --transition-times '521.7347, 534.4898, 537.0408, 550.7143, 553.2653, 565.102, 568.7755, 581.6327'

# --- 14092026 BLE_8_Karl (gain 8) ---
run BLE_8_Karl_PC \
  $D1409/BLE_8_Karl_Sit_Stand_2026-09-14/BLE_8_Karl_Sit_Stand_2026-09-14_PC.csv \
  --gain 8 --transition-times '1256.8367, 1269.2857, 1272.9592, 1284.5918, 1288.6735, 1299.898, 1303.6735, 1315.9184'
run BLE_8_Karl_SD \
  $D1409/BLE_8_Karl_Sit_Stand_2026-09-14/BLE_8_Karl_Sit_Stand_2026-09-14_SD.csv \
  --gain 8 --transition-times '1256.8367, 1269.2857, 1272.9592, 1284.5918, 1288.6735, 1299.898, 1303.6735, 1315.9184'

echo "=== ALL DONE ==="
