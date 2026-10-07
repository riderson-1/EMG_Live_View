#!/bin/bash
# Rerun the 12 Sit_Stand analyses with the exact settings from the existing
# *_imu_analysis.log files (gains, transition times, default ch1, percent
# axis), regenerating logs + PNGs + the per-trial pipeline CSVs
# (<stem>_signals.csv and <stem>_emg.csv via --save-trial-csv).
set -u
TOOLS_DIR=$(cd "$(dirname "$0")" && pwd)
SCRIPT=$TOOLS_DIR/sit_stand_analysis.py
VENV=$TOOLS_DIR/../.venv/bin/python
BASE=/home/karl/Documents/Master_Thesis/Testing/application_testing

run() {
  echo "=== RUN: $1"
  "$VENV" "$SCRIPT" "$2" "${@:3}" --channel 1 --time-axis percent \
      --no-plot --save-results --save-trial-csv || echo "!!! FAILED: $1"
}

D0209=$BASE/EMG_Testing_02092026
D0809=$BASE/EMG_Testing_08092026
D1409=$BASE/EMG_Testing_14092026

# --- 02092026 USB_1_Karl (gain 1) ---
run USB_1_Karl_PC \
  $D0209/USB_1_Karl_Sit_Stand_2026-09-02/USB_1_Karl_Sit_Stand_2026-09-02_PC.csv \
  --gain 1 --transition-times '26.5, 38.5, 41.9, 53.7, 57.2, 69.4, 74.8, 86'
run USB_1_Karl_SD \
  $D0209/USB_1_Karl_Sit_Stand_2026-09-02/USB_1_Karl_Sit_Stand_2026-09-02_SD.csv \
  --gain 1 --transition-times '26.5, 38.5, 41.9, 53.7, 57.2, 69.4, 74.8, 86'

# --- 02092026 USB_1_Max (gain 1) ---
run USB_1_Max_PC \
  $D0209/USB_1_Max_Sit_Stand_2026-09-02/USB_1_Max_Sit_Stand_2026-09-02_PC.csv \
  --gain 1 --transition-times '158, 170.6, 173.7, 187.3, 190.2, 202.4, 204.9, 216.9'
run USB_1_Max_SD \
  $D0209/USB_1_Max_Sit_Stand_2026-09-02/USB_1_Max_Sit_Stand_2026-09-02_SD.csv \
  --gain 1 --transition-times '158, 170.6, 173.7, 187.3, 190.2, 202.4, 204.9, 216.9'

# --- 08092026 BLE_1_Karl (gain 1) ---
run BLE_1_Karl_PC \
  $D0809/BLE_1_Karl_Sit_Stand_2026-09-08/BLE_1_Karl_Sit_Stand_2026-09-08_PC.csv \
  --gain 1 --transition-times '641.6, 653.1, 656.7, 669.5, 672.5, 684, 688.3, 699.9'
run BLE_1_Karl_SD \
  $D0809/BLE_1_Karl_Sit_Stand_2026-09-08/BLE_1_Karl_Sit_Stand_2026-09-08_SD.csv \
  --gain 1 --transition-times '641.6, 653.1, 656.7, 669.5, 672.5, 684, 688.3, 699.9'

# --- 14092026 BLE_8_Max (gain 8) ---
run BLE_8_Max_PC \
  $D1409/BLE_8_Max_Sit_Stand_2026-09-14/BLE_8_Max_Sit_Stand_2026-09-14_PC.csv \
  --gain 8 --transition-times '145.6, 157.6, 159.9, 172.5, 174.2, 187.1, 189.7, 201.9'
run BLE_8_Max_SD \
  $D1409/BLE_8_Max_Sit_Stand_2026-09-14/BLE_8_Max_Sit_Stand_2026-09-14_SD.csv \
  --gain 8 --transition-times '145.6, 157.6, 159.9, 172.5, 174.2, 187.1, 189.7, 201.9'

# --- 14092026 BLE_1_Max (gain 1) ---
run BLE_1_Max_PC \
  $D1409/BLE_1_Max_Sit_Stand_2026-09-14/BLE_1_Max_Sit_Stand_2026-09-14_PC.csv \
  --gain 1 --transition-times '511.3, 523.8, 526.3, 539.7, 542.2, 553.8, 557.4, 570'
run BLE_1_Max_SD \
  $D1409/BLE_1_Max_Sit_Stand_2026-09-14/BLE_1_Max_Sit_Stand_2026-09-14_SD.csv \
  --gain 1 --transition-times '511.3, 523.8, 526.3, 539.7, 542.2, 553.8, 557.4, 570'

# --- 14092026 BLE_8_Karl (gain 8) ---
run BLE_8_Karl_PC \
  $D1409/BLE_8_Karl_Sit_Stand_2026-09-14/BLE_8_Karl_Sit_Stand_2026-09-14_PC.csv \
  --gain 8 --transition-times '1231.7, 1243.9, 1247.5, 1258.9, 1262.9, 1273.9, 1277.6, 1289.6'
run BLE_8_Karl_SD \
  $D1409/BLE_8_Karl_Sit_Stand_2026-09-14/BLE_8_Karl_Sit_Stand_2026-09-14_SD.csv \
  --gain 8 --transition-times '1231.7, 1243.9, 1247.5, 1258.9, 1262.9, 1273.9, 1277.6, 1289.6'

echo "=== ALL DONE ==="
