#!/bin/bash
# Rerun the 18 Walk analyses (6 sessions x 3 walk types) for both the PC and
# SD variants (36 runs) with the exact settings from the existing
# *_imu_analysis.log files (per-trial EMG channel, env-cutoff 15, gains,
# heel-strike times, default percent axis), regenerating logs + PNGs + the
# per-trial pipeline CSVs (<stem>_signals.csv and <stem>_emg.csv via
# --save-trial-csv).
#
# Heel-strike times come from each log's '# Command:' line (complete and
# consistent for all 18 trials; the heel_strikes.txt files are incomplete
# and partly stale).
set -u
TOOLS_DIR=$(cd "$(dirname "$0")" && pwd)
SCRIPT=$TOOLS_DIR/gait_analysis.py
VENV=$TOOLS_DIR/../.venv/bin/python
BASE=/home/karl/Documents/Master_Thesis/Testing/application_testing

run() {
  echo "=== RUN: $1"
  "$VENV" "$SCRIPT" "$2" "${@:3}" --time-axis percent \
      --no-plot --save-results --save-trial-csv || echo "!!! FAILED: $1"
}

D0209=$BASE/EMG_Testing_02092026
D0809=$BASE/EMG_Testing_08092026
D1409=$BASE/EMG_Testing_14092026

# ---------------------------------------------------------------------------
# 02092026 USB_1_Karl (gain 1, ch16)
# ---------------------------------------------------------------------------
for W in Fast Slow Incline; do
  for S in PC SD; do
    D=$D0209/USB_1_Karl_${W}_Walk_2026-09-02
    C=$D/USB_1_Karl_${W}_Walk_2026-09-02_${S}.csv
    case $W in
      Fast)    H='44.9, 46, 47.1, 48.1, 49.2, 50.2, 51.2, 52.3, 53.3, 54.3, 55.3';;
      Slow)    H='40.8, 42, 43.2, 44.5, 45.7, 46.8, 48, 49.1, 50.3, 51.4, 52.6';;
      Incline) H='11.1, 12.4, 13.7, 15, 16.3, 17.5, 18.8, 20, 21.2, 22.4, 23.6';;
    esac
    run USB_1_Karl_${W}_${S} "$C" --channel 16 --env-cutoff 15 --gain 1 \
        --heel-strikes "$H"
  done
done

# ---------------------------------------------------------------------------
# 02092026 USB_1_Max (gain 1, ch2)
# ---------------------------------------------------------------------------
for W in Fast Slow Incline; do
  for S in PC SD; do
    D=$D0209/USB_1_Max_${W}_Walk_2026-09-02
    C=$D/USB_1_Max_${W}_Walk_2026-09-02_${S}.csv
    case $W in
      Fast)    H='38.8, 39.8, 40.8, 41.8, 42.8, 43.9, 44.9, 45.9, 46.9, 47.9, 49';;
      Slow)    H='43.5, 44.7, 45.8, 46.9, 48, 49.2, 50.3, 51.5, 52.6, 53.7, 54.9';;
      Incline) H='59.1, 60.3, 61.5, 62.6, 63.8, 64.9, 66.1, 67.3, 68.5, 69.6, 70.8';;
    esac
    run USB_1_Max_${W}_${S} "$C" --channel 2 --env-cutoff 15 --gain 1 \
        --heel-strikes "$H"
  done
done

# ---------------------------------------------------------------------------
# 08092026 BLE_1_Karl (gain 1, ch16)
# ---------------------------------------------------------------------------
for W in Fast Slow Incline; do
  for S in PC SD; do
    D=$D0809/BLE_1_Karl_${W}_Walk_2026-09-08
    C=$D/BLE_1_Karl_${W}_Walk_2026-09-08_${S}.csv
    case $W in
      Fast)    H='1120.5, 1121.5, 1122.6, 1123.6, 1124.6, 1125.6, 1126.6, 1127.6, 1128.6, 1129.6, 1130.6';;
      Slow)    H='864.6, 865.7, 866.9, 868.1, 869.2, 870.4, 871.6, 872.8, 874, 875.2, 876.4';;
      Incline) H='1218.9, 1220.2, 1221.4, 1222.6, 1223.8, 1225, 1226.2, 1227.3, 1228.5, 1229.7, 1230.9';;
    esac
    run BLE_1_Karl_${W}_${S} "$C" --channel 16 --env-cutoff 15 --gain 1 \
        --heel-strikes "$H"
  done
done

# ---------------------------------------------------------------------------
# 14092026 BLE_1_Max (gain 1, ch3)
# ---------------------------------------------------------------------------
for W in Fast Slow Incline; do
  for S in PC SD; do
    D=$D1409/BLE_1_Max_${W}_Walk_2026-09-14
    C=$D/BLE_1_Max_${W}_Walk_2026-09-14_${S}.csv
    case $W in
      Fast)    H='858.1, 859.2, 860.2, 861.2, 862.2, 863.3, 864.3, 865.3, 866.3, 867.3, 868.4';;
      Slow)    H='780.7, 781.8, 783, 784.2, 785.3, 786.4, 787.5, 788.7, 789.8, 790.9, 792';;
      Incline) H='688.5, 689.7, 690.9, 692, 693.2, 694.4, 695.6, 696.8, 697.9, 699.1, 700.2';;
    esac
    run BLE_1_Max_${W}_${S} "$C" --channel 3 --env-cutoff 15 --gain 1 \
        --heel-strikes "$H"
  done
done

# ---------------------------------------------------------------------------
# 14092026 BLE_8_Karl (gain 8 default, ch16)
# ---------------------------------------------------------------------------
for W in Fast Slow Incline; do
  for S in PC SD; do
    D=$D1409/BLE_8_Karl_${W}_Walk_2026-09-14
    C=$D/BLE_8_Karl_${W}_Walk_2026-09-14_${S}.csv
    case $W in
      Fast)    H='945, 946, 947, 948.1, 949.1, 950.2, 951.2, 952.3, 953.3, 954.4, 955.4';;
      Slow)    H='1026.1, 1027.3, 1028.5, 1029.7, 1030.9, 1032, 1033.2, 1034.4, 1035.6, 1036.7, 1037.9';;
      Incline) H='1121, 1122.2, 1123.4, 1124.7, 1125.9, 1127.1, 1128.3, 1129.6, 1130.8, 1132, 1133.2';;
    esac
    run BLE_8_Karl_${W}_${S} "$C" --channel 16 --env-cutoff 15 \
        --heel-strikes "$H"
  done
done

# ---------------------------------------------------------------------------
# 14092026 BLE_8_Max (gain 8 default, ch3)
# ---------------------------------------------------------------------------
for W in Fast Slow Incline; do
  for S in PC SD; do
    D=$D1409/BLE_8_Max_${W}_Walk_2026-09-14
    C=$D/BLE_8_Max_${W}_Walk_2026-09-14_${S}.csv
    case $W in
      Fast)    H='343.7, 344.7, 345.7, 346.8, 347.8, 348.8, 349.8, 350.8, 351.9, 352.9, 353.9';;
      Slow)    H='276.2, 277.4, 278.5, 279.7, 280.8, 281.9, 283.1, 284.2, 285.3, 286.4, 287.6';;
      Incline) H='434.7, 435.9, 437.1, 438.2, 439.4, 440.6, 441.7, 442.9, 444.1, 445.3, 446.4';;
    esac
    run BLE_8_Max_${W}_${S} "$C" --channel 3 --env-cutoff 15 \
        --heel-strikes "$H"
  done
done

echo "=== ALL DONE ==="