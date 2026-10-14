#!/bin/bash
# Rerun the 18 Walk analyses (6 sessions x 3 walk types) for both the PC and
# SD variants (36 runs) with the exact settings from the existing
# *_imu_analysis.log files (per-trial EMG channel, env-cutoff 15, gains,
# heel-strike times, default percent axis), regenerating logs + PNGs + the
# per-trial pipeline CSVs (<stem>_signals.csv and <stem>_emg.csv via
# --save-trial-csv).
#
# Heel-strike times: the hardcoded times below come from each log's
# '# Command:' line (complete and consistent for all 18 trials). If a trial
# folder contains a heel_strikes.txt WRITTEN BY pick_heel_strikes.py
# (identified by 'pick_heel_strikes.py' in its first line), those picked
# times override the hardcoded ones for both the PC and SD runs of that
# trial. Picked times are already on the absolute 980 sps time base and
# need no scaling; the hardcoded log times were scaled by 1000/980 so they
# select the SAME samples as the old fs=1000 runs (same convention as
# rerun_iso_dor.sh and rerun_sit_stand.sh). The old hand-written
# heel_strikes.txt files (incomplete and partly stale) are NOT picked up:
# they lack the picker header.
#
# Sample rate is 980 sps (NOT 1000): --emg-fs 980 is appended by run().
set -u
TOOLS_DIR=$(cd "$(dirname "$0")" && pwd)
SCRIPT=$TOOLS_DIR/gait_analysis.py
# Python venv: prefer the one next to this checkout; when running from a
# git worktree (this branch checked out beside the main repo), fall back to
# the main workspace's venv - the first entry of `git worktree list`.
# Override with VENV=/path/to/python if neither location applies.
if [ -z "${VENV:-}" ]; then
  if [ -x "$TOOLS_DIR/../.venv/bin/python" ]; then
    VENV=$TOOLS_DIR/../.venv/bin/python
  else
    MAIN_WT=$(git -C "$TOOLS_DIR" worktree list --porcelain 2>/dev/null \
        | awk '/^worktree /{print $2; exit}')
    VENV=$MAIN_WT/.venv/bin/python
  fi
fi
BASE=/home/karl/Documents/Master_Thesis/Testing/application_testing

run() {
  echo "=== RUN: $1"
  "$VENV" "$SCRIPT" "$2" "${@:3}" --emg-fs 980 --time-axis percent \
      --no-plot --save-results --save-trial-csv || echo "!!! FAILED: $1"
}

# hs_override <trial-dir> <fallback-times>: echo the picked heel-strike
# times from <trial-dir>/heel_strikes.txt when that file was written by
# pick_heel_strikes.py, else echo the fallback unchanged.
hs_override() {
  local F=$1/heel_strikes.txt H
  if [ -f "$F" ] && grep -q "pick_heel_strikes.py" "$F"; then
    H=$(grep -v '^#' "$F" | sed '/^[[:space:]]*$/d' | paste -sd, -)
    if [ -n "$H" ]; then
      echo "    heel strikes: $F" >&2
      echo "$H"
      return
    fi
  fi
  echo "$2"
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
      Fast)    H='45.8163, 46.9388, 48.0612, 49.0816, 50.2041, 51.2245, 52.2449, 53.3673, 54.3878, 55.4082, 56.4286';;
      Slow)    H='41.6327, 42.8571, 44.0816, 45.4082, 46.6327, 47.7551, 48.9796, 50.102, 51.3265, 52.449, 53.6735';;
      Incline) H='11.3265, 12.6531, 13.9796, 15.3061, 16.6327, 17.8571, 19.1837, 20.4082, 21.6327, 22.8571, 24.0816';;
    esac
    H=$(hs_override "$D" "$H")
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
      Fast)    H='39.5918, 40.6122, 41.6327, 42.6531, 43.6735, 44.7959, 45.8163, 46.8367, 47.8571, 48.8776, 50';;
      Slow)    H='44.3878, 45.6122, 46.7347, 47.8571, 48.9796, 50.2041, 51.3265, 52.551, 53.6735, 54.7959, 56.0204';;
      Incline) H='60.3061, 61.5306, 62.7551, 63.8776, 65.102, 66.2245, 67.449, 68.6735, 69.898, 71.0204, 72.2449';;
    esac
    H=$(hs_override "$D" "$H")
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
      Fast)    H='1143.37, 1144.39, 1145.51, 1146.53, 1147.55, 1148.57, 1149.59, 1150.61, 1151.63, 1152.65, 1153.67';;
      Slow)    H='882.245, 883.367, 884.592, 885.816, 886.939, 888.163, 889.388, 890.612, 891.837, 893.061, 894.286';;
      Incline) H='1243.78, 1245.1, 1246.33, 1247.55, 1248.78, 1250, 1251.22, 1252.35, 1253.57, 1254.8, 1256.02';;
    esac
    H=$(hs_override "$D" "$H")
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
      Fast)    H='875.612, 876.735, 877.755, 878.775, 879.796, 880.918, 881.939, 882.959, 883.98, 885, 886.122';;
      Slow)    H='796.633, 797.755, 798.98, 800.204, 801.327, 802.449, 803.571, 804.796, 805.918, 807.041, 808.163';;
      Incline) H='702.551, 703.775, 705, 706.122, 707.347, 708.571, 709.796, 711.02, 712.143, 713.367, 714.49';;
    esac
    H=$(hs_override "$D" "$H")
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
      Fast)    H='964.286, 965.306, 966.327, 967.449, 968.469, 969.592, 970.612, 971.735, 972.755, 973.878, 974.898';;
      Slow)    H='1047.04, 1048.27, 1049.49, 1050.71, 1051.94, 1053.06, 1054.29, 1055.51, 1056.73, 1057.86, 1059.08';;
      Incline) H='1143.88, 1145.1, 1146.33, 1147.65, 1148.88, 1150.1, 1151.33, 1152.65, 1153.88, 1155.1, 1156.33';;
    esac
    H=$(hs_override "$D" "$H")
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
      Fast)    H='350.714, 351.735, 352.755, 353.878, 354.898, 355.918, 356.939, 357.959, 359.082, 360.102, 361.122';;
      Slow)    H='281.837, 283.061, 284.184, 285.408, 286.531, 287.653, 288.878, 290, 291.122, 292.245, 293.469';;
      Incline) H='443.571, 444.796, 446.02, 447.143, 448.367, 449.592, 450.714, 451.939, 453.163, 454.388, 455.51';;
    esac
    H=$(hs_override "$D" "$H")
    run BLE_8_Max_${W}_${S} "$C" --channel 3 --env-cutoff 15 \
        --heel-strikes "$H"
  done
done

echo "=== ALL DONE ==="