#!/usr/bin/env bash
# Run emg_gap_stat.py on every raw trial capture (*_PC.csv / *_SD.csv) under
# application_testing/ and compile the combined gap-statistics CSV.
#
# Derived files are skipped by the *_PC.csv / *_SD.csv match:
#   *_emg.csv, *_signals.csv (pipeline outputs), *_SD_cropped.csv,
#   *_SD_shortened.csv (edited captures).
#
# Usage: ./run_gap_stats.sh [base_dir]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
PY="$REPO/.venv/bin/python"
GAP_STAT="$HERE/emg_gap_stat.py"
COLLECT="$HERE/collect_gap_stats.py"
OUT="$HERE/gap_statistics_combined.csv"
BASE="${1:-$HOME/Documents/Master_Thesis/Testing/application_testing}"
FS=980

if [ ! -d "$BASE" ]; then
  echo "Base directory not found: $BASE" >&2
  exit 1
fi
if [ ! -x "$PY" ]; then
  echo "Python not found: $PY" >&2
  exit 1
fi

mapfile -t CSVS < <(find "$BASE" -type f \( -name '*_PC.csv' -o -name '*_SD.csv' \) | sort)
if [ "${#CSVS[@]}" -eq 0 ]; then
  echo "No trial captures (*_PC.csv / *_SD.csv) under $BASE" >&2
  exit 1
fi

echo "Processing ${#CSVS[@]} trial CSVs (fs=$FS) under $BASE"
failures=()
for csv in "${CSVS[@]}"; do
  printf '  %s\n' "${csv#"$BASE"/}"
  if ! "$PY" "$GAP_STAT" "$csv" --fs "$FS" >/dev/null; then
    failures+=("$csv")
  fi
done

if [ "${#failures[@]}" -gt 0 ]; then
  printf 'FAILED (%d):\n' "${#failures[@]}" >&2
  printf '  %s\n' "${failures[@]}" >&2
fi

"$PY" "$COLLECT" --base "$BASE" --out "$OUT" --fs "$FS"
"$PY" "$HERE/plot_gap_violin.py" --csv "$OUT" --out-dir "$HERE/outputs"