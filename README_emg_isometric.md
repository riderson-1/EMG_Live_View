# EMG Isometric Contraction Analysis (`emg_isometric.py`)

Reads an isometric EMG recording (CSV from the Sokosti wearable), applies filtering, computes a linear envelope, detects strong and weak contractions, and produces a single figure with overlaid time-normalized envelopes.

## What the script does (pipeline)

1. **Load CSV** — reads the sample counter column for gap detection and all `ch1…ch16` EMG channels.
2. **Filter** — per contiguous valid segment:
   - 4th-order Butterworth bandpass 20–400 Hz
   - 2nd-order Butterworth notch 49–51 Hz (50 Hz mains)
3. **Linear envelope** — full-wave rectification (`abs`) followed by a 4th-order Butterworth low-pass (default 6 Hz cutoff).
4. **Contraction detection** — threshold-based, with two independent thresholds relative to the *global* envelope maximum:
   - `--thresh-strong` (default 30 %) — runs above this with duration ≥ `--strong-min` (default 2 s) → **strong**
   - `--thresh-weak` (default 5 %) — runs above this with duration ≥ `--weak-min` (default 10 s) → **weak**
   - Runs that overlap a strong contraction are excluded from the weak set.
   - Runs separated by gaps shorter than `--merge-gap` (default 0.5 s) are merged.
5. **Plot** — one figure, three panels:
   - **Top (full width):** filtered EMG magnitude (gray) + linear envelope (red).
   - **Bottom-left:** strong contraction envelopes overlaid, each resampled to 0–100 % of its duration, amplitude as % of the global envelope max.
   - **Bottom-right:** weak contraction envelopes, same normalization.
   - Mean line (black, thick) per group; legend with contraction count.

## Input file format

CSV with a numeric sample-counter column as the first column (used for gap detection) and EMG channels named `ch1`, `ch2`, … `ch16`. Example:

```
sample,status1_ok,status2_ok,ch1,ch2,...,ch16,roll,pitch,yaw,...
229194,1,1,289125,283217,...,283187,-178.17,70.86,-124.00
```

If no CSV path is given the script looks for the most recent `sokosti_capture_*.csv` in `captures/`.

## Command-line arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `csv` | positional | — | Path to the CSV file. Omit to use the most recent capture in `captures/`. |
| `--channels` | int list | `1` | EMG channel(s) to analyze (1-based). |
| `--thresh-strong` | float | `30` | Strong-contraction threshold as % of the **global** envelope max. |
| `--thresh-weak` | float | `5` | Weak-contraction threshold as % of the **global** envelope max. |
| `--strong-min` | float | `2.0` | Minimum duration (s) for a contraction to be classified strong. |
| `--weak-min` | float | `10.0` | Minimum duration (s) for a contraction to be classified weak. |
| `--merge-gap` | float | `0.5` | Merge above-threshold runs separated by gaps shorter than this (s). |
| `--env-cutoff` | float | `6.0` | Low-pass cutoff (Hz) for the linear envelope. |
| `--contractions` | str | `None` | Manual contraction windows, e.g. `"10-15,20-50,60-65"` (seconds). Overrides auto-detection. |
| `--labels` | str | `None` | Comma-separated `s`/`w` labels for manual contractions (only with `--contractions`). |
| `--save` | str | `None` | Prefix for saved PNG files (`<prefix>_magnitude.png`, `<prefix>_overlays.png`). |

## Example commands

Basic run on the Iso_Dor file (4 strong + 4 weak contractions):

```bash
source .venv/bin/activate
python emg_isometric.py \
  "Testing/application_testing/EMG_Testing_02092026/USB_1_Karl_Iso_Dor_2026-09-02/USB_1_Karl_Iso_Dor_2026-09-02_PC.csv" \
  --channels 1 \
  --thresh-strong 30 \
  --save /tmp/iso_output
```

Adjust thresholds if the weak contractions are not detected (e.g. if the weak envelope is noisier):

```bash
python emg_isometric.py <csv> --channels 1 --thresh-strong 30 --thresh-weak 3 --save /tmp/iso_output
```

Manual override of contraction windows (useful when auto-detection misses or over-detects):

```bash
python emg_isometric.py <csv> --channels 1 \
  --contractions "5-11,21-27,37-42,52-57,77-97,117-147,166-196,216-246" \
  --labels "s,s,s,s,w,w,w,w" \
  --save /tmp/iso_manual
```

## Output

- **Console:** gap statistics, detected contraction windows (strong / weak), total count.
- **Figure (interactive or saved):**
  - Top: filtered EMG + envelope with threshold lines (removed in current version).
  - Bottom-left: strong contraction overlays (n = count).
  - Bottom-right: weak contraction overlays (n = count).
  - Y-axis: "% of global max" (same global max used for both subplots, so strong vs. weak magnitude difference is directly visible).

## Notes

- The script uses the same gap-aware filtering as `emg_processing.py` (per-segment `filtfilt` to avoid edge artifacts).
- Weak contractions are often missed at the default `--thresh-strong 20` because the weak envelope briefly crosses the strong threshold at onset. Use `--thresh-strong 30` for cleaner separation.
- For multi-channel analysis, repeat the command with `--channels 1 2 3` etc. — each channel produces its own figure.
