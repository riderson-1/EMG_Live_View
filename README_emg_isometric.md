# EMG Isometric Contraction Analysis (`emg_isometric.py`)

Reads an isometric EMG recording (CSV from the Sokosti wearable), applies filtering, computes a linear envelope, detects strong and weak contractions, and produces a single figure with overlaid contraction envelopes.

## What the script does (pipeline)

1. **Load CSV** — reads the sample counter column for gap detection and all `ch1…ch16` EMG channels.
2. **Scale to voltage** — converts raw ADS1299 ADC codes to volts using `Vref = 4.5 V` and the PGA gain (`--gain`), then to the chosen unit (`--unit`).
3. **Filter** — per contiguous valid segment:
   - 4th-order Butterworth bandpass 20–400 Hz
   - 2nd-order Butterworth notch 49–51 Hz (50 Hz mains)
4. **Linear envelope** — full-wave rectification (`abs`) followed by a 4th-order Butterworth low-pass (default 6 Hz cutoff).
5. **Contraction detection** — threshold-based, with two independent thresholds relative to the *global* envelope maximum:
   - `--thresh-strong` (default 30 %) — runs above this with duration ≥ `--strong-min` (default 2 s) → **strong**
   - `--thresh-weak` (default 5 %) — runs above this with duration ≥ `--weak-min` (default 10 s) → **weak**
   - Runs that overlap a strong contraction are excluded from the weak set.
   - Runs separated by gaps shorter than `--merge-gap` (default 0.5 s) are merged.
6. **Plot** — one figure, three panels:
   - **Top (full width):** filtered EMG magnitude (gray) + linear envelope (red), y-axis in absolute voltage (µV / mV / V).
   - **Bottom-left:** strong contraction envelopes overlaid.
   - **Bottom-right:** weak contraction envelopes overlaid.
   - Each contraction is plotted on its **real time axis** (seconds relative to its own onset, 0 = threshold crossing), with 20 % before/after context. Contractions are aligned at onset but **not** time-rescaled — longer contractions simply last longer.
   - Individual traces are light-gray dashed lines; the **mean** (black) and **90 % CI** (dark gray band) are computed only where **all** contractions are active (they stop at the earliest contraction end).
   - Y-axis of the bottom panels: "% of global max" (same global max for both, so strong vs. weak magnitude difference is directly visible).

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
| `--gain` | float | `1.0` | ADS1299 PGA gain used during recording (Vref = 4.5 V). Converts ADC codes to voltage. |
| `--unit` | str | auto | Y-axis unit for the magnitude plot: `uV`, `mV` or `V`. Auto: µV if max < 1 mV, else mV. |
| `--contractions` | str | `None` | Manual contraction windows, e.g. `"10-15,20-50,60-65"` (seconds). Overrides auto-detection. |
| `--labels` | str | `None` | Comma-separated `s`/`w` labels for manual contractions (only with `--contractions`). |
| `--save` | str | `None` | Prefix for saved PNG files (`<prefix>_magnitude.png`, `<prefix>_overlays.png`). |

## Example commands

Basic run on the Iso_Dor file (4 strong + 4 weak contractions), with gain 24 and µV output:

```
(.venv) karl@karl-HP-ZBook-Power-G7-Mobile-Workstation:~/repos/Sokosti_tools$ 

/home/karl/repos/Sokosti_tools/.venv/bin/python /home/karl/repos/Sokosti_tools/emg_isometric.py /home/karl/Documents/Master_Thesis/Testing/application_testing/EMG_Testing_02092026/USB_1_Karl_Iso_Dor_2026-09-02/USB_1_Karl_Iso_Dor_2026-09-02_PC.csv --thresh-strong 30 --unit mv --gain 1
```

```bash
source .venv/bin/activate
python emg_isometric.py \
  "Testing/application_testing/EMG_Testing_02092026/USB_1_Karl_Iso_Dor_2026-09-02/USB_1_Karl_Iso_Dor_2026-09-02_PC.csv" \
  --channels 1 \
  --thresh-strong 30 \
  --gain 24 \
  --save /tmp/iso_output
```

Force a specific unit (e.g. millivolts):

```bash
python emg_isometric.py <csv> --channels 1 --thresh-strong 30 --gain 24 --unit mv --save /tmp/iso_output
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
  - Top: filtered EMG + envelope, y-axis in absolute voltage (µV / mV / V), x-axis in seconds.
  - Bottom-left: strong contraction overlays (n = count).
  - Bottom-right: weak contraction overlays (n = count).
  - Bottom panels: x-axis = time from contraction onset (s); y-axis = "% of global max". Individual traces are light-gray dashed; the black mean line and dark-gray 90 % CI band are shown only where all contractions are active, with the legend label `Mean (n=…)`.

## Notes

- The script uses the same gap-aware filtering as `emg_processing.py` (per-segment `filtfilt` to avoid edge artifacts).
- Weak contractions are often missed at the default `--thresh-strong 20` because the weak envelope briefly crosses the strong threshold at onset. Use `--thresh-strong 30` for cleaner separation.
- The `--gain` value must match the PGA gain used during recording for the voltage values to be physically correct. With the default gain 1, the y-axis still shows a voltage unit but the absolute values are only meaningful once the correct gain is supplied.
- For multi-channel analysis, repeat the command with `--channels 1 2 3` etc. — each channel produces its own figure.
