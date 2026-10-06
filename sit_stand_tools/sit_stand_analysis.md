# Sit-to-Stand IMU/EMG Analysis (`gait_analysis.py`)

PCA-based sagittal-plane isolation for Sokosti IMU captures during **sit-to-stand-to-sit** movements, with EMG overlay and cycle-locked averaging.

## What the script does

The IMU sits on the lower leg. The script processes the capture in these steps:

1. **Load** — reads `sample, roll, pitch, yaw, accel_x, accel_y, accel_z` from the capture CSV. Rows before the first IMU packet are dropped; isolated NaNs are interpolated.
2. **Resample** — the 50 Hz IMU streams are resampled onto a uniform time grid.
3. **Filter**
   - 4th-order zero-phase Butterworth **low-pass (10 Hz)** on the angles.
   - 4th-order zero-phase Butterworth **high-pass (0.2 Hz)** on each accel axis → dynamic acceleration (gravity removed).
4. **PCA**
   - **Angular-rate PCA** on `[d(roll)/dt, d(pitch)/dt, d(yaw)/dt]`: PC1 is the dominant rotation axis. For sagittal-dominant sit-to-stand motion, PC1 should align with the pitch axis and explain >~0.8 of the variance.
   - **Dynamic-accel PCA**: PC1/PC2 should span the sagittal (forward+vertical) plane; a small PC3 fraction means the motion is mostly planar.
5. **SVM (Signal Vector Magnitude)** — orientation-independent motion intensity: `|omega|` from the angular rates and `|a_dyn|` from the dynamic acceleration. SVM says *how much* total motion there is; PCA says *which plane* it lives in.
6. **Sign fix** — the PCA sign is arbitrary, so it is flipped so the largest-magnitude peak is positive.
7. **Figures** (see below).

### Legacy accelerometer disclaimer

Captures recorded **before 2026-10-02** stored the accelerometer with a wrong conversion (raw/16 instead of raw/4096), so 1 g reads as ~256. The script auto-detects this (median |accel| >> 1 g) and rescales by 1/256. Override with `--accel-unit {auto,g,legacy}`.

## Figures

| Figure | Content |
|--------|---------|
| **PCA figure** (5 subplots) | Filtered angles; dynamic accel + accel PC1/PC2; angular rates + rate PC1; \|omega\| SVM; \|a_dyn\| SVM |
| **EMG/IMU figure** (3 subplots) | EMG envelope (µV), dynamic-accel PC1, angular-rate PC1 on one shared time axis |
| **Sit-to-stand-cycle overlays** (3 subplots) | Same three signals overlaid across cycles: individual traces (grey dashed) + mean (black) + 90% CI |

## Cycles

One **sit-to-stand cycle** = from the **start of standing up** to the **end of sitting down**.

You provide the transition times yourself with `--transition-times`:

- Times come in **pairs**: `(start of standing up, end of sitting down)` per cycle.
- `n` times → `n / 2` cycles (an even number of times is required).
- Times must be strictly increasing.
- Each cycle is normalized to **0–100 %** (default) or kept in **real seconds** from the start of standing up (`--time-axis time`).

Example: `--transition-times 10.5,12.3,14.1,16.0` gives 2 cycles: 10.5→12.3 s and 14.1→16.0 s.

## Usage

```bash
python gait_analysis.py CAPTURE.csv
```

### Common options

```bash
# Restrict the PCA fit window (exclude standing still / rest periods)
python gait_analysis.py CAPTURE.csv --fit-start 10 --fit-end 60

# Cycle overlays with your own transition times
python gait_analysis.py CAPTURE.csv --transition-times 10.5,12.3,14.1,16.0

# Real-time axis instead of percent
python gait_analysis.py CAPTURE.csv --transition-times 10.5,12.3,14.1,16.0 --time-axis time

# Different EMG channel, save everything to files
python gait_analysis.py CAPTURE.csv --channel 2 --save-results --no-plot
```

### All options

| Option | Default | Description |
|--------|---------|-------------|
| `input` | — | Capture CSV (columns: sample, roll, pitch, yaw, accel_x/y/z) |
| `--emg-fs` | 1000 | CSV row rate in Hz (time base) |
| `--imu-fs` | 50 | IMU update rate in Hz to resample to |
| `--lp-fc` | 10 | Low-pass cutoff for angles (Hz) |
| `--hp-fc` | 0.2 | High-pass cutoff for gravity removal (Hz) |
| `--order` | 4 | Butterworth filter order |
| `--accel-unit` | auto | `auto` / `g` / `legacy` accelerometer scaling |
| `--fit-start`, `--fit-end` | none | PCA fit window in seconds |
| `--save-csv` | none | Save processed signals to a CSV |
| `--save-plot` | none | Save the PCA figure to an image |
| `--no-plot` | off | Don't open an interactive plot window |
| `--channel` | 1 | 1-based EMG channel for the envelope |
| `--env-cutoff` | 6 | EMG envelope low-pass cutoff (Hz) |
| `--gain` | 8 | ADS1299 PGA gain (for µV conversion, Vref = 4.5 V) |
| `--transition-times` | none | Comma-separated transition times in seconds, in pairs (start of standing up, end of sitting down) per cycle |
| `--time-axis` | percent | `percent` (0–100 %) or `time` (seconds) for cycle overlays |
| `--save-results` | off | Save all figures + a log file (`<csv>_imu_analysis.log`) next to the CSV |

## Output files (with `--save-results`)

| File | Content |
|------|---------|
| `<csv>_sagittal_pca.png` | PCA figure |
| `<csv>_emg_imu.png` | EMG/IMU figure |
| `<csv>_sit_to_stand_cycles.png` | Cycle overlay figure (only if `--transition-times` given) |
| `<csv>_imu_analysis.log` | Full terminal output (markdown tables) |

The terminal output includes markdown tables: capture info, gravity estimate, EMG sample loss, EMG envelope filters, angular-rate PCA, dynamic-accel PCA, rate/accel correlations, SVM statistics, and sit-to-stand cycle durations.

## Interpreting the results

- **Angular-rate PC1** should align with the **pitch axis** for sagittal-dominant sit-to-stand (explained variance > ~0.8).
- **Accel PC3 fraction** small → motion is mostly planar (sagittal).
- **Cycle overlays**: consistent cycles produce tightly bundled individual traces; the mean + 90% CI shows the typical EMG/motion pattern across the sit-to-stand-to-sit movement.