# Sit-to-Stand Analysis Pipeline Documentation

This document describes the complete analysis pipeline for the sit-to-stand
(Sit_Stand) EMG/IMU trials: from the raw CSV recordings to the cycle-locked
overlay plots. It mirrors the Iso_Dor pipeline
(`emg_isometric_tools/iso_dor_analysis_pipeline.md`), but instead of parsing
the analysis logs, the plotter reads **per-trial CSVs** written by the
analysis itself.

---

## Overview

```
Raw CSV recordings          sit_stand_analysis.py         rerun_sit_stand.sh        plot_sit_stand_trials.py
(BLE/USB capture)     -->   per-trial analysis      -->   batch runner, 12    -->   cycle-locked overlay
                            + *_imu_analysis.log          trials (logged            plots + plot-data CSV
                            + PNG figures                 transition times)
                            + <stem>_signals.csv
                            + <stem>_emg.csv
```

| Stage | File | Location |
|-------|------|----------|
| 1. Analysis | `sit_stand_analysis.py` | `/home/karl/repos/Sokosti_tools/sit_stand_tools/` |
| 2. Batch runner | `rerun_sit_stand.sh` | `/home/karl/repos/Sokosti_tools/sit_stand_tools/` |
| 3. Plotting | `plot_sit_stand_trials.py` | `/home/karl/repos/Sokosti_tools/sit_stand_tools/` |

Data locations:

- Raw CSVs, per-trial logs, PNGs and per-trial CSVs:
  `/home/karl/Documents/Master_Thesis/Testing/application_testing/EMG_Testing_<date>/<trial>/`
- Plots + plot-data CSV: written into `sit_stand_tools/` (or `--out-dir`).

There are **12 Sit_Stand trials**: 6 recording sessions (2 subjects × 3 dates)
× 2 sources each (PC + SD variant), same structure as the Iso_Dor trials.

---

## Stage 1: Per-trial analysis — `sit_stand_analysis.py`

Run per trial (batched by `rerun_sit_stand.sh`, see Stage 2):

```
.venv/bin/python sit_stand_analysis.py <csv> --gain <g> \
    --transition-times '<t1,t2,...>' --channel 1 --time-axis percent \
    --no-plot --save-results --save-trial-csv
```

The processing chain (load → resample → filter → PCA → SVM → sign fix →
figures) is documented in `sit_stand_analysis.md`. The pipeline-relevant
addition is `--save-trial-csv`, which writes **two CSVs per trial** next to
the capture CSV:

### `<stem>_signals.csv` — 50 Hz IMU/PCA grid

One row per resampled IMU sample:

```
t_s, roll_deg, pitch_deg, yaw_deg, droll_dps, dpitch_dps, dyaw_dps,
rate_pc1_dps, acc_dyn_x_g, acc_dyn_y_g, acc_dyn_z_g, acc_pc1_g, acc_pc2_g,
acc_pc3_g, rate_svm_dps, acc_svm_g
```

`# key: value` header comments carry everything the plotter (and any audit)
needs:

- `command` — the exact runnable command line (quoted, like the Iso_Dor logs)
- capture info: `imu_fs_hz`, `emg_fs_hz`, filter cutoffs, `accel_unit`,
  `imu_rows`, `duration_s`, `imu_t0_s` (IMU time offset in the EMG
  sample-time base — the IMU time axis is zero-based at the first IMU row)
- PCA statistics: explained variance PC1–PC3 (rate + accel), PC1 axis
  components, dominant axis, PC1+PC2 plane span, rate↔accel correlations
- SVM statistics: `|omega|` and `|a_dyn|` mean/max
- cycle metadata: `transition_times_s`, `n_cycles`, duration mean/SD/min/max,
  and per-cycle `cycle_<k>_start_s` / `cycle_<k>_end_s` / `cycle_<k>_duration_s`

### `<stem>_emg.csv` — 1000 Hz EMG envelope

One row per EMG sample: `t_s, emg_envelope_uV`. Header comments carry the
command line, channel, gain, µV/code scale and the filter settings
(bandpass 20–400 Hz, notch 48–52 Hz, envelope low-pass).

### Decisions

- **Two CSVs instead of one**: the IMU/PCA signals live on the 50 Hz
  resampled grid, the EMG envelope on the 1000 Hz EMG grid; merging them
  would NaN-fill 95 % of the IMU columns or downsample the EMG.
- **Statistics as header comments**: a CSV is a single table, so the PCA/cycle
  metadata go into `# key: value` comment lines at the top — auditable,
  machine-readable, and ignored by `pandas.read_csv(..., comment="#")`.
- **Transition times come from the existing logs**: all 12 trials were
  already analyzed with manual transition times; the batch runner reuses
  them verbatim, so no new manual input is needed.
- **`imu_t0_s`**: the IMU time axis is zero-based at the first IMU row while
  the EMG time axis is absolute (sample/fs). The offset is stored so the
  plotter can align both onto one timeline.

---

## Stage 2: Batch runner — `rerun_sit_stand.sh`

12 commands (6 sessions × PC/SD), one per trial. Each command reproduces the
settings of the existing `*_imu_analysis.log` exactly: the per-session gain
(1 or 8) and the manual transition times, default ch1 (`--channel 1`),
percent time axis, `--no-plot --save-results --save-trial-csv`.

Rerunning it regenerates all logs, PNGs and the per-trial CSVs with the
current script version:

```bash
bash rerun_sit_stand.sh
```

---

## Stage 3: Plotting — `plot_sit_stand_trials.py`

```
.venv/bin/python plot_sit_stand_trials.py            # all trials
.venv/bin/python plot_sit_stand_trials.py --source PC   # PC recordings only
```

Walks `application_testing/` for `*_signals.csv` + `*_emg.csv` pairs inside
any `*Sit_Stand*` trial folder and reads **only the CSVs** (no log parsing).

### Method

1. **Cycle extraction**: the transition times from the `_signals.csv` header
   are paired into cycles (start of standing up, end of sitting down). Each
   signal is interpolated onto a 0–100 % cycle grid (201 points); cycles with
   fewer than 10 valid samples are skipped.
2. **Signals plotted** (one subplot each): EMG ch1 envelope (µV), dynamic
   accel PC1 (g), angular-rate PC1 (deg/s) — the same three signals as the
   per-trial `sit_to_stand_cycles.png`.
3. **Per-trial figure** (`<trial_id>_cycle_overlays.png`): individual cycle
   traces (grey dashed) + mean (black) + 90 % CI, like the Iso_Dor
   contraction overlays.
4. **Cross-trial figure** (`sit_stand_cycle_means.png`): every trial's mean
   curve on one subplot per signal, one colour per trial.
5. **`sit_stand_plot_data.csv`**: exactly the plotted numbers — one row per
   trial × signal × cycle-% point: `trial_id, signal, cycle_pct, mean,
   ci_low, ci_high, n_cycles`.

### Decisions

- **CSVs, not logs**: unlike the Iso_Dor pipeline (which parses markdown
  tables out of the logs), the full signal traces are needed here, so the
  analysis writes them directly; the header comments replace log parsing.
- **0–100 % normalization** (default): sit-to-stand cycle durations vary
  between trials, so percent normalization makes the cross-trial mean curves
  comparable. (`--time-axis time` support in the plotter follows the
  per-trial script if ever needed.)
- **90 % CI** = 1.645 · SD / √n, same convention as the per-trial overlays.

---

## Reproducing the whole pipeline

```bash
cd /home/karl/repos/Sokosti_tools

# 1. per-trial analyses (regenerates logs + PNGs + per-trial CSVs)
bash sit_stand_tools/rerun_sit_stand.sh

# 2. plots (figures + sit_stand_plot_data.csv into sit_stand_tools/)
.venv/bin/python sit_stand_tools/plot_sit_stand_trials.py
```

Python environment: the workspace venv (`.venv/bin/python`, Python 3.12) has
numpy/scipy/pandas/matplotlib installed.

---

## Known caveats

- **BLE gap rates vary widely** between trials; the EMG envelope is computed
  gap-aware per contiguous valid segment, and the gap statistics are in each
  log's EMG sample-loss table.
- **Manual transition times** are subjective; they are recorded in each
  log's `# Command:` line and in each `_signals.csv` header
  (`transition_times_s`), so the exact cycles can always be audited.
- **PCA sign is arbitrary**; it is fixed per trial so the largest-magnitude
  peak is positive. Cross-trial mean curves of PC1 can therefore have
  inverted polarity between subjects/sessions — check the per-trial figures
  before over-interpreting the cross-trial plot.
  **DISCLAIMER (BLE_8_Karl_Sit_Stand_2026-09-14):** the PC and SD variants
  of this trial look like mirror images of each other (rate-PC1 correlation
  −0.96). This is NOT a data error: the two files are the same session
  (raw EMG identical, envelope correlation 0.9996), but the SD capture is
  longer (91.2 s vs 79.6 s), so the PCA fit windows differ and the
  largest-peak sign heuristic resolved the arbitrary PCA sign in opposite
  directions (PC dominant axis roll(x) vs SD yaw(z) — the same roll/yaw
  sagittal axis, opposite sign). The EMG envelope subplot, which has no sign
  ambiguity, is unaffected and agrees between the two variants.
- **SD variants** are analyzed and plotted by default; pass `--source PC` to
  restrict the plots to the PC recordings (as in the Iso_Dor pipeline).
