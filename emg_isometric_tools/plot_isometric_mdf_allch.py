#!/usr/bin/env python3
"""
Pooled all-channel MDF (median frequency) box plots for the PC isometric trials.

Three figures (one per connection/gain combination), matching the SNR plots
of plot_isometric_trials_allch.py:
  - USB gain 1   (Max + Karl, 2026-09-02)
  - BLE gain 1   (Max 2026-09-14, Karl 2026-09-08)
  - BLE gain 8   (Max + Karl, 2026-09-14)

Each figure is one horizontal box plot per subject:
  - y axis = subject (Max, Karl)
  - x axis = MDF in Hz
  - Box-and-whisker: whiskers span min..max, the box spans Q1..Q3, the line
    inside is the median and the diamond is the mean.
  - Values pooled over ALL 16 channels of the subject's PC trial (n = 16).

MDF is computed by emg_isometric.py once per channel over the whole
recording, so there is no strong/weak split and no per-contraction rows.

Input : emg_isometric_summary.ods (sheets 'mdf' and 'trials')
Output: mdf_allch_usb_gain1.(png|pdf), mdf_allch_ble_gain1.(png|pdf),
        mdf_allch_ble_gain8.(png|pdf), mdf_allch_plot_data.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

# ===== SETTINGS =====
BOX_HEIGHT = 0.32              # height of the MDF boxes
SUBJECTS = ["Max", "Karl"]
BOX_COLOR = "lightgreen"
MEAN_COLOR = "seagreen"

REQUIRED_COLS = ["trial_id", "subject", "source", "connection", "channel", "mdf_hz"]

# ===== ARGUMENTS =====
parser = argparse.ArgumentParser(
    description="Pooled all-channel MDF box plots for the PC isometric trials.")
parser.add_argument("ods", help="Path to the summary workbook "
                                "(e.g. emg_isometric_summary.ods)")
parser.add_argument("--sheet", default="mdf", help="Sheet with per-channel MDF "
                                                   "(default: mdf)")
parser.add_argument("--out-dir", default=".", help="Folder for the output files (default: .)")
parser.add_argument("--no-show", action="store_true", help="Save the figures without opening windows.")
args = parser.parse_args()


# ===== LOAD =====
def load(path, sheet):
    if not Path(path).exists():
        raise SystemExit(f"Workbook not found: {path}")
    df = pd.read_excel(path, sheet_name=sheet)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in sheet '{sheet}': {missing}")
    df = df.dropna(subset=["mdf_hz"])
    if df.empty:
        raise SystemExit("No MDF values in sheet '{}'.".format(sheet))
    # the PGA gain lives in the 'trials' sheet; join it onto the per-channel rows
    trials_sheet = pd.read_excel(path, sheet_name="trials")
    if "gain" not in trials_sheet.columns:
        raise SystemExit("No 'gain' column in sheet 'trials'. "
                         "Regenerate the workbook with parse_emg_logs.py.")
    df = df.merge(trials_sheet[["trial_id", "gain"]], on="trial_id", how="left")
    return df


# ===== ANALYSIS =====
def build_groups(df):
    """Group the PC trials into (connection, gain) figures with one trial per subject."""
    pc = df[df["source"] == "PC"]
    if pc.empty:
        raise SystemExit("No rows with source == 'PC'.")
    groups = []
    for (connection, gain), g in pc.groupby(["connection", "gain"], sort=False):
        trials = []
        for subject in SUBJECTS:
            tids = sorted(g.loc[g["subject"] == subject, "trial_id"].unique())
            if not tids:
                raise SystemExit(f"No PC trial for subject '{subject}' in "
                                 f"{connection} gain {gain}.")
            trials.append({"subject": subject, "trial_ids": tids,
                           "dates": sorted({str(t)[:10] for t in
                                            pc.loc[pc["trial_id"].isin(tids), "date"]})})
        groups.append({"connection": connection, "gain": int(gain), "trials": trials})
    # stable, readable order: USB first, then BLE gain 1, then BLE gain 8
    groups.sort(key=lambda d: (d["connection"] != "USB", d["connection"], d["gain"]))
    return groups


def subject_mdf(df, trial_ids):
    """Pooled MDF values (Hz) over all channels of the given trials."""
    vals = df.loc[df["trial_id"].isin(trial_ids), "mdf_hz"].to_numpy(float)
    if len(vals) == 0:
        raise SystemExit(f"No MDF values for trials: {', '.join(trial_ids)}")
    return vals


# ===== PLOT =====
def draw_box(ax, y, vals, facecolor, mean_color):
    """Horizontal box-and-whisker at y: whiskers = min..max, box = Q1..Q3
    (25/25/25/25), line = median. Mean shown as a diamond with its value."""
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    lo, hi, mean = vals.min(), vals.max(), vals.mean()
    # whiskers
    ax.hlines(y, lo, q1, color="black", linewidth=1.2, zorder=2)
    ax.hlines(y, q3, hi, color="black", linewidth=1.2, zorder=2)
    for x in (lo, hi):
        ax.vlines(x, y - BOX_HEIGHT / 4, y + BOX_HEIGHT / 4,
                  color="black", linewidth=1.2, zorder=2)
    # IQR box
    ax.add_patch(Rectangle((q1, y - BOX_HEIGHT / 2), q3 - q1, BOX_HEIGHT,
                           facecolor=facecolor, edgecolor="black",
                           alpha=0.8, linewidth=1.2, zorder=3))
    # median line
    ax.vlines(med, y - BOX_HEIGHT / 2, y + BOX_HEIGHT / 2,
              color="black", linewidth=2.0, zorder=4)
    # mean diamond with its value
    ax.scatter(mean, y, marker="D", s=70, facecolor=mean_color,
               edgecolor="black", zorder=5)
    ax.annotate(f"{mean:.1f}", (mean, y), textcoords="offset points",
                xytext=(0, BOX_HEIGHT / 2 * 72 + 6), ha="center", va="bottom",
                fontsize=13, color=mean_color, fontweight="bold")
    # min / q1 / median / q3 / max labels below the box
    for val, label in [(lo, f"{lo:.1f}"), (q1, f"{q1:.1f}"), (med, f"{med:.1f}"),
                       (q3, f"{q3:.1f}"), (hi, f"{hi:.1f}")]:
        ax.annotate(label, (val, y), textcoords="offset points",
                    xytext=(0, -(BOX_HEIGHT / 2 * 72 + 6)), ha="center", va="top",
                    fontsize=10, color="black")


def plot_figure(df, group, out_stem):
    """One figure: one horizontal MDF box per subject (Max, Karl)."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    stats = {t["subject"]: subject_mdf(df, t["trial_ids"]) for t in group["trials"]}
    centres = np.arange(1, len(group["trials"]) + 1)

    all_vals = np.concatenate(list(stats.values()))
    pad = 0.08 * (all_vals.max() - all_vals.min())
    xlim = (all_vals.min() - pad, all_vals.max() + pad)

    for yc, t in zip(centres, group["trials"]):
        draw_box(ax, yc, stats[t["subject"]], BOX_COLOR, MEAN_COLOR)

    ax.set_xlim(*xlim)
    # Max on top, Karl below (first subject at the top)
    ax.set_ylim(centres[-1] + 0.75, centres[0] - 0.75)
    ax.set_yticks(centres)
    ax.set_yticklabels(
        [f"{t['subject']}\n({', '.join(t['dates'])})" for t in group["trials"]],
        fontsize=14)
    ax.tick_params(axis="x", labelsize=13)
    ax.set_xlabel("MDF (Hz)", fontsize=15)
    ax.set_ylabel("Subject", fontsize=15)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    ax.set_title(f"All 16 channels pooled, n = "
                 f"{', '.join(str(len(stats[t['subject']])) for t in group['trials'])} "
                 f"MDF values per subject",
                 fontsize=14, pad=10)

    handles = [
        Patch(facecolor=BOX_COLOR, edgecolor="black", alpha=0.8,
              label="MDF (all channels): min / Q1 / median / Q3 / max"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=MEAN_COLOR,
               markeredgecolor="black", markersize=8, label="Mean MDF"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=12, frameon=False)
    fig.suptitle(f"{group['connection']} recording, PGA gain {group['gain']} — "
                 f"median frequency (PC recordings)",
                 fontsize=17, y=0.995)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300)
    return fig


# ===== MAIN =====
def main():
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load(args.ods, args.sheet)
    groups = build_groups(df)

    all_rows = []
    figs = []
    for group in groups:
        stem = str(out_dir / f"mdf_allch_{group['connection'].lower()}_gain{group['gain']}")
        print(f"\n=== {group['connection']} gain {group['gain']} ===")
        for t in group["trials"]:
            print(f"  {t['subject']}: trials {', '.join(t['trial_ids'])}")
        figs.append(plot_figure(df, group, stem))

        # ---- CSV rows: exactly the numbers plotted ----
        for t in group["trials"]:
            vals = subject_mdf(df, t["trial_ids"])
            all_rows.append({
                "connection": group["connection"], "gain": group["gain"],
                "subject": t["subject"], "trial_ids": ";".join(t["trial_ids"]),
                "n": len(vals),
                "mdf_min_Hz": vals.min(),
                "mdf_q1_Hz": np.percentile(vals, 25),
                "mdf_median_Hz": np.median(vals),
                "mdf_q3_Hz": np.percentile(vals, 75),
                "mdf_mean_Hz": vals.mean(),
                "mdf_max_Hz": vals.max(),
            })

    pd.DataFrame(all_rows).to_csv(out_dir / "mdf_allch_plot_data.csv", index=False)

    print("\nSaved:")
    for f in sorted(out_dir.glob("mdf_allch_*")):
        print(f"  {f.name}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
