#!/usr/bin/env python3
"""
Pooled all-channel SNR plots for the PC isometric trials.

Three figures (one per connection/gain combination):
  - USB gain 1   (Max + Karl, 2026-09-02)
  - BLE gain 1   (Max 2026-09-14, Karl 2026-09-08)
  - BLE gain 8   (Max + Karl, 2026-09-14)

Each figure has two subplots: top = STRONG contractions, bottom = WEAK
contractions. Every subplot shows the two subjects (Max, Karl); for each
subject the signal box, the noise box and the SNR dot share one vertical
line (their y ranges do not overlap on the log axis):

  - Signal box  = ALL 16 channels x ALL contractions of that type pooled
                  (not just the best channel). Box-and-whisker: whiskers
                  span min..max, the box spans the lower..upper quartile
                  (25/25/25/25), the line inside is the median and the
                  diamond is the mean.
  - Noise box   = ALL 16 channels x weak rests 2..4 pooled (the first weak
                  rest is excluded because it follows the maximal strong
                  block). Identical in both subplots, same box-and-whisker
                  style.
  - SNR dot     = mean over the 16 per-channel SNR dB values, where each
                  channel's SNR = 20*log10(mean signal of that channel /
                  noise floor of that channel). One dot per subject per
                  subplot, on the right axis, on the same vertical line.

This is the "overall picture" variant of plot_isometric_trials.py (which
shows only the best channel per trial); the old script is kept unchanged.

Input : emg_isometric_summary.ods (sheets 'snr' and 'trials', values in uV)
Output: snr_allch_usb_gain1.(png|pdf), snr_allch_ble_gain1.(png|pdf),
        snr_allch_ble_gain8.(png|pdf), snr_allch_plot_data.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter

# ===== SETTINGS =====
NOISE_SKIP_FIRST_WEAK = 1     # leading weak rests excluded from the noise floor
BOX_WIDTH = 0.32              # width of the signal / noise boxes
GROUP_GAP = 1.0               # x distance between the Max and Karl groups
SUBJECTS = ["Max", "Karl"]
SIGNAL_COLOR = "lightblue"
NOISE_COLOR = "wheat"

REQUIRED_COLS = ["trial_id", "subject", "date", "connection", "source", "channel",
                 "contraction_type", "window_start_s", "signal_uV", "noise_uV"]

# ===== ARGUMENTS =====
parser = argparse.ArgumentParser(
    description="Pooled all-channel strong/weak SNR plots for the PC isometric trials.")
parser.add_argument("ods", help="Path to the summary workbook "
                                "(e.g. emg_isometric_summary.ods)")
parser.add_argument("--sheet", default="snr", help="Sheet with per-contraction data (default: snr)")
parser.add_argument("--out-dir", default=".", help="Folder for the output files (default: .)")
parser.add_argument("--noise-stat", choices=["mean", "median"], default="mean",
                    help="Statistic over weak rests 2-4 used as the per-channel noise floor "
                         "(default: mean).")
parser.add_argument("--linear", action="store_true",
                    help="Linear uV axis instead of log.")
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
    # the PGA gain lives in the 'trials' sheet; join it onto the per-contraction rows
    trials_sheet = pd.read_excel(path, sheet_name="trials")
    if "gain" not in trials_sheet.columns:
        raise SystemExit("No 'gain' column in sheet 'trials'. "
                         "Regenerate the workbook with parse_emg_logs.py.")
    df = df.merge(trials_sheet[["trial_id", "gain"]], on="trial_id", how="left")
    bad = df[~df["contraction_type"].isin(["strong", "weak"])]
    if len(bad):
        raise SystemExit("contraction_type is missing/invalid for trials: "
                         + ", ".join(sorted(bad["trial_id"].unique())))
    return df


# ===== ANALYSIS =====
def channel_table(df):
    """Per (trial, channel): noise floor (weak rests 2-4) and per-type mean signal."""
    weak = df[df["contraction_type"] == "weak"].copy()
    weak["weak_rank"] = (weak.groupby(["trial_id", "channel"])["window_start_s"]
                         .rank(method="first").astype(int))
    keys = ["trial_id", "channel"]

    used = weak[weak["weak_rank"] > NOISE_SKIP_FIRST_WEAK]
    noise = used.groupby(keys)["noise_uV"].agg(noise_uV=args.noise_stat, noise_sd="std",
                                               noise_n="count")

    sig = (df.groupby(keys + ["contraction_type"])["signal_uV"].mean()
           .unstack("contraction_type")
           .rename(columns={"strong": "sig_strong_mean", "weak": "sig_weak_mean"}))
    ch = sig.join(noise)
    ch["snr_strong_db"] = 20 * np.log10(ch["sig_strong_mean"] / ch["noise_uV"])
    ch["snr_weak_db"] = 20 * np.log10(ch["sig_weak_mean"] / ch["noise_uV"])
    return ch


def subject_stats(df, ch, trial_ids, ctype):
    """Pooled signal/noise values and pooled SNR for one subject within one figure."""
    sub = df[df["trial_id"].isin(trial_ids)]
    sig_vals = sub.loc[sub["contraction_type"] == ctype, "signal_uV"].to_numpy(float)

    # noise box: every channel's rests 2-4, pooled (same for strong and weak subplot)
    weak = sub[sub["contraction_type"] == "weak"].copy()
    weak["weak_rank"] = (weak.groupby(["trial_id", "channel"])["window_start_s"]
                         .rank(method="first").astype(int))
    noise_vals = weak.loc[weak["weak_rank"] > NOISE_SKIP_FIRST_WEAK, "noise_uV"].to_numpy(float)

    # pooled SNR = mean of the per-channel SNR dB values
    ch_rows = ch.loc[ch.index.get_level_values("trial_id").isin(trial_ids)]
    snr_db = float(ch_rows[f"snr_{ctype}_db"].mean())

    return {"signal": sig_vals, "noise": noise_vals, "snr_db": snr_db,
            "n_channels": ch_rows.index.get_level_values("channel").nunique()}


def build_groups(df, ch):
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


# ===== PLOT =====
def draw_box(ax, x, vals, facecolor, mean_color, fmt):
    """Box-and-whisker at x: whiskers = min..max, box = Q1..Q3 (25/25/25/25),
    line = median. Median value labelled next to the box, mean shown as a
    diamond with its value."""
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    lo, hi, mean = vals.min(), vals.max(), vals.mean()
    # whiskers
    ax.vlines(x, lo, q1, color="black", linewidth=1.2, zorder=2)
    ax.vlines(x, q3, hi, color="black", linewidth=1.2, zorder=2)
    for y in (lo, hi):
        ax.hlines(y, x - BOX_WIDTH / 4, x + BOX_WIDTH / 4,
                  color="black", linewidth=1.2, zorder=2)
    # IQR box
    ax.add_patch(Rectangle((x - BOX_WIDTH / 2, q1), BOX_WIDTH, q3 - q1,
                           facecolor=facecolor, edgecolor="black",
                           alpha=0.8, linewidth=1.2, zorder=3))
    # median line
    ax.hlines(med, x - BOX_WIDTH / 2, x + BOX_WIDTH / 2,
              color="black", linewidth=2.0, zorder=4)
    # # median value label next to the box
    # ax.annotate(fmt.format(med), (x, med), textcoords="offset points",
    #             xytext=(BOX_WIDTH / 2 * 72 + 35, -1), ha="left", va="center",
    #             fontsize=13, color=facecolor, fontweight="bold")
    # mean diamond with its value
    ax.scatter(x, mean, marker="D", s=70, facecolor=mean_color,
               edgecolor="black", zorder=5)
    ax.annotate(fmt.format(mean), (x, mean), textcoords="offset points",
                xytext=(-(BOX_WIDTH / 2 * 72 + 35), -1), ha="right", va="center",
                fontsize=13, color=mean_color, fontweight="bold")


def plot_figure(df, ch, group, log_scale, out_stem):
    """One figure: top = strong, bottom = weak; two subjects per subplot."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 11), sharex=True)

    # pooled stats per subject per contraction type
    stats = {}
    for ctype in ("strong", "weak"):
        for t in group["trials"]:
            stats[(ctype, t["subject"])] = subject_stats(df, ch, t["trial_ids"], ctype)

    # shared y-limits across both subplots
    all_vals = np.concatenate([stats[k]["signal"] for k in stats] +
                              [stats[k]["noise"] for k in stats])
    all_snrs = [stats[k]["snr_db"] for k in stats]
    if log_scale:
        ylim = (all_vals.min() * 0.5, all_vals.max() * 1.5)
    else:
        ylim = (0, all_vals.max() * 1.1)
    snr_top = np.ceil(max(all_snrs) * 1.15 / 10) * 10

    # Map the SNR (dB) axis into the empty band between the noise boxes and the
    # signal boxes so the SNR dots never overlap the boxes. The mapping is
    # linear in dB (ticks stay honest); in log space the band is the gap
    # between the highest noise whisker and the lowest signal whisker.
    band_lo = max(stats[k]["noise"].max() for k in stats)   # top of noise boxes
    band_hi = min(stats[k]["signal"].min() for k in stats)  # bottom of signal boxes
    if log_scale:
        lo_l, hi_l = np.log10(band_lo), np.log10(band_hi)
        pad = 0.08 * (hi_l - lo_l) if hi_l > lo_l else 0.0
        band_lo, band_hi = 10 ** (lo_l + pad), 10 ** (hi_l - pad)
    else:
        pad = 0.08 * (band_hi - band_lo) if band_hi > band_lo else 0.0
        band_lo, band_hi = band_lo + pad, band_hi - pad

    def snr_to_y(v):
        """dB value -> y position inside the noise/signal gap."""
        frac = np.clip(v / snr_top, 0, 1)
        if log_scale:
            return 10 ** (np.log10(band_lo) + frac * (np.log10(band_hi) - np.log10(band_lo)))
        return band_lo + frac * (band_hi - band_lo)

    for row, ctype in enumerate(("strong", "weak")):
        ax = axes[row]
        ax2 = ax.twinx()
        centres = np.arange(1, len(group["trials"]) + 1) * GROUP_GAP

        for ci, (xc, t) in enumerate(zip(centres, group["trials"])):
            s = stats[(ctype, t["subject"])]
            # ---- signal box-and-whisker (pooled all channels x contractions) ----
            draw_box(ax, xc, s["signal"], SIGNAL_COLOR, "royalblue", fmt="{:.1f}")
            # ---- noise box-and-whisker (pooled all channels x rests 2-4) ----
            draw_box(ax, xc, s["noise"], NOISE_COLOR, "darkorange", fmt="{:.2f}")
            # ---- pooled SNR dot (mean per-channel dB), right axis, in the gap band ----
            ax2.scatter(xc, snr_to_y(s["snr_db"]), color="crimson", s=150, marker="o",
                        edgecolors="black", linewidths=1.5, zorder=5)
            ax2.annotate(f"{s['snr_db']:.1f}", (xc, snr_to_y(s["snr_db"])),
                         textcoords="offset points", xytext=(14, -3),
                         fontsize=13, ha="left", va="center", color="crimson",
                         fontweight="bold")

        ax.set_yscale("log" if log_scale else "linear")
        ax.set_ylim(*ylim)
        ax.set_xlim(centres[0] - 0.75, centres[-1] + 0.75)
        if log_scale:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ax.set_ylabel(f"{ctype.capitalize()} signal (µV)", fontsize=15)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5, which="major")
        ax.set_xticks(centres)
        ax.set_xticklabels(
            [f"{t['subject']}\n({', '.join(t['dates'])})" for t in group["trials"]],
            fontsize=14)
        ax2.set_ylim(0, snr_top)
        # place the dB ticks so they align with the mapped dot positions:
        # invert snr_to_y for a set of round dB values inside the band
        tick_db = np.arange(0, snr_top + 1, 10.0)
        tick_db = tick_db[(tick_db >= 0) & (tick_db <= snr_top)]
        tick_y = [snr_to_y(v) for v in tick_db]
        ax2.set_yticks(tick_y)
        ax2.set_yticklabels([f"{v:g}" for v in tick_db])
        ax2.set_ylim(min(tick_y) - 0.05 * (max(tick_y) - min(tick_y)),
                     max(tick_y) + 0.05 * (max(tick_y) - min(tick_y)))
        if row == 0:
            ax2.set_ylabel("SNR (dB)", color="crimson", fontsize=15)
        ax2.tick_params(axis="y", labelcolor="crimson", labelsize=13)
        ax.tick_params(axis="x", labelsize=14)
        ax.set_title(f"{ctype.capitalize()} contractions "
                     f"(all 16 channels pooled, n = "
                     f"{', '.join(str(len(stats[(ctype, t['subject'])]['signal'])) for t in group['trials'])} "
                     f"signal / "
                     f"{', '.join(str(len(stats[(ctype, t['subject'])]['noise'])) for t in group['trials'])} "
                     f"noise values per subject)",
                     fontsize=14, pad=10)

    handles = [
        Patch(facecolor=SIGNAL_COLOR, edgecolor="black", alpha=0.8,
              label="Signal (all channels)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="royalblue",
               markeredgecolor="black", markersize=8, label="Mean signal"),
        Patch(facecolor=NOISE_COLOR, edgecolor="black", alpha=0.8,
              label="Noise (all channels)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="darkorange",
               markeredgecolor="black", markersize=8, label="Mean noise"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson",
               markeredgecolor="black", markersize=10, label="SNR (mean per-channel dB)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=13, frameon=False)
    fig.suptitle(f"{group['connection']} recording, PGA gain {group['gain']} — "
                 f"signal, baseline noise and SNR (PC recordings)",
                 fontsize=17, y=0.995)
    fig.tight_layout(rect=[0, 0.07, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300)
    return fig


# ===== MAIN =====
def main():
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load(args.ods, args.sheet)
    ch = channel_table(df)
    groups = build_groups(df, ch)

    log_scale = not args.linear
    all_rows = []
    figs = []
    for group in groups:
        stem = str(out_dir / f"snr_allch_{group['connection'].lower()}_gain{group['gain']}")
        print(f"\n=== {group['connection']} gain {group['gain']} ===")
        for t in group["trials"]:
            print(f"  {t['subject']}: trials {', '.join(t['trial_ids'])}")
        figs.append(plot_figure(df, ch, group, log_scale, stem))

        # ---- CSV rows: exactly the numbers plotted ----
        for ctype in ("strong", "weak"):
            for t in group["trials"]:
                s = subject_stats(df, ch, t["trial_ids"], ctype)
                all_rows.append({
                    "connection": group["connection"], "gain": group["gain"],
                    "subject": t["subject"], "trial_ids": ";".join(t["trial_ids"]),
                    "contraction_type": ctype,
                    "n_channels": s["n_channels"],
                    "signal_n": len(s["signal"]),
                    "signal_min_uV": s["signal"].min(),
                    "signal_q1_uV": np.percentile(s["signal"], 25),
                    "signal_median_uV": np.median(s["signal"]),
                    "signal_q3_uV": np.percentile(s["signal"], 75),
                    "signal_mean_uV": s["signal"].mean(),
                    "signal_max_uV": s["signal"].max(),
                    "noise_n": len(s["noise"]),
                    "noise_min_uV": s["noise"].min(),
                    "noise_q1_uV": np.percentile(s["noise"], 25),
                    "noise_median_uV": np.median(s["noise"]),
                    "noise_q3_uV": np.percentile(s["noise"], 75),
                    "noise_mean_uV": s["noise"].mean(),
                    "noise_max_uV": s["noise"].max(),
                    "snr_db_pooled": s["snr_db"],
                })

    pd.DataFrame(all_rows).to_csv(out_dir / "snr_allch_plot_data.csv", index=False)

    print("\nSaved:")
    for f in sorted(out_dir.glob("snr_allch_*")):
        print(f"  {f.name}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()