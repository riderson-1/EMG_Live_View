#!/usr/bin/env python3
"""
SNR plots for the six PC isometric trials (one figure for STRONG, one for WEAK).

Input : emg_isometric_summary.ods (sheet 'snr', all values in uV)
Output: snr_strong.(png|pdf), snr_weak.(png|pdf), snr_plot_data.csv

Method (per trial, PC recordings only)
  1. Noise floor of a channel = mean (or median, --noise-stat) of noise_uV over
     the weak contractions 2..4 (chronological rank within the weak
     contractions). The rest before the FIRST weak contraction is excluded
     because it directly follows the maximal-effort strong block (noise is
     10-20x higher there in 4 of 6 trials); rests 2..4 follow submaximal weak
     contractions. The same noise value is used for the strong and weak plot.
  2. Signal values = signal_uV of every strong (or weak) contraction of the
     channel; the box spans min..max, the diamond is the mean.
  3. SNR (dB) = 20*log10(mean signal / noise floor), one dot per trial and plot.
  4. Channel per trial = highest mean of (strong SNR dB, weak SNR dB), so one
     channel is used for both plots.
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
SOURCE = "PC"                 # recording source to use (SD trials excluded for now)
NOISE_SKIP_FIRST_WEAK = 1     # number of leading weak rests excluded from the noise floor
BOX_WIDTH = 0.4
NOISE_HALF_WIDTH = 0.25
DOT_OFFSET = 0.35             # x offset of the SNR dot from the trial centre (0 = centred)

REQUIRED_COLS = ["trial_id", "subject", "date", "connection", "source", "channel",
                 "contraction_type", "window_start_s", "signal_uV", "noise_uV"]

# ===== ARGUMENTS =====
parser = argparse.ArgumentParser(description="Strong/weak SNR plots for the six PC trials.")
parser.add_argument("ods", help="Path to the summary workbook "
                                 "(e.g. /home/karl/Documents/Master_Thesis/Testing/"
                                 "application_testing/emg_isometric_summary.ods)")
parser.add_argument("--sheet", default="snr", help="Sheet with per-contraction data (default: snr)")
parser.add_argument("--out-dir", default=".", help="Folder for the output files (default: .)")
parser.add_argument("--noise-stat", choices=["mean", "median"], default="mean",
                    help="Statistic over weak rests 2-4 used as the noise floor (default: mean).")
parser.add_argument("--linear", action="store_true",
                    help="Linear uV axis instead of log (noise bars become hard to see).")
parser.add_argument("--separate-ylim", action="store_true",
                    help="Use separate y-limits for the strong and the weak plot.")
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
    # the PGA gain lives in the 'trials' sheet (parsed from the log command line);
    # join it onto the per-contraction rows by trial_id
    trials_sheet = pd.read_excel(path, sheet_name="trials")
    if "gain" not in trials_sheet.columns:
        raise SystemExit("No 'gain' column in sheet 'trials'. "
                         "Regenerate the workbook with parse_emg_logs.py.")
    df = df.merge(trials_sheet[["trial_id", "gain"]], on="trial_id", how="left")
    df = df[df["source"] == SOURCE].copy()
    if df.empty:
        raise SystemExit(f"No rows with source == '{SOURCE}'.")
    bad = df[~df["contraction_type"].isin(["strong", "weak"])]
    if len(bad):
        raise SystemExit("contraction_type is missing/invalid for trials: "
                         + ", ".join(sorted(bad["trial_id"].unique())))
    return df


# ===== ANALYSIS =====
def build_trials(df):
    # chronological rank of each weak contraction within (trial, channel)
    weak = df[df["contraction_type"] == "weak"].copy()
    weak["weak_rank"] = (weak.groupby(["trial_id", "channel"])["window_start_s"]
                         .rank(method="first").astype(int))
    keys = ["trial_id", "channel"]

    used = weak[weak["weak_rank"] > NOISE_SKIP_FIRST_WEAK]
    noise = used.groupby(keys)["noise_uV"].agg(noise_uV=args.noise_stat, noise_sd="std", noise_n="count")
    excluded = (weak[weak["weak_rank"] <= NOISE_SKIP_FIRST_WEAK]
                .groupby(keys)["noise_uV"].mean().rename("excluded_rest_noise_uV"))

    sig = (df.groupby(keys + ["contraction_type"])["signal_uV"].mean()
           .unstack("contraction_type")
           .rename(columns={"strong": "sig_strong_mean", "weak": "sig_weak_mean"}))
    ch = sig.join(noise).join(excluded)
    ch["snr_strong_db"] = 20 * np.log10(ch["sig_strong_mean"] / ch["noise_uV"])
    ch["snr_weak_db"] = 20 * np.log10(ch["sig_weak_mean"] / ch["noise_uV"])
    ch["score_db"] = ch[["snr_strong_db", "snr_weak_db"]].mean(axis=1)

    best_idx = ch["score_db"].groupby(level="trial_id").idxmax()   # trial -> (trial, channel)

    trials = []
    for tid, (_, chan) in best_idx.items():
        sub = df[(df["trial_id"] == tid) & (df["channel"] == chan)].sort_values("window_start_s")
        first = sub.iloc[0]
        trials.append({
            "trial_id": tid,
            "channel": chan,
            "subject": first["subject"],
            "date": str(first["date"])[:10],
            "connection": first["connection"],
            "gain": int(first["gain"]),
            "noise": float(ch.loc[(tid, chan), "noise_uV"]),
            "strong": sub.loc[sub["contraction_type"] == "strong", "signal_uV"].to_numpy(float),
            "weak": sub.loc[sub["contraction_type"] == "weak", "signal_uV"].to_numpy(float),
        })
    trials.sort(key=lambda d: (d["date"], d["subject"], d["gain"]))   # chronological
    return trials, ch


def snr_db(values, noise):
    return 20 * np.log10(np.mean(values) / noise)


# ===== PLOT =====
def plot_snr(trials, ctype, ylim_left, ylim_right, log_scale, out_stem):
    x = np.arange(1, len(trials) + 1)
    fig, ax1 = plt.subplots(figsize=(12, 7.5))

    for xi, t in zip(x, trials):
        vals = t[ctype]
        lo, hi, mean = vals.min(), vals.max(), vals.mean()
        # box = full range min..max
        ax1.add_patch(Rectangle((xi - BOX_WIDTH / 2, lo), BOX_WIDTH, hi - lo,
                                facecolor="lightblue", edgecolor="black",
                                alpha=0.7, linewidth=1.2, zorder=2))
        ax1.scatter(xi, mean, marker="D", s=70, facecolor="royalblue",
                    edgecolor="black", zorder=4)
        # one noise value per trial (same for strong and weak)
        ax1.hlines(t["noise"], xi - NOISE_HALF_WIDTH, xi + NOISE_HALF_WIDTH,
                   colors="dimgray", linewidths=3, zorder=4)

    ax1.set_yscale("log" if log_scale else "linear")
    ax1.set_ylim(*ylim_left)
    ax1.set_xlim(0.5, len(trials) + 0.5)
    if log_scale:
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax1.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax1.set_ylabel("EMG envelope amplitude (µV)", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{t['subject']}\n{t['date']}\n{t['connection']}, gain {t['gain']}\n{t['channel']}"
                         for t in trials], fontsize=9)
    ax1.set_xlabel("Trial (subject, date, connection, PGA gain, selected channel)", fontsize=12)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5, which="major")

    # SNR dots on the right axis (mean signal / trial noise floor),
    # centred on the trial so box, noise bar and SNR dot align vertically
    ax2 = ax1.twinx()
    snrs = [snr_db(t[ctype], t["noise"]) for t in trials]
    ax2.scatter(x, snrs, color="crimson", s=150, marker="o",
                edgecolors="black", linewidths=1.5, zorder=5)
    ax2.set_ylim(*ylim_right)
    ax2.set_ylabel("SNR (dB)", color="crimson", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="crimson")

    handles = [
        Patch(facecolor="lightblue", edgecolor="black", alpha=0.7, label="Signal range (min–max)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="royalblue",
               markeredgecolor="black", markersize=8, label="Mean signal"),
    ]
    handles += [
        Line2D([0], [0], color="dimgray", linewidth=3, label=f"Baseline noise ({args.noise_stat} of weak rests 2–4)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson",
               markeredgecolor="black", markersize=10, label="SNR = 20·log10(mean signal / noise)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    ax1.set_title(f"{ctype.capitalize()} contractions: signal range, baseline noise and SNR "
                  f"(n = {', '.join(str(len(t[ctype])) for t in trials)} per trial)",
                  fontsize=14, pad=14)
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300)
    return fig


def main():
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load(args.ods, args.sheet)
    trials, ch = build_trials(df)
    if len(trials) != 6:
        print(f"Warning: expected 6 PC trials, found {len(trials)}.")

    # ---- console summary + CSV of exactly what is plotted ----
    rows = []
    print(f"{'trial':34s} {'ch':5s} {'noise µV':>9s} {'(sd, n)':>13s} {'excl. rest':>10s}"
          f" {'S mean':>8s} {'W mean':>8s} {'S dB':>6s} {'W dB':>6s}")
    for t in trials:
        r = ch.loc[(t["trial_id"], t["channel"])]
        s_db, w_db = snr_db(t["strong"], t["noise"]), snr_db(t["weak"], t["noise"])
        print(f"{t['trial_id']:34s} {t['channel']:5s} {t['noise']:9.2f} "
              f"({r['noise_sd']:5.2f}, {int(r['noise_n'])}) {r['excluded_rest_noise_uV']:10.2f}"
              f" {t['strong'].mean():8.1f} {t['weak'].mean():8.1f} {s_db:6.1f} {w_db:6.1f}")
        rows.append({
            "trial_id": t["trial_id"], "subject": t["subject"], "date": t["date"],
            "connection": t["connection"], "gain": t["gain"], "channel": t["channel"],
            "noise_uV": t["noise"], "noise_sd_uV": r["noise_sd"], "noise_n": int(r["noise_n"]),
            "excluded_first_weak_rest_noise_uV": r["excluded_rest_noise_uV"],
            "strong_n": len(t["strong"]), "strong_min_uV": t["strong"].min(),
            "strong_mean_uV": t["strong"].mean(), "strong_max_uV": t["strong"].max(),
            "strong_snr_db": s_db,
            "weak_n": len(t["weak"]), "weak_min_uV": t["weak"].min(),
            "weak_mean_uV": t["weak"].mean(), "weak_max_uV": t["weak"].max(),
            "weak_snr_db": w_db,
        })
    pd.DataFrame(rows).to_csv(out_dir / "snr_plot_data.csv", index=False)

    # ---- axis limits (shared by both plots unless --separate-ylim) ----
    def limits(types):
        vals = np.concatenate([t[c] for t in trials for c in types])
        noise = np.array([t["noise"] for t in trials])
        snrs = [snr_db(t[c], t["noise"]) for t in trials for c in types]
        top_db = np.ceil(max(snrs) * 1.15 / 10) * 10
        if args.linear:
            left = (0, vals.max() * 1.1)
        else:
            left = (noise.min() * 0.5, vals.max() * 1.5)
        return left, (0, top_db)

    log_scale = not args.linear
    figs = []
    for ctype in ("strong", "weak"):
        yl, yr = limits(("strong", "weak") if not args.separate_ylim else (ctype,))
        figs.append(plot_snr(trials, ctype, yl, yr, log_scale,
                             str(out_dir / f"snr_{ctype}")))
    print(f"\nSaved: {out_dir / 'snr_strong.png'}, {out_dir / 'snr_weak.png'} (+ .pdf), "
          f"{out_dir / 'snr_plot_data.csv'}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()