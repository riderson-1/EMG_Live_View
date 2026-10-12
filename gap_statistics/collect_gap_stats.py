#!/usr/bin/env python3
# Compile the per-trial "<csv>-gap_statistic.log" files written by
# emg_gap_stat.py into one Excel-friendly CSV with two sections:
#   1) summary - one row per trial capture CSV
#   2) details - one row per gap
#
# Usage:
#   python collect_gap_stats.py --base <application_testing> --out <file.csv> [--fs 980]

import argparse
import csv
import os
import re
import statistics
import sys

SUMMARY_COLUMNS = [
    "session", "trial", "config", "subject", "task", "date", "recording",
    "csv_file", "csv_path", "fs",
    "samples_received", "samples_missing", "missing_pct", "num_gaps",
    "gap_mean_samples", "gap_sd_samples", "gap_median_samples",
    "gap_min_samples", "gap_max_samples",
    "gap_mean_ms", "gap_sd_ms", "gap_max_ms",
]

DETAIL_COLUMNS = [
    "session", "trial", "recording", "csv_file", "gap_no",
    "start_index", "length_samples", "length_ms",
]

# trial stem: <config>_<subject>_<task>_<YYYY-MM-DD>_<PC|SD>
TRIAL_RE = re.compile(
    r"^(?P<config>[A-Za-z]+_\d+)_(?P<subject>[^_]+)_(?P<task>.+)"
    r"_(?P<date>\d{4}-\d{2}-\d{2})$"
)
SAMPLES_RE = re.compile(
    r"^Samples: (\d+) received, (\d+) missing \(([\d.]+)%\)$", re.M)
GAPS_RE = re.compile(r"^Gaps: (\d+) total$", re.M)
METRIC_RE = re.compile(r"^\| ([^|]+?) \| ([^|]+?) \|$", re.M)
FS_RE = re.compile(r"Gap length mean \(ms @(\d+) Hz\)")
PAIR_RE = re.compile(r"^([\d.]+) \+/- ([\d.]+) \(SD\)$")
DETAIL_RE = re.compile(r"^\| (\d+) \| (\d+) \| (\d+) \|$", re.M)


def parse_trial_name(stem):
    """'USB_1_Karl_Fast_Walk_2026-09-02_PC' -> trial metadata dict."""
    if stem.endswith("_PC"):
        trial, recording = stem[:-len("_PC")], "PC"
    elif stem.endswith("_SD"):
        trial, recording = stem[:-len("_SD")], "SD"
    else:
        return None
    m = TRIAL_RE.match(trial)
    if not m:
        print(f"WARNING: cannot parse trial name '{trial}'", file=sys.stderr)
        return {"trial": trial, "recording": recording,
                "config": "", "subject": "", "task": "", "date": ""}
    return {"trial": trial, "recording": recording, **m.groupdict()}


def parse_log(path):
    """Parse one -gap_statistic.log file into summary fields + gap rows."""
    with open(path) as f:
        text = f.read()

    m = SAMPLES_RE.search(text)
    if not m:
        raise ValueError("'Samples:' line not found")
    received, missing, pct = int(m.group(1)), int(m.group(2)), m.group(3)

    m = GAPS_RE.search(text)
    if not m:
        raise ValueError("'Gaps:' line not found")
    num_gaps = int(m.group(1))

    metrics = dict(METRIC_RE.findall(text))

    def metric_prefix(prefix):
        for k, v in metrics.items():
            if k.startswith(prefix):
                return v
        return ""

    def metric_pair(prefix):
        m = PAIR_RE.match(metric_prefix(prefix))
        return (m.group(1), m.group(2)) if m else ("", "")

    fs_m = FS_RE.search(text)
    fs = int(fs_m.group(1)) if fs_m else None

    if num_gaps:
        mean_s, sd_s = metric_pair("Gap length mean (samples)")
        mean_ms, sd_ms = metric_pair("Gap length mean (ms")
        stats = {
            "gap_mean_samples": mean_s,
            "gap_sd_samples": sd_s,
            "gap_median_samples": metric_prefix("Gap length median (samples)"),
            "gap_min_samples": metric_prefix("Gap length min (samples)"),
            "gap_max_samples": metric_prefix("Gap length max (samples)"),
            "gap_mean_ms": mean_ms,
            "gap_sd_ms": sd_ms,
            "gap_max_ms": metric_prefix("Gap length max (ms"),
        }
        for key, val in stats.items():
            if not val:
                raise ValueError(f"metric '{key}' missing from table")
    else:
        stats = {k: "" for k in SUMMARY_COLUMNS if k.startswith("gap_")}

    gaps = [(int(a), int(b), int(c))
            for a, b, c in DETAIL_RE.findall(text)]
    if len(gaps) != num_gaps:
        raise ValueError(f"{num_gaps} gaps reported but {len(gaps)} detail rows")

    return {"received": received, "missing": missing, "pct": pct,
            "num_gaps": num_gaps, "fs": fs, "stats": stats, "gaps": gaps}


def main():
    ap = argparse.ArgumentParser(
        description="Compile per-trial -gap_statistic.log files into one CSV.")
    ap.add_argument("--base", required=True,
                    help="Root folder searched for *_PC.csv / *_SD.csv trial captures.")
    ap.add_argument("--out", required=True, help="Output CSV path.")
    ap.add_argument("--fs", type=int, default=980,
                    help="Fallback sampling rate (Hz) if a log does not record one.")
    args = ap.parse_args()

    if not os.path.isdir(args.base):
        sys.exit(f"Base directory not found: {args.base}")

    csvs = sorted(
        os.path.join(dp, fn)
        for dp, _, fns in os.walk(args.base)
        for fn in fns
        if fn.endswith("_PC.csv") or fn.endswith("_SD.csv"))
    if not csvs:
        sys.exit(f"No trial captures (*_PC.csv / *_SD.csv) under {args.base}")

    summary_rows, detail_rows = [], []
    problems = []
    for path in csvs:
        stem = os.path.splitext(os.path.basename(path))[0]
        log_path = os.path.splitext(path)[0] + "-gap_statistic.log"
        if not os.path.isfile(log_path):
            problems.append(f"missing log: {log_path}")
            continue
        try:
            parsed = parse_log(log_path)
        except ValueError as e:
            problems.append(f"{log_path}: {e}")
            continue

        meta = parse_trial_name(stem)
        if meta is None:
            problems.append(f"unexpected file name: {path}")
            continue
        session = os.path.basename(os.path.dirname(path))
        csv_file = os.path.basename(path)
        fs = parsed["fs"] if parsed["fs"] else args.fs

        summary_rows.append([
            session, meta["trial"], meta["config"], meta["subject"],
            meta["task"], meta["date"], meta["recording"],
            csv_file, os.path.relpath(path, args.base), fs,
            parsed["received"], parsed["missing"], parsed["pct"],
            parsed["num_gaps"],
            parsed["stats"]["gap_mean_samples"], parsed["stats"]["gap_sd_samples"],
            parsed["stats"]["gap_median_samples"],
            parsed["stats"]["gap_min_samples"], parsed["stats"]["gap_max_samples"],
            parsed["stats"]["gap_mean_ms"], parsed["stats"]["gap_sd_ms"],
            parsed["stats"]["gap_max_ms"],
        ])
        for i, (start, length, ms) in enumerate(parsed["gaps"], start=1):
            detail_rows.append([session, meta["trial"], meta["recording"],
                                csv_file, i, start, length, ms])

    total_gaps = sum(int(r[SUMMARY_COLUMNS.index("num_gaps")])
                     for r in summary_rows)
    if total_gaps != len(detail_rows):
        problems.append(f"internal error: summary says {total_gaps} gaps, "
                        f"detail section has {len(detail_rows)}")

    # ===== OVERALL SUMMARY across all trials =====
    ci = {name: i for i, name in enumerate(SUMMARY_COLUMNS)}
    di = {name: i for i, name in enumerate(DETAIL_COLUMNS)}
    trials = len(summary_rows)
    total_received = sum(int(r[ci["samples_received"]]) for r in summary_rows)
    total_missing = sum(int(r[ci["samples_missing"]]) for r in summary_rows)
    overall_pct = 100.0 * total_missing / (total_received + total_missing)
    trial_pcts = [float(r[ci["missing_pct"]]) for r in summary_rows]
    trial_gaps = [int(r[ci["num_gaps"]]) for r in summary_rows]
    gap_s = [int(r[di["length_samples"]]) for r in detail_rows]
    gap_ms = [int(r[di["length_ms"]]) for r in detail_rows]

    def stats_block(prefix, values, fmt):
        return [
            (f"{prefix}_mean", f"{statistics.mean(values):{fmt}}"),
            (f"{prefix}_sd", f"{statistics.pstdev(values):{fmt}}"),
            (f"{prefix}_median", f"{statistics.median(values):{fmt}}"),
            (f"{prefix}_min", f"{min(values):{fmt}}"),
            (f"{prefix}_max", f"{max(values):{fmt}}"),
        ]

    overall_rows = [
        ("trials_capture_files", trials),
        ("total_samples_received", total_received),
        ("total_samples_missing", total_missing),
        ("overall_missing_pct", f"{overall_pct:.2f}"),
        ("total_gaps", total_gaps),
    ]
    overall_rows += stats_block("trial_missing_pct", trial_pcts, ".2f")
    overall_rows += stats_block("trial_num_gaps", trial_gaps, ".1f")
    if gap_s:
        overall_rows += stats_block("gap_length_samples", gap_s, ".1f")
        overall_rows += stats_block("gap_length_ms", gap_ms, ".1f")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow([f"# Gap statistics summary - one row per trial capture CSV "
                    f"({len(summary_rows)} files)"])
        w.writerow(SUMMARY_COLUMNS)
        w.writerows(summary_rows)
        w.writerow([])
        w.writerow([f"# Gap details - one row per gap ({len(detail_rows)} gaps)"])
        w.writerow(DETAIL_COLUMNS)
        w.writerows(detail_rows)
        w.writerow([])
        w.writerow([f"# Overall summary - all {trials} trial capture CSVs"])
        w.writerow(["metric", "value"])
        w.writerows(overall_rows)

    print(f"Summary rows: {len(summary_rows)}")
    print(f"Gap detail rows: {len(detail_rows)}")
    print(f"Output: {args.out}")

    if problems:
        print("\nPROBLEMS:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()