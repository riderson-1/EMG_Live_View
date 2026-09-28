"""
Parse the *_analysis.log files produced by emg_isometric.py --save-results
(one per Iso_Dor trial, PC + SD variants) into tidy DataFrames and write an
ODS workbook in the style of the EMG metric summary:

    trials  - one row per trial            (data-quality / lost-samples / settings)
    mdf     - one row per trial x channel  (per-channel summary: signal, noise, SNR, MDF)
    snr     - one row per trial x channel x contraction (finest-grain SNR)
    gaps    - one row per trial x gap event

All signal and noise values are in MICROVOLTS (µV); the column names carry the
_uV suffix explicitly.

Every table is located by searching for its header cells rather than assuming
a fixed position, so the parser tolerates layout changes.

Usage:
    python3 parse_emg_logs.py                 # parse all Iso_Dor logs, write ODS
    python3 parse_emg_logs.py --out out.ods   # custom output path
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

BASE = "/home/karl/Documents/Master_Thesis/Testing/application_testing"
DEFAULT_OUT = os.path.join(BASE, "emg_isometric_summary.ods")

# trial folder name, e.g. USB_1_Karl_Iso_Dor_2026-09-02
DIR_RE = re.compile(
    r"^(?P<connection>USB|BLE)_(?P<rep>\d+)_(?P<subject>[A-Za-z]+)_Iso_Dor_"
    r"(?P<date>\d{4}-\d{2}-\d{2})$"
)
# log/csv file stem, e.g. USB_1_Karl_Iso_Dor_2026-09-02_PC or ..._SD_cropped
STEM_RE = re.compile(
    r"^(?P<connection>USB|BLE)_(?P<rep>\d+)_(?P<subject>[A-Za-z]+)_Iso_Dor_"
    r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<source>PC|SD_cropped|SD_shortened|SD)$"
)


def cellstr(v):
    return "" if v is None else str(v).strip()


def num(s):
    """Extract the first number from a string ('1.61e+08 µV' -> 161000000.0)."""
    if s is None:
        return None
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(s))
    return float(m.group()) if m else None


def parse_mean_sd(s):
    """'201.5 +/- 170.5 (SD)' -> (201.5, 170.5)"""
    if s is None:
        return None, None
    m = re.match(r"\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\+/-\s*"
                 r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", str(s))
    if m:
        return float(m.group(1)), float(m.group(2))
    return num(s), None


def parse_missing(s):
    """'403 (0.16%)' -> (403, 0.16)"""
    if s is None:
        return None, None
    m = re.match(r"\s*([\d,]+)\s*\(([\d.]+)%\)", str(s))
    if m:
        return int(m.group(1).replace(",", "")), float(m.group(2))
    return None, None


def find_table_header(lines, header_cells, start=0):
    """Find a line index >= start whose markdown table header starts with
    header_cells (in order). Header cells may be matched by prefix, so
    'Signal' matches 'Signal (µV)'. Returns (line_idx, col_texts) or (None, None)."""
    for r in range(start, len(lines)):
        cells = [c.strip() for c in lines[r].strip().strip("|").split("|")]
        cells = [c for c in cells if c]
        if len(cells) >= len(header_cells) and all(
            cells[i].startswith(header_cells[i]) for i in range(len(header_cells))
        ):
            return r, cells[: len(header_cells)]
    return None, None


def read_table(lines, header_cells, start_search=0):
    """Locate a header row (>= start_search), skip the dashed separator, and
    read data rows until a blank line. Returns (DataFrame, next_line_idx)."""
    r, cols = find_table_header(lines, header_cells, start_search)
    if r is None:
        return None, start_search
    i = r + 1
    # skip the |---|---| separator row
    if i < len(lines) and set(lines[i]) <= set("|- "):
        i += 1
    rows = []
    while i < len(lines):
        line = lines[i].strip()
        if not line or not line.startswith("|"):
            break
        vals = [c.strip() for c in line.strip("|").split("|")]
        # pad/truncate to header width
        vals = (vals + [""] * len(cols))[: len(cols)]
        rows.append(vals)
        i += 1
    return pd.DataFrame(rows, columns=cols), i


def parse_metric_value(lines):
    """'### Gap statistics' Metric/Value table -> dict."""
    table, _ = read_table(lines, ["Metric", "Value"])
    if table is None:
        return {}
    return dict(zip(table["Metric"], table["Value"]))


def parse_contraction_windows(lines):
    """Ordered list of (window_str, type) from the '- **Strong** (...)' /
    '- **Weak** (...)' lines (strong first, then weak — same order the
    per-channel detail tables use)."""
    windows = []
    for tag in ("Strong", "Weak"):
        for line in lines:
            m = re.search(rf"\*\*{tag}\*\*\s*\(\d+\):\s*(.+)", line)
            if not m:
                continue
            for t in m.group(1).split(","):
                t = t.strip().rstrip(".").rstrip("s")
                if t:
                    windows.append((t, tag.lower()))
            break
    return windows


def parse_command(lines):
    """Parse the '# Command: ...' line into a dict of settings."""
    cmd = ""
    for line in lines:
        if line.startswith("# Command:"):
            cmd = line
            break
    settings = {}
    m = re.search(r"--gain (\S+)", cmd)
    if m:
        settings["gain"] = float(m.group(1))
    m = re.search(r"--unit (\S+)", cmd)
    settings["unit"] = m.group(1) if m else "auto"
    m = re.search(r"--channels ([\d ]+?)(?: --|$)", cmd)
    if m:
        settings["channels"] = m.group(1).strip()
    m = re.search(r"--thresh-strong (\S+)", cmd)
    if m:
        settings["thresh_strong_pct"] = float(m.group(1))
    m = re.search(r"--thresh-weak (\S+)", cmd)
    if m:
        settings["thresh_weak_pct"] = float(m.group(1))
    m = re.search(r"--signal-trim (\S+)", cmd)
    if m:
        settings["signal_trim"] = float(m.group(1))
    m = re.search(r"--noise-offset (\S+)", cmd)
    if m:
        settings["noise_offset_s"] = float(m.group(1))
    m = re.search(r"--noise-window (\S+)", cmd)
    if m:
        settings["noise_window_pct"] = float(m.group(1))
    m = re.search(r"--psd (\S+)", cmd)
    if m:
        settings["psd_method"] = m.group(1)
    settings["manual_windows"] = "--strong-windows" in cmd or "--weak-windows" in cmd
    return settings


def parse_unit(lines):
    """'- Values in **µV** (gain 1 applied: ...)' -> 'µV'"""
    for line in lines:
        m = re.search(r"Values in \*\*(.+?)\*\*", line)
        if m:
            return m.group(1)
    return None


def parse_log_file(path):
    """Parse one analysis log file. Returns (trial_row, mdf_rows, snr_rows, gap_rows)."""
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    stem = os.path.basename(path)[: -len("_analysis.log")]
    m = STEM_RE.match(stem)
    if not m:
        raise ValueError(f"Log file name doesn't match expected pattern: {path}")
    meta = m.groupdict()
    meta["rep"] = int(meta["rep"])
    meta["trial_id"] = stem

    # --- gap statistics ---
    metrics = parse_metric_value(lines)
    gap_count = num(metrics.get("Number of gaps"))
    missing_n, missing_pct = parse_missing(metrics.get("Samples missing"))
    gap_mean_ms, gap_mean_sd_ms = parse_mean_sd(metrics.get("Gap length mean (ms @1000 Hz)"))
    gap_max_ms = num(metrics.get("Gap length max (ms @1000 Hz)"))

    received = None
    for line in lines:
        m2 = re.search(r"Samples:\s*([\d,]+)\s*received", line)
        if m2:
            received = int(m2.group(1).replace(",", ""))
            break

    # --- contractions ---
    windows = parse_contraction_windows(lines)
    n_strong = sum(1 for _, t in windows if t == "strong")
    n_weak = sum(1 for _, t in windows if t == "weak")

    trial_row = {
        **meta,
        **parse_command(lines),
        "unit": parse_unit(lines),
        "samples_received": received,
        "samples_missing": missing_n,
        "missing_pct": missing_pct,
        "gap_count": gap_count,
        "gap_len_mean_ms": gap_mean_ms,
        "gap_len_mean_sd_ms": gap_mean_sd_ms,
        "gap_len_max_ms": gap_max_ms,
        "n_contractions_strong": n_strong,
        "n_contractions_weak": n_weak,
        "n_contractions_total": n_strong + n_weak,
        "log_path": path,
    }

    # --- gap details table ---
    gap_rows = []
    gtable, _ = read_table(lines, ["Start index", "Length (samples)", "Length (ms)"])
    if gtable is not None:
        for i, r in gtable.iterrows():
            gap_rows.append({
                "trial_id": meta["trial_id"],
                "gap_idx": i + 1,
                "start_index": num(r["Start index"]),
                "length_samples": num(r["Length (samples)"]),
                "length_ms": num(r["Length (ms)"]),
            })

    # --- per-channel summary table ---
    mdf_rows, snr_rows = [], []
    chan_table, _ = read_table(
        lines, ["Channel", "Signal", "Noise", "SNR", "SNR (dB)", "MDF (Hz)", "n"]
    )
    if chan_table is not None:
        # headers carry unit suffixes ('Signal (µV)') -> rename to canonical names
        # (only strip from Signal/Noise; keep 'SNR (dB)' and 'MDF (Hz)' intact)
        chan_table = chan_table.rename(
            columns={c: c.split(" (")[0] for c in chan_table.columns
                     if c.startswith(("Signal", "Noise"))}
        )
        trial_row["has_channel_data"] = True
        for _, r in chan_table.iterrows():
            channel = r["Channel"]
            mdf_rows.append({
                "trial_id": meta["trial_id"],
                "channel": channel,
                "signal_channel_mean": num(r["Signal"]),
                "noise_channel_mean": num(r["Noise"]),
                "snr_channel_mean": num(r["SNR"]),
                "snr_db_channel_mean": num(r["SNR (dB)"]),
                "mdf_hz": num(r["MDF (Hz)"]),
                "n_contractions_used": num(r["n"]),
            })

            # --- per-contraction detail table for this channel ---
            hdr = None
            for li, line in enumerate(lines):
                if line.strip().startswith(f"#### {channel} per-contraction detail"):
                    hdr = li
                    break
            if hdr is None:
                continue
            detail, _ = read_table(
                lines, ["Window (s)", "Signal", "Noise", "SNR", "SNR (dB)"],
                start_search=hdr,
            )
            if detail is None:
                continue
            detail = detail.rename(
                columns={c: c.split(" (")[0] for c in detail.columns
                         if c.startswith(("Signal", "Noise"))}
            )
            for i, r in detail.iterrows():
                win = r["Window (s)"]
                ctype = None
                if i < len(windows) and windows[i][0] == win:
                    ctype = windows[i][1]
                start_s = end_s = None
                if "-" in win:
                    a, b = win.split("-", 1)
                    start_s, end_s = num(a), num(b)
                snr_rows.append({
                    "trial_id": meta["trial_id"],
                    "channel": channel,
                    "contraction_idx": i + 1,
                    "contraction_type": ctype,
                    "window_start_s": start_s,
                    "window_end_s": end_s,
                    "signal": num(r["Signal"]),
                    "noise": num(r["Noise"]),
                    "snr": num(r["SNR"]),
                    "snr_db": num(r["SNR (dB)"]),
                })

    # --- best / worst channels ---
    for line in lines:
        m2 = re.search(r"\*\*BEST channel:\*\* (Ch\d+)", line)
        if m2:
            trial_row["best_channel"] = m2.group(1)
        m2 = re.search(r"\*\*WORST channel:\*\* (Ch\d+)", line)
        if m2:
            trial_row["worst_channel"] = m2.group(1)

    return trial_row, mdf_rows, snr_rows, gap_rows


def find_log_files(base=BASE):
    """All *_analysis.log files inside any *Iso_Dor* trial folder."""
    logs = []
    for dirpath, dirnames, filenames in os.walk(base):
        if not DIR_RE.match(os.path.basename(dirpath)):
            continue
        for fn in sorted(filenames):
            if fn.endswith("_analysis.log"):
                logs.append(os.path.join(dirpath, fn))
    return sorted(logs)


def build_dataframes(base=BASE):
    logs = find_log_files(base)
    if not logs:
        raise SystemExit(f"No *_analysis.log files found under {base}")
    trial_rows, mdf_rows, snr_rows, gap_rows = [], [], [], []
    for path in logs:
        t, m, s, g = parse_log_file(path)
        trial_rows.append(t)
        mdf_rows.extend(m)
        snr_rows.extend(s)
        gap_rows.extend(g)
        print(f"parsed: {path}  (channels={len(m)}, contractions={len(s)}, gaps={len(g)})")

    id_cols = ["trial_id", "connection", "rep", "subject", "date", "source"]

    trials = pd.DataFrame(trial_rows)
    trials = trials[id_cols + [c for c in trials.columns if c not in id_cols]]

    mdf = pd.DataFrame(mdf_rows)
    mdf = trials[id_cols].merge(mdf, on="trial_id", how="right")
    # all signal/noise values are in microvolts (µV) — mark it in the column names
    mdf = mdf.rename(columns={
        "signal_channel_mean": "signal_channel_mean_uV",
        "noise_channel_mean": "noise_channel_mean_uV",
    })

    snr = pd.DataFrame(snr_rows)
    snr = trials[id_cols].merge(snr, on="trial_id", how="right")
    snr = snr.rename(columns={
        "signal": "signal_uV",
        "noise": "noise_uV",
    })

    gaps = pd.DataFrame(gap_rows)
    gaps = trials[id_cols].merge(gaps, on="trial_id", how="right")

    return {"trials": trials, "mdf": mdf, "snr": snr, "gaps": gaps}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=BASE, help="Root folder to search for logs")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output ODS file path")
    args = ap.parse_args()

    dfs = build_dataframes(args.base)
    for name, d in dfs.items():
        print(f"{name}: {d.shape}")

    with pd.ExcelWriter(args.out, engine="odf") as writer:
        for name, d in dfs.items():
            d.to_excel(writer, sheet_name=name, index=False)
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
