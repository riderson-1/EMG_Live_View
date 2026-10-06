#!/usr/bin/env python3
"""Plot per-thread CPU load over time from a Zephyr thread_analyzer log.

Usage: python plot_cpu_load.py session_0013.log [--stacked] [--cumulative] [--min-peak 1] [--save out.png]
"""
import argparse
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt

TS_RE = re.compile(r"\[(\d+):(\d{2}):(\d{2})\.(\d{3}),(\d{3})\]")
# a complete timestamped log line (used to strip lines spliced into analyzer output)
LOGLINE_RE = re.compile(r"\[\d+:\d{2}:\d{2}\.\d{3},\d{3}\] <\w+> [^\r\n]*\r?\n")
THREAD_RE = re.compile(r"^\s*(.+?)\s*: STACK: .*?CPU:\s*(\d+)\s*%", re.M)
CYCLES_RE = re.compile(r"Total CPU cycles used:\s*(\d+)")


def ts_to_s(m):
    h, mi, s, ms, us = map(int, m.groups())
    return h * 3600 + mi * 60 + s + ms / 1e3 + us / 1e6


def parse(path):
    """Return {thread_name: [(time_s, cpu_percent), ...]}."""
    with open(path, errors="replace") as f:
        lines = f.read().splitlines(keepends=True)

    # 1) Find analyzer blocks and stamp each with the last timestamp seen before it.
    last_t = 0.0
    blocks = []  # (time, [raw lines])
    cur = None
    for line in lines:
        if "Thread analyze:" in line:
            cur = (last_t, [])
            blocks.append(cur)
            continue
        m = TS_RE.search(line)
        if m:
            last_t = ts_to_s(m)
        if cur is not None:
            cur[1].append(line)

    # 2) Per block: drop interleaved log lines (they can splice into the middle
    #    of an analyzer line, so remove them first, then the pieces re-join).
    series = defaultdict(list)   # name -> [(t, cpu% as printed)]
    cycles = defaultdict(list)   # name -> [(t, cumulative cycles)]
    for t, raw in blocks:
        text = LOGLINE_RE.sub("", "".join(raw))
        text = text.replace("\r", "")
        for name, cpu in THREAD_RE.findall(text):
            series[name].append((t, int(cpu)))
        cur = None
        for line in text.split("\n"):
            m = THREAD_RE.match(line)
            if m:
                cur = m.group(1)
                continue
            m = CYCLES_RE.search(line)
            if m and cur:
                cycles[cur].append((t, int(m.group(1))))
                cur = None
    return series, cycles


def interval_load(cycles):
    """Per-interval CPU %: delta cycles of a thread / delta cycles of all threads."""
    names = list(cycles)
    n = min(len(v) for v in cycles.values())
    times = [cycles[names[0]][i][0] for i in range(1, n)]
    out = {nm: [] for nm in names}
    for i in range(1, n):
        d = {nm: cycles[nm][i][1] - cycles[nm][i - 1][1] for nm in names}
        tot = sum(d.values()) or 1
        for nm in names:
            out[nm].append(100.0 * d[nm] / tot)
    return {nm: list(zip(times, v)) for nm, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--stacked", action="store_true", help="stacked area instead of lines")
    ap.add_argument("--min-peak", type=float, default=1.0,
                    help="hide threads whose peak CPU %% is below this (default 1)")
    ap.add_argument("--cumulative", action="store_true",
                    help="plot the analyzer's printed CPU %% (averaged since boot) instead of per-interval load")
    ap.add_argument("--save", help="save figure to this file instead of showing it")
    a = ap.parse_args()

    series, cycles = parse(a.log)
    if not a.cumulative and cycles:
        series = interval_load(cycles)
    if not series:
        sys.exit("No thread analyzer data found.")

    # sort by mean load so the busiest threads come first in the legend
    names = sorted(series, key=lambda n: -sum(c for _, c in series[n]) / len(series[n]))
    shown = [n for n in names if max(c for _, c in series[n]) >= a.min_peak]
    hidden = [n for n in names if n not in shown]

    fig, ax = plt.subplots(figsize=(13, 6))
    if a.stacked:
        times = [t for t, _ in series[shown[0]]]
        ys = [[c for _, c in series[n]][:len(times)] for n in shown]
        ax.stackplot(times, ys, labels=shown)
        ax.set_ylim(0, 100)
    else:
        for n in shown:
            ts, cs = zip(*series[n])
            ax.plot(ts, cs, marker=".", ms=3, lw=1.2, label=n)
    ax.set_xlabel("time since boot [s]")
    ax.set_ylabel("CPU load [%]")
    ax.set_title("Per-thread CPU load (%s)" % ("as printed, avg since boot" if a.cumulative else "per analyzer interval"))
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.tight_layout()

    if hidden:
        print("Hidden (peak < %.1f%%): %s" % (a.min_peak, ", ".join(hidden)))
    print("Threads plotted:", ", ".join(shown))
    if a.save:
        fig.savefig(a.save, dpi=150)
        print("Saved", a.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()