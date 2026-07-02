#!/usr/bin/env python3
"""
Live EMG plotter for the Sokosti ADS1299 CDC-ACM stream.

Expects CSV lines of the form:
    sample,status_ok,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8
(with a one-time header line "sample,status_ok,ch1,..." which is skipped)

- Serial reading happens in a background thread so the plot never
  causes dropped/blocked reads.
- A rolling window (default 5 s @ 250 sps) is kept per channel.
- Every line read is also written to disk immediately, so this
  script doubles as your capture tool -- no separate logging run needed.

Usage:
    python3 emg_live_plot.py --port /dev/ttyACM0 --fs 250 --window 5
"""

import argparse
import csv
import sys
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def parse_args():
    p = argparse.ArgumentParser(description="Live EMG plot from USB CDC-ACM")
    p.add_argument("--port", default="/dev/ttyACM0", help="Serial port")
    p.add_argument("--baud", type=int, default=115200, help="Baud (ignored by CDC-ACM but required by pyserial)")
    p.add_argument("--fs", type=float, default=250.0, help="Sample rate (sps)")
    p.add_argument("--window", type=float, default=5.0, help="Rolling window length (s)")
    p.add_argument("--channels", type=int, default=8, help="Number of EMG channels")
    p.add_argument("--outfile", default=None, help="CSV log path (default: emg_capture_<timestamp>.csv)")
    p.add_argument("--refresh-ms", type=int, default=50, help="Plot refresh interval (ms)")
    return p.parse_args()


class SerialReader(threading.Thread):
    """Reads lines from serial in the background and pushes parsed samples into buffers."""

    def __init__(self, ser, n_channels, maxlen, csv_writer, lock):
        super().__init__(daemon=True)
        self.ser = ser
        self.n_channels = n_channels
        self.lock = lock
        self.csv_writer = csv_writer
        self.sample_idx = deque(maxlen=maxlen)
        self.ch_data = [deque(maxlen=maxlen) for _ in range(n_channels)]
        self.status_bad_count = 0
        self.total_count = 0
        self.stop_flag = threading.Event()

    def run(self):
        while not self.stop_flag.is_set():
            try:
                raw = self.ser.readline()
            except serial.SerialException as e:
                print(f"Serial error: {e}", file=sys.stderr)
                break
            if not raw:
                continue  # timeout, no data
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line or line.startswith("sample"):
                continue  # skip blank lines / header

            parts = line.split(",")
            if len(parts) != self.n_channels + 2:
                continue  # malformed line, drop it

            try:
                idx = int(parts[0])
                status_ok = int(parts[1])
                chans = [int(x) for x in parts[2:]]
            except ValueError:
                continue  # corrupted frame, drop it

            self.total_count += 1
            if not status_ok:
                self.status_bad_count += 1

            with self.lock:
                self.sample_idx.append(idx)
                for i, v in enumerate(chans):
                    self.ch_data[i].append(v)

            if self.csv_writer is not None:
                self.csv_writer.writerow([idx, status_ok] + chans)

    def stop(self):
        self.stop_flag.set()


def main():
    args = parse_args()
    maxlen = int(args.fs * args.window)

    outfile = args.outfile or f"emg_capture_{datetime.now():%Y%m%d_%H%M%S}.csv"
    f = open(outfile, "w", newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow(["sample", "status_ok"] + [f"ch{i+1}" for i in range(args.channels)])

    print(f"Opening {args.port} @ {args.baud} baud")
    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(0.5)  # let the port settle

    lock = threading.Lock()
    reader = SerialReader(ser, args.channels, maxlen, csv_writer, lock)
    reader.start()

    fig, axes = plt.subplots(args.channels, 1, sharex=True, figsize=(10, 13))
    fig.suptitle(f"Live EMG -- {args.port} -- logging to {outfile}")
    lines = []
    range_labels = []
    for i, ax in enumerate(axes):
        (ln,) = ax.plot([], [], lw=0.8)
        ax.set_ylabel(f"ch{i+1}")
        ax.set_xlim(0, args.window)
        # Shows the ACTUAL numeric range each subplot is autoscaled to, since
        # independent autoscaling makes differently-scaled signals look visually
        # identical -- this makes the real amplitude/offset differences explicit.
        lbl = ax.text(
            0.005, 0.85, "", transform=ax.transAxes, fontsize=8,
            family="monospace", va="top",
        )
        range_labels.append(lbl)
        lines.append(ln)
    axes[-1].set_xlabel("Time (s)")

    status_text = fig.text(0.01, 0.005, "", fontsize=9, family="monospace")
    corr_text = fig.text(0.55, 0.005, "", fontsize=8, family="monospace")

    def init():
        for ln in lines:
            ln.set_data([], [])
        for lbl in range_labels:
            lbl.set_text("")
        status_text.set_text("")
        corr_text.set_text("")
        return lines + range_labels + [status_text, corr_text]

    def update(frame):
        with lock:
            idx = list(reader.sample_idx)
            data = [list(d) for d in reader.ch_data]
        if len(idx) < 2:
            return lines + range_labels + [status_text, corr_text]

        t = [(i - idx[-1]) / args.fs + args.window for i in idx]

        for ln, ch in zip(lines, data):
            ln.set_data(t, ch)

        for ax, ch, lbl in zip(axes, data, range_labels):
            if ch:
                lo, hi = min(ch), max(ch)
                pad = max((hi - lo) * 0.1, 1)
                ax.set_ylim(lo - pad, hi + pad)
                lbl.set_text(f"range=[{lo:,}, {hi:,}]  pp={hi - lo:,}")

        # Cross-channel correlation: if every channel tracks ch1 near 1.0,
        # that is common-mode pickup (mains hum, motion, floating reference),
        # not independent EMG -- distinct muscle channels should decorrelate.
        arr = np.array(data)
        if arr.shape[1] >= 2 and arr.shape[0] == args.channels:
            with np.errstate(invalid="ignore"):
                corr = np.corrcoef(arr)[0]  # correlation of each channel vs ch1
            corr_str = " ".join(f"{c:+.2f}" for c in corr)
            corr_text.set_text(f"corr vs ch1: {corr_str}")

        bad_pct = 100.0 * reader.status_bad_count / max(reader.total_count, 1)
        status_text.set_text(
            f"samples: {reader.total_count}  bad-status: {reader.status_bad_count} ({bad_pct:.1f}%)"
        )
        return lines + range_labels + [status_text, corr_text]

    ani = animation.FuncAnimation(
        fig, update, init_func=init, interval=args.refresh_ms, blit=False, cache_frame_data=False
    )

    try:
        plt.tight_layout(rect=[0, 0.02, 1, 0.97])
        plt.show()
    finally:
        reader.stop()
        reader.join(timeout=2)
        ser.close()
        f.close()
        print(f"\nSaved {reader.total_count} samples to {outfile}")


if __name__ == "__main__":
    main()