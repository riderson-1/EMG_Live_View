#!/usr/bin/env python3
"""
Live EMG plotter - Binary ADS1299 USB CDC-ACM packets.
"""

import argparse
import csv
import sys
import threading
import time
from collections import deque
from datetime import datetime
import struct

import serial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

from ads1299 import code_to_volts

# Create captures folder if it doesn't exist
os.makedirs("captures", exist_ok=True)

# ============================================================================
# ADS1299 Binary Packet Decoding (unchanged)
# ============================================================================
ADS1299_NUM_BITS = 24
ADS1299_MAX_CODE = 2 ** (ADS1299_NUM_BITS - 1)
ADS1299_VREF = 4.5
ADS1299_GAIN = 8

SAMPLE_PACKET_FORMAT = '!BB I BB 48s B'
SAMPLE_PACKET_SIZE = struct.calcsize(SAMPLE_PACKET_FORMAT)
NUM_CHANNELS = 16


def extract_24bit_signed(data, offset):
    val = (data[offset] << 16) | (data[offset + 1] << 8) | data[offset + 2]
    if val & 0x800000:
        val -= 0x1000000
    return val


def find_sync_marker(data, start=0):
    for i in range(start, len(data) - 1):
        if data[i] == 0xAA and data[i + 1] == 0x55:
            return i
    return -1


def parse_sample_packet(data):
    if len(data) < SAMPLE_PACKET_SIZE:
        return None
    try:
        sync0, sync1, sample_idx, status1_ok, status2_ok, ch_data_raw, checksum = \
            struct.unpack(SAMPLE_PACKET_FORMAT, data[:SAMPLE_PACKET_SIZE])
    except struct.error:
        return None

    if sync0 != 0xAA or sync1 != 0x55:
        return None

    channels = [extract_24bit_signed(ch_data_raw, i * 3) for i in range(NUM_CHANNELS)]
    return {
        'sample_idx': sample_idx,
        'status1_ok': status1_ok,
        'status2_ok': status2_ok,
        'channels': channels,
    }


# ============================================================================
# Serial Reader
# ============================================================================
class SerialReader(threading.Thread):
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
        self.buffer = b''

    def run(self):
        while not self.stop_flag.is_set():
            try:
                chunk = self.ser.read(self.ser.in_waiting or 64)
            except serial.SerialException as e:
                print(f"Serial error: {e}", file=sys.stderr)
                break
            if not chunk:
                continue

            self.buffer += chunk

            while len(self.buffer) >= SAMPLE_PACKET_SIZE:
                sync_idx = find_sync_marker(self.buffer)
                if sync_idx == -1:
                    break
                if sync_idx > 0:
                    self.buffer = self.buffer[sync_idx:]

                if len(self.buffer) < SAMPLE_PACKET_SIZE:
                    break

                packet = parse_sample_packet(self.buffer[:SAMPLE_PACKET_SIZE])
                if packet:
                    self.total_count += 1
                    if not packet['status1_ok'] or not packet['status2_ok']:
                        self.status_bad_count += 1

                    with self.lock:
                        self.sample_idx.append(self.total_count)          # Use simple counter for plotting
                        for i, code in enumerate(packet['channels']):
                            self.ch_data[i].append(code)

                    if self.csv_writer is not None:
                        self.csv_writer.writerow(
                            [packet['sample_idx'], packet['status1_ok'], packet['status2_ok']] + 
                            packet['channels']
                        )

                    self.buffer = self.buffer[SAMPLE_PACKET_SIZE:]
                else:
                    self.buffer = self.buffer[1:]   # resync

    def stop(self):
        self.stop_flag.set()


# ============================================================================
# Main + Plotting (Main fix here)
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Live EMG plot from ADS1299 binary packets")
    p.add_argument("--port", default="/dev/ttyACM0", help="Serial port")
    p.add_argument("--baud", type=int, default=115200, help="Baud rate")
    p.add_argument("--fs", type=float, default=250.0, help="Sample rate in Hz")
    p.add_argument("--window", type=float, default=5.0, help="Rolling window length in seconds")
    p.add_argument("--channels", type=int, default=16, help="Number of EMG channels")
    p.add_argument("--outfile", default=None, help="CSV log path")
    p.add_argument("--refresh-ms", type=int, default=50, help="Plot refresh interval in ms")

    p.add_argument("--gain", type=float, default=8.0, help="ADS1299 PGA gain")
    p.add_argument("--vref", type=float, default=4.5, help="ADS1299 reference voltage")
    p.add_argument("--unit", choices=["v", "mv", "uv"], default="uv", help="Display unit")
    p.add_argument("--ylim", nargs=2, type=float, default=None, metavar=("YMIN", "YMAX"),
                   help="Fixed y-axis limits")

    return p.parse_args()


def convert_units(data_codes, vref, gain, unit):
    volts = code_to_volts(data_codes, vref=vref, gain=gain)
    if unit == "v":
        return volts, "V"
    elif unit == "mv":
        return volts * 1e3, "mV"
    else:
        return volts * 1e6, "µV"


def main():
    args = parse_args()
    maxlen = int(args.fs * args.window)

    outfile = args.outfile or os.path.join("captures", f"emg_capture_{datetime.now():%Y%m%d_%H%M%S}.csv")
    f = open(outfile, "w", newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow(["sample", "status1_ok", "status2_ok"] + [f"ch{i+1}" for i in range(args.channels)])

    print(f"Opening {args.port} @ {args.baud} baud")
    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(0.5)

    lock = threading.Lock()
    reader = SerialReader(ser, args.channels, maxlen, csv_writer, lock)
    reader.start()

    fig, axes = plt.subplots(args.channels, 1, sharex=True, figsize=(10, 12))
    fig.suptitle(f"Live EMG -- {args.port} -- gain={args.gain:g}, {args.unit} -- logging to {outfile}")

    lines = []
    range_labels = []
    for i, ax in enumerate(axes):
        ln, = ax.plot([], [], lw=0.8)
        ax.set_ylabel(f"ch{i+1}")
        ax.set_xlim(0, args.window)
        if args.ylim is not None:
            ax.set_ylim(args.ylim[0], args.ylim[1])
        lbl = ax.text(0.005, 0.85, "", transform=ax.transAxes, fontsize=7, family="monospace", va="top")
        lines.append(ln)
        range_labels.append(lbl)

    axes[-1].set_xlabel("Time (s)")
    status_text = fig.text(0.01, 0.005, "", fontsize=9, family="monospace")

    def update(frame):
        with lock:
            data_raw = [list(d) for d in reader.ch_data]

        if len(data_raw[0]) < 2:
            return lines + range_labels + [status_text]

        # === FIXED TIME CALCULATION ===
        n = len(data_raw[0])
        t = np.linspace(0, args.window, n)                     # Simple, stable rolling time axis

        data_array = np.asarray(data_raw, dtype=np.float64)
        data_converted, unit_label = convert_units(data_array, vref=args.vref, gain=args.gain, unit=args.unit)

        for ln, ch in zip(lines, data_converted):
            ln.set_data(t, ch)

        for ax, ch, lbl in zip(axes, data_converted, range_labels):
            if len(ch) > 1:
                lo, hi = np.min(ch), np.max(ch)
                if args.ylim is None:
                    pad = max((hi - lo) * 0.1, 1)
                    ax.set_ylim(lo - pad, hi + pad)
                lbl.set_text(f"{lo:.1f}..{hi:.1f} {unit_label}")

        bad_pct = 100.0 * reader.status_bad_count / max(reader.total_count, 1)
        status_text.set_text(
            f"samples: {reader.total_count}  bad-status: {reader.status_bad_count} ({bad_pct:.1f}%)"
        )
        return lines + range_labels + [status_text]

    ani = animation.FuncAnimation(
        fig, update, interval=args.refresh_ms, blit=False, cache_frame_data=False
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