#!/usr/bin/env python3
"""
Live EMG plotter - Binary ADS1299 USB CDC-ACM packets.
"""

import argparse
import csv
import os
import struct
import sys
import threading
import time
from collections import deque
from datetime import datetime

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import serial

from ads1299 import code_to_volts

# Create captures folder if it doesn't exist
os.makedirs("captures", exist_ok=True)

# ============================================================================
# ADS1299 Binary Packet Decoding
# ============================================================================
ADS1299_NUM_BITS = 24
ADS1299_MAX_CODE = 2 ** (ADS1299_NUM_BITS - 1)
ADS1299_VREF = 5
ADS1299_GAIN = 8

EMG_SYNC_MARKER = b'\xAA\x55'
IMU_SYNC_MARKER = b'\xBB\x66'

SAMPLE_PACKET_FORMAT = '<BB I BB 48s B'
SAMPLE_PACKET_SIZE = struct.calcsize(SAMPLE_PACKET_FORMAT)
IMU_PACKET_FORMAT = '<2sI B B 10s B'
IMU_PACKET_SIZE = struct.calcsize(IMU_PACKET_FORMAT)
NUM_CHANNELS = 16


def extract_24bit_signed(data, offset):
    val = (data[offset] << 16) | (data[offset + 1] << 8) | data[offset + 2]
    if val & 0x800000:
        val -= 0x1000000
    return val


def find_sync_marker(data, start=0, marker=EMG_SYNC_MARKER):
    marker_len = len(marker)
    for i in range(start, len(data) - marker_len + 1):
        if data[i:i + marker_len] == marker:
            return i
    return -1


def parse_sample_packet(data):
    if len(data) < SAMPLE_PACKET_SIZE:
        return None

    packet_bytes = data[:SAMPLE_PACKET_SIZE]
    try:
        sync0, sync1, sample_idx, status1_ok, status2_ok, ch_data_raw, checksum = \
            struct.unpack(SAMPLE_PACKET_FORMAT, packet_bytes)
    except struct.error:
        return None

    if sync0 != 0xAA or sync1 != 0x55:
        return None

    # Reject false-positive sync matches: XOR all bytes except the checksum byte itself
    computed = 0
    for b in packet_bytes[:-1]:
        computed ^= b
    if computed != checksum:
        return None

    channels = [extract_24bit_signed(ch_data_raw, i * 3) for i in range(NUM_CHANNELS)]
    return {
        'sample_idx': sample_idx,
        'status1_ok': bool(status1_ok),
        'status2_ok': bool(status2_ok),
        'channels': channels,
    }


def parse_imu_packet(data):
    if len(data) < IMU_PACKET_SIZE:
        return None

    packet_bytes = data[:IMU_PACKET_SIZE]
    try:
        sync_bytes, sample_idx, sensor_id, data_len, data_raw, checksum = \
            struct.unpack(IMU_PACKET_FORMAT, packet_bytes)
    except struct.error:
        return None

    if sync_bytes != IMU_SYNC_MARKER:
        return None

    computed = 0
    for b in packet_bytes[:-1]:
        computed ^= b
    if computed != checksum:
        return None

    return {
        'sample_idx': sample_idx,
        'sensor_id': sensor_id,
        'data_len': data_len,
        'data': list(data_raw[:data_len]),
    }


# ============================================================================
# Serial Reader
# ============================================================================
class SerialReader(threading.Thread):
    def __init__(self, ser, n_channels, maxlen, csv_writer, lock, imu_channels):
        super().__init__(daemon=True)
        self.ser = ser
        self.n_channels = n_channels
        self.lock = lock
        self.csv_writer = csv_writer
        self.sample_idx = deque(maxlen=maxlen)
        self.ch_data = [deque(maxlen=maxlen) for _ in range(n_channels)]
        self.imu_sample_idx = deque(maxlen=maxlen)
        self.imu_ch_data = [deque(maxlen=maxlen) for _ in range(imu_channels)]
        self.status1_bad_count = 0
        self.status2_bad_count = 0
        self.total_count = 0
        self.imu_total_count = 0
        self.stop_flag = threading.Event()
        self.buffer = b''
        self.error_count = 0

    def run(self):
        print("Binary reader started...", file=sys.stderr)
        while not self.stop_flag.is_set():
            try:
                chunk = self.ser.read(self.ser.in_waiting or 128)
            except serial.SerialException as e:
                print(f"Serial error: {e}", file=sys.stderr)
                break

            if not chunk:
                continue

            self.buffer += chunk

            processed = 0
            while len(self.buffer) >= 2:
                emg_idx = find_sync_marker(self.buffer, marker=EMG_SYNC_MARKER)
                imu_idx = find_sync_marker(self.buffer, marker=IMU_SYNC_MARKER)

                if emg_idx == -1 and imu_idx == -1:
                    break

                if emg_idx == -1:
                    sync_idx, packet_size, packet_type = imu_idx, IMU_PACKET_SIZE, 'imu'
                elif imu_idx == -1:
                    sync_idx, packet_size, packet_type = emg_idx, SAMPLE_PACKET_SIZE, 'emg'
                elif emg_idx < imu_idx:
                    sync_idx, packet_size, packet_type = emg_idx, SAMPLE_PACKET_SIZE, 'emg'
                else:
                    sync_idx, packet_size, packet_type = imu_idx, IMU_PACKET_SIZE, 'imu'

                if sync_idx > 0:
                    self.buffer = self.buffer[sync_idx:]

                if len(self.buffer) < packet_size:
                    break

                if packet_type == 'emg':
                    packet = parse_sample_packet(self.buffer[:packet_size])
                    if packet:
                        self.total_count += 1
                        if not packet['status1_ok']:
                            self.status1_bad_count += 1
                        if not packet['status2_ok']:
                            self.status2_bad_count += 1

                        with self.lock:
                            self.sample_idx.append(packet['sample_idx'])
                            for i, code in enumerate(packet['channels']):
                                self.ch_data[i].append(code)

                        if self.csv_writer is not None:
                            self.csv_writer.writerow(
                                [packet['sample_idx'], int(packet['status1_ok']), int(packet['status2_ok'])] +
                                packet['channels']
                            )

                        self.buffer = self.buffer[packet_size:]
                        processed += 1
                    else:
                        self.buffer = self.buffer[1:]
                        self.error_count += 1
                else:
                    packet = parse_imu_packet(self.buffer[:packet_size])
                    if packet:
                        self.imu_total_count += 1
                        with self.lock:
                            self.imu_sample_idx.append(packet['sample_idx'])
                            for i, value in enumerate(packet['data']):
                                if i < len(self.imu_ch_data):
                                    self.imu_ch_data[i].append(value)

                        self.buffer = self.buffer[packet_size:]
                        processed += 1
                    else:
                        self.buffer = self.buffer[1:]
                        self.error_count += 1

            if processed == 0 and len(self.buffer) > SAMPLE_PACKET_SIZE * 4:
                self.buffer = self.buffer[-SAMPLE_PACKET_SIZE * 2:]  # prevent buffer bloat

    def stop(self):
        self.stop_flag.set()


# ============================================================================
# Main + Plotting
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
    p.add_argument("--imu-channels", type=int, default=8, help="Number of IMU data traces to plot")

    p.add_argument("--gain", type=float, default=ADS1299_GAIN, help="ADS1299 PGA gain")
    p.add_argument("--vref", type=float, default=ADS1299_VREF, help="ADS1299 reference voltage")
    p.add_argument("--unit", choices=["v", "mv", "uv"], default="uv", help="Display unit")
    p.add_argument("--ylim", nargs=2, type=float, default=None, metavar=("YMIN", "YMAX"),
                   help="Fixed y-axis limits")

    return p.parse_args()


def convert_units(data_codes, vref, gain, unit):
    """Convert ADC codes to selected unit."""
    volts = code_to_volts(data_codes, vref=vref, gain=gain)

    if unit == "v":
        return volts, "V"
    elif unit == "mv":
        return volts * 1e3, "mV"
    else:  # uv
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
    reader = SerialReader(ser, args.channels, maxlen, csv_writer, lock, args.imu_channels)
    reader.start()

    n_rows = max(args.channels, args.imu_channels)
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(n_rows, 2, width_ratios=[1, 1], wspace=0.18, hspace=0.05)
    emg_axes = [fig.add_subplot(gs[i, 0]) for i in range(args.channels)]
    imu_axes = [fig.add_subplot(gs[i, 1]) for i in range(args.imu_channels)]
    fig.suptitle(f"Live EMG + IMU -- {args.port} -- gain={args.gain:g}, {args.unit} -- logging to {outfile}")

    emg_lines = []
    emg_range_labels = []
    for i, ax in enumerate(emg_axes):
        ln, = ax.plot([], [], lw=0.8, color='tab:blue')
        ax.set_ylabel(f"ch{i+1}")
        ax.set_xlim(0, args.window)
        if args.ylim is not None:
            ax.set_ylim(args.ylim[0], args.ylim[1])
        lbl = ax.text(0.005, 0.85, "", transform=ax.transAxes, fontsize=7, family="monospace", va="top")
        emg_lines.append(ln)
        emg_range_labels.append(lbl)

    imu_lines = []
    imu_range_labels = []
    for i, ax in enumerate(imu_axes):
        ln, = ax.plot([], [], lw=0.8, color='tab:orange')
        ax.set_ylabel(f"imu{i+1}")
        ax.set_xlim(0, args.window)
        lbl = ax.text(0.005, 0.85, "", transform=ax.transAxes, fontsize=7, family="monospace", va="top")
        imu_lines.append(ln)
        imu_range_labels.append(lbl)

    emg_axes[-1].set_xlabel("Time (s)")
    imu_axes[-1].set_xlabel("Time (s)")
    status_text = fig.text(0.01, 0.005, "", fontsize=9, family="monospace")

    def update(frame):
        with lock:
            emg_raw = [list(d) for d in reader.ch_data]
            imu_raw = [list(d) for d in reader.imu_ch_data]

        if len(emg_raw[0]) < 2:
            return emg_lines + emg_range_labels + imu_lines + imu_range_labels + [status_text]

        with lock:
            sample_idx = list(reader.sample_idx)
            emg_raw = [list(d) for d in reader.ch_data]
            imu_sample_idx = list(reader.imu_sample_idx)
            imu_raw = [list(d) for d in reader.imu_ch_data]

            n_emg = len(emg_raw[0])
            t_emg = (np.asarray(sample_idx[-n_emg:], dtype=np.float64) - sample_idx[-n_emg]) / args.fs
            n_imu = len(imu_raw[0]) if imu_raw else 0
            t_imu = (np.asarray(imu_sample_idx[-n_imu:], dtype=np.float64) - imu_sample_idx[-n_imu]) / args.fs if n_imu else np.array([], dtype=np.float64)

        emg_data_array = np.asarray(emg_raw, dtype=np.float64)
        emg_converted, unit_label = convert_units(emg_data_array, vref=args.vref, gain=args.gain, unit=args.unit)

        for ln, ch in zip(emg_lines, emg_converted):
            ln.set_data(t_emg, ch)

        for ax, ch, lbl in zip(emg_axes, emg_converted, emg_range_labels):
            if len(ch) > 1:
                lo, hi = np.min(ch), np.max(ch)
                if args.ylim is None:
                    pad = max((hi - lo) * 0.1, 1)
                    ax.set_ylim(lo - pad, hi + pad)
                lbl.set_text(f"{lo:.1f}..{hi:.1f} {unit_label}")

        if imu_raw:
            imu_data_array = np.asarray(imu_raw, dtype=np.float64)
            for ln, ch in zip(imu_lines, imu_data_array):
                ln.set_data(t_imu, ch)

            for ax, ch, lbl in zip(imu_axes, imu_data_array, imu_range_labels):
                if len(ch) > 1:
                    lo, hi = np.min(ch), np.max(ch)
                    pad = max((hi - lo) * 0.1, 1)
                    ax.set_ylim(lo - pad, hi + pad)
                    lbl.set_text(f"{lo:.1f}..{hi:.1f}")

        bad1_pct = 100.0 * reader.status1_bad_count / max(reader.total_count, 1)
        bad2_pct = 100.0 * reader.status2_bad_count / max(reader.total_count, 1)
        status_text.set_text(
            f"emg samples: {reader.total_count}  bad1-status: {reader.status1_bad_count} ({bad1_pct:.1f}%) bad2-status: {reader.status2_bad_count} ({bad2_pct:.1f}%)  imu samples: {reader.imu_total_count}"
        )
        return emg_lines + emg_range_labels + imu_lines + imu_range_labels + [status_text]

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