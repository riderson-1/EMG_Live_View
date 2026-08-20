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
from scipy.spatial.transform import Rotation as R

# Create captures folder if it doesn't exist
os.makedirs("captures", exist_ok=True)

# ============================================================================
# ADS1299 Binary Packet Decoding
# ============================================================================
ADS1299_NUM_BITS = 24
ADS1299_MAX_CODE = 2 ** (ADS1299_NUM_BITS - 1)
ADS1299_VREF = 4.5
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

    # The driver sends two different BHI360 virtual-sensor payloads:
    #   sensor 37 (GAMERV): int16 x,y,z,w + uint16 accuracy = 10 bytes
    #   sensor  4 (ACC):    int16 x,y,z = 6 bytes
    # Quaternion components are Q14; corrected acceleration is 1/16 m/s^2/LSB.
    if sensor_id == 37 and data_len == 10:
        x, y, z, w, accuracy = struct.unpack('<4hH', data_raw[:10])
        quat = np.asarray([x, y, z, w], dtype=np.float64) / 16384.0
        norm = np.linalg.norm(quat)
        if norm == 0:
            return None
        quat /= norm
        euler = R.from_quat(quat).as_euler('xyz', degrees=True)
        return {
            'sample_idx': sample_idx,
            'sensor_id': sensor_id,
            'kind': 'quaternion',
            'quat': quat.tolist(),
            'accuracy': accuracy / 16384.0,
            'euler': euler.tolist(),
        }

    if sensor_id == 4 and data_len == 6:
        accel_raw = struct.unpack('<3h', data_raw[:6])
        return {
            'sample_idx': sample_idx,
            'sensor_id': sensor_id,
            'kind': 'acceleration',
            'accel': [value / 16.0 for value in accel_raw],
        }

    return None


# ============================================================================
# Serial Reader
# ============================================================================
class SerialReader(threading.Thread):
    def __init__(self, ser, n_channels, maxlen, csv_writer, lock, imu_channels,
                 process_emg=True, process_imu=True):
        super().__init__(daemon=True)
        self.ser = ser
        self.n_channels = n_channels
        self.lock = lock
        self.csv_writer = csv_writer
        self.process_emg = process_emg
        self.process_imu = process_imu
        self.sample_idx = deque(maxlen=maxlen)
        self.ch_data = [deque(maxlen=maxlen) for _ in range(n_channels)]
        self.imu_euler_sample_idx = deque(maxlen=maxlen)
        self.imu_accel_sample_idx = deque(maxlen=maxlen)
        # IMU data: quaternion (4) + linear acceleration (3) + optional extra
        self.imu_quat_data = [deque(maxlen=maxlen) for _ in range(4)]  # qx, qy, qz, qw
        self.imu_accel_data = [deque(maxlen=maxlen) for _ in range(3)]  # ax, ay, az
        self.imu_euler_data = [deque(maxlen=maxlen) for _ in range(3)]  # roll, pitch, yaw
        self.imu_latest_quat = [None] * 4
        self.imu_latest_accel = [None] * 3
        self.imu_latest_euler = [None] * 3
        self.status1_bad_count = 0
        self.status2_bad_count = 0
        self.emg_total_count = 0
        self.imu_total_count = 0
        self.stop_flag = threading.Event()
        self.buffer = b''
        self.error_count = 0

    def run(self):
        print("Binary reader started...", file=sys.stderr)
        while not self.stop_flag.is_set():
            try:
                # Read bounded chunks so a large host-side USB backlog does not
                # monopolize the GUI thread's companion reader thread.
                chunk = self.ser.read(min(self.ser.in_waiting or 128, 512))
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
                        if self.process_emg:
                            self.emg_total_count += 1
                            if not packet['status1_ok']:
                                self.status1_bad_count += 1
                            if not packet['status2_ok']:
                                self.status2_bad_count += 1

                            with self.lock:
                                self.sample_idx.append(packet['sample_idx'])
                                for i, code in enumerate(packet['channels']):
                                    self.ch_data[i].append(code)

                        if self.process_emg and self.csv_writer is not None:
                            # Always write IMU columns. EMG columns are written
                            # as zeros when the EMG stream is not active.
                            euler_values = [v if v is not None else '' for v in self.imu_latest_euler]
                            accel_values = [v if v is not None else '' for v in self.imu_latest_accel]
                            emg_channels = (
                                packet['channels']
                                if self.process_emg else
                                [''] * self.n_channels
                            )
                            self.csv_writer.writerow(
                                [packet['sample_idx'], int(packet['status1_ok']), int(packet['status2_ok'])] +
                                emg_channels +
                                euler_values +
                                accel_values
                            )

                        self.buffer = self.buffer[packet_size:]
                        processed += 1
                    else:
                        self.buffer = self.buffer[1:]
                        self.error_count += 1
                else:
                    packet = parse_imu_packet(self.buffer[:packet_size])
                    if packet:
                        if self.process_imu:
                            self.imu_total_count += 1
                        with self.lock:
                            if self.process_imu and packet['kind'] == 'quaternion':
                                self.imu_euler_sample_idx.append(packet['sample_idx'])
                                for i, value in enumerate(packet['quat']):
                                    self.imu_quat_data[i].append(value)
                                    self.imu_latest_quat[i] = value
                                for i, value in enumerate(packet['euler']):
                                    self.imu_euler_data[i].append(value)
                                    self.imu_latest_euler[i] = value
                            elif self.process_imu and packet['kind'] == 'acceleration':
                                self.imu_accel_sample_idx.append(packet['sample_idx'])
                                for i, value in enumerate(packet['accel']):
                                    self.imu_accel_data[i].append(value)
                                    self.imu_latest_accel[i] = value

                        # Write to CSV in IMU-only mode (no EMG packets to trigger writes)
                        if self.csv_writer is not None and not self.process_emg:
                            euler_values = [v if v is not None else '' for v in self.imu_latest_euler]
                            accel_values = [v if v is not None else '' for v in self.imu_latest_accel]
                            emg_channels = [''] * self.n_channels
                            self.csv_writer.writerow(
                                [packet['sample_idx'], '', ''] +
                                emg_channels +
                                euler_values +
                                accel_values
                            )

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
    p.add_argument("--fs", type=float, default=1000.0, help="EMG sample rate in Hz")
    p.add_argument("--imu-fs", type=float, default=200.0, help="IMU sample rate in Hz (quaternion + acceleration combined)")
    p.add_argument("--window", type=float, default=5.0, help="Rolling window length in seconds")
    p.add_argument("--channels", type=int, default=16, help="Number of EMG channels")
    p.add_argument("--outfile", default=None, help="CSV log path")
    p.add_argument("--refresh-ms", type=int, default=100, help="Plot refresh interval in ms")
    p.add_argument("--max-plot-points", type=int, default=1200,
                   help="Maximum points per trace to draw per refresh")
    p.add_argument("--imu-angle-ylim", nargs=2, type=float, default=(-180.0, 180.0),
                   metavar=("MIN", "MAX"),
                   help="Fixed Euler-angle axis limits in degrees")
    p.add_argument("--imu-accel-ylim", nargs=2, type=float, default=(-20.0, 20.0),
                   metavar=("MIN", "MAX"),
                   help="Fixed linear-acceleration axis limits in m/s^2")
    p.add_argument("--imu-channels", type=int, default=8, help="Number of IMU data traces to plot")
    p.add_argument("--mode", choices=("both", "emg", "imu"), default="both",
                   help="What to display: both sensors, EMG only, or IMU only")

    p.add_argument("--gain", type=float, default=ADS1299_GAIN, help="ADS1299 PGA gain")
    p.add_argument("--vref", type=float, default=ADS1299_VREF, help="ADS1299 reference voltage")
    p.add_argument("--unit", choices=["v", "mv", "uv"], default="mv", help="Display unit")
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
    show_emg = args.mode in ("both", "emg")
    show_imu = args.mode in ("both", "imu")

    outfile = args.outfile or os.path.join("captures", f"sokosti_capture_{datetime.now():%Y%m%d_%H%M%S}.csv")
    f = open(outfile, "w", newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow(
        ["sample", "status1_ok", "status2_ok"] +
        [f"ch{i+1}" for i in range(args.channels)] +
        ["roll", "pitch", "yaw", "accel_x", "accel_y", "accel_z"]
    )

    print(f"Opening {args.port} @ {args.baud} baud")
    ser = serial.Serial(args.port, args.baud, timeout=0.01)
    time.sleep(0.5)

    lock = threading.Lock()
    # Discard bytes that accumulated before this run. This prevents an old
    # host-side USB buffer from appearing as a several-second live delay.
    ser.reset_input_buffer()
    reader = SerialReader(
        ser, args.channels, maxlen, csv_writer, lock, args.imu_channels,
        process_emg=show_emg, process_imu=show_imu,
    )
    reader.start()
    start_time = time.time()

    # IMU plots: 3 for Euler angles + 3 for linear acceleration
    imu_plot_channels = 6  # roll, pitch, yaw, accel_x, accel_y, accel_z
    n_rows = max(args.channels if show_emg else 0, imu_plot_channels if show_imu else 0)
    fig = plt.figure(figsize=(16, 12))
    n_columns = int(show_emg) + int(show_imu)
    gs = fig.add_gridspec(n_rows, n_columns, width_ratios=[1] * n_columns,
                          wspace=0.18, hspace=0.05)
    emg_column = 0 if show_emg else None
    imu_column = int(show_emg) if show_imu else None
    emg_axes = ([fig.add_subplot(gs[i, emg_column]) for i in range(args.channels)]
                if show_emg else [])
    imu_axes = ([fig.add_subplot(gs[i, imu_column]) for i in range(imu_plot_channels)]
                if show_imu else [])
    fig.suptitle(f"Live {args.mode.upper()} -- {args.port} -- gain={args.gain:g}, {args.unit} -- logging to {outfile}")

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
    imu_labels = ['roll', 'pitch', 'yaw', 'accel_x', 'accel_y', 'accel_z']
    for i, ax in enumerate(imu_axes):
        ln, = ax.plot([], [], lw=0.8, color='tab:orange')
        ax.set_ylabel(imu_labels[i])
        ax.set_xlim(0, args.window)
        if i < 3:
            ax.set_ylim(args.imu_angle_ylim[0], args.imu_angle_ylim[1])
        else:
            ax.set_ylim(args.imu_accel_ylim[0], args.imu_accel_ylim[1])
        lbl = ax.text(0.005, 0.85, "", transform=ax.transAxes, fontsize=7, family="monospace", va="top")
        imu_lines.append(ln)
        imu_range_labels.append(lbl)

    if emg_axes:
        emg_axes[-1].set_xlabel("Time (s)")
    if imu_axes:
        imu_axes[-1].set_xlabel("Time (s)")
    status_text = fig.text(0.01, 0.005, "", fontsize=9, family="monospace")

    def update(frame):
        # Take one atomic snapshot. The reader thread continues receiving data,
        # but it cannot change these lists while the snapshot is being made.
        with lock:
            sample_idx = list(reader.sample_idx)
            emg_raw = [list(d) for d in reader.ch_data]
            imu_euler_sample_idx = list(reader.imu_euler_sample_idx)
            euler_raw = [list(d) for d in reader.imu_euler_data]
            imu_accel_sample_idx = list(reader.imu_accel_sample_idx)
            accel_raw = [list(d) for d in reader.imu_accel_data]
            emg_total_count = reader.emg_total_count
            imu_total_count = reader.imu_total_count
            status1_bad_count = reader.status1_bad_count
            status2_bad_count = reader.status2_bad_count

        if show_emg and (not sample_idx or len(emg_raw[0]) < 2):
            return emg_lines + emg_range_labels + imu_lines + imu_range_labels + [status_text]

        def trace_data(indices, values, fs):
            """Return matching, bounded numpy arrays for one trace group."""
            n = min(len(indices), len(values))
            if n < 1:
                return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
            indices = indices[-n:]
            values = values[-n:]
            if n > args.max_plot_points:
                step = int(np.ceil(n / args.max_plot_points))
                indices = indices[::step]
                values = values[::step]
            # Keep the newest sample at the right edge of the live window.
            t = ((np.asarray(indices, dtype=np.float64) - indices[-1]) / fs
                 + args.window)
            return t, np.asarray(values, dtype=np.float64)

        t_emg = np.array([], dtype=np.float64)
        emg_data = []
        if show_emg and sample_idx and emg_raw and emg_raw[0]:
            t_emg, _ = trace_data(sample_idx, emg_raw[0], args.fs)
            emg_data = [trace_data(sample_idx, values, args.fs)[1] for values in emg_raw]
        t_euler = np.array([], dtype=np.float64)
        euler_data = []
        if show_imu and euler_raw and euler_raw[0]:
            t_euler, _ = trace_data(imu_euler_sample_idx, euler_raw[0], args.imu_fs)
            euler_data = [trace_data(imu_euler_sample_idx, values, args.imu_fs)[1] for values in euler_raw]
        t_accel = np.array([], dtype=np.float64)
        accel_data = []
        if show_imu and accel_raw and accel_raw[0]:
            t_accel, _ = trace_data(imu_accel_sample_idx, accel_raw[0], args.imu_fs)
            accel_data = [trace_data(imu_accel_sample_idx, values, args.imu_fs)[1] for values in accel_raw]

        if show_emg and emg_data:
            emg_data_array = np.asarray(emg_data, dtype=np.float64)
            emg_converted, unit_label = convert_units(emg_data_array, vref=args.vref, gain=args.gain, unit=args.unit)

            for ln, ch in zip(emg_lines, emg_converted):
                ln.set_data(t_emg, ch)

            for ax, ch, lbl in zip(emg_axes, emg_converted, emg_range_labels):
                if len(ch) > 1 and frame % 5 == 0:
                    lo, hi = np.min(ch), np.max(ch)
                    if args.ylim is None:
                        pad = max((hi - lo) * 0.1, 1)
                        ax.set_ylim(lo - pad, hi + pad)
                    lbl.set_text(f"{lo:.1f}..{hi:.1f} {unit_label}")

        # Plot IMU data: Euler angles (0-2) and linear acceleration (3-5)
        # Euler angles (roll, pitch, yaw). Their limits are fixed above to
        # avoid expensive and visually distracting autoscaling.
        for i in range(3):
            if not show_imu:
                break
            if i < len(euler_data) and len(euler_data[i]) > 0:
                imu_lines[i].set_data(t_euler[-len(euler_data[i]):], euler_data[i])
                if len(euler_data[i]) > 1 and frame % 5 == 0:
                    lo, hi = np.min(euler_data[i]), np.max(euler_data[i])
                    pad = max((hi - lo) * 0.1, 1.0)
                    imu_axes[i].set_ylim(lo - pad, hi + pad)
                    imu_range_labels[i].set_text(f"{lo:.1f}..{hi:.1f} °")

        # Linear acceleration - use its own sample timeline.
        if show_imu:
            has_accel_data = any(len(d) > 0 for d in accel_data)
            if has_accel_data:
                for i in range(3):
                    if i < len(accel_data) and len(accel_data[i]) > 0:
                        imu_lines[i + 3].set_data(t_accel[-len(accel_data[i]):], accel_data[i])
                        if len(accel_data[i]) > 1 and frame % 5 == 0:
                            lo, hi = np.min(accel_data[i]), np.max(accel_data[i])
                            pad = max((hi - lo) * 0.1, 0.1)
                            imu_axes[i + 3].set_ylim(lo - pad, hi + pad)
                            imu_range_labels[i + 3].set_text(f"{lo:.2f}..{hi:.2f} m/s²")
            else:
                # No acceleration data available
                for i in range(3):
                    imu_lines[i + 3].set_data([], [])
                    imu_range_labels[i + 3].set_text("no data")

        bad1_pct = 100.0 * status1_bad_count / max(emg_total_count, 1)
        bad2_pct = 100.0 * status2_bad_count / max(emg_total_count, 1)
        status_text.set_text(
            f"emg samples: {emg_total_count}  bad1-status: {status1_bad_count} ({bad1_pct:.1f}%) bad2-status: {status2_bad_count} ({bad2_pct:.1f}%)  imu samples: {imu_total_count}"
        )

        if frame % 100 == 0:
            elapsed = time.time() - start_time
            if elapsed > 0:
                print(f"EMG rate: {emg_total_count / elapsed:.1f} Hz, "
                      f"IMU rate: {imu_total_count / elapsed:.1f} Hz")

        return emg_lines + emg_range_labels + imu_lines + imu_range_labels + [status_text]

    ani = animation.FuncAnimation(
        fig, update, interval=args.refresh_ms, blit=False, cache_frame_data=False
    )

    try:
        plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.05)
        plt.show()
    finally:
        reader.stop()
        reader.join(timeout=2)
        ser.close()
        f.close()
        print(f"\nSaved {reader.emg_total_count} emg samples and {reader.imu_total_count} imu samples to {outfile}")


if __name__ == "__main__":
    main()