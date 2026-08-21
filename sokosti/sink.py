"""
sink.py - Thread-safe packet storage and CSV logging (PacketSink).

A PacketSink owns the rolling deques, the latest-value caches, the
status/count statistics, and the CSV writer. Both transports (serial and
BLE) call ``add_emg`` / ``add_imu`` from their own threads; the plotter
calls ``snapshot()`` from the GUI thread. A single lock keeps all of that
consistent.

This is the code that used to live inside SerialReader.
"""

import threading
from collections import deque


class PacketSink:
    """Thread-safe accumulator for EMG/IMU packets, with optional CSV logging."""

    def __init__(self, n_channels, maxlen, csv_writer=None,
                 process_emg=True, process_imu=True):
        self.n_channels = n_channels
        self.csv_writer = csv_writer
        self.process_emg = process_emg
        self.process_imu = process_imu

        self.lock = threading.Lock()

        self.sample_idx = deque(maxlen=maxlen)
        self.ch_data = [deque(maxlen=maxlen) for _ in range(n_channels)]

        self.imu_euler_sample_idx = deque(maxlen=maxlen)
        self.imu_accel_sample_idx = deque(maxlen=maxlen)
        # IMU data: quaternion (4) + linear acceleration (3)
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

    # ------------------------------------------------------------------
    # Ingestion (called from the transport thread)
    # ------------------------------------------------------------------
    def add_emg(self, packet):
        """Record one EMG sample packet and write a CSV row."""
        if self.process_emg:
            self.emg_total_count += 1
            if not packet["status1_ok"]:
                self.status1_bad_count += 1
            if not packet["status2_ok"]:
                self.status2_bad_count += 1

            with self.lock:
                self.sample_idx.append(packet["sample_idx"])
                for i, code in enumerate(packet["channels"]):
                    self.ch_data[i].append(code)

        if self.process_emg and self.csv_writer is not None:
            # Always write IMU columns. EMG columns are written as zeros
            # when the EMG stream is not active.
            euler_values = [v if v is not None else "" for v in self.imu_latest_euler]
            accel_values = [v if v is not None else "" for v in self.imu_latest_accel]
            emg_channels = (
                packet["channels"]
                if self.process_emg else
                [""] * self.n_channels
            )
            self.csv_writer.writerow(
                [packet["sample_idx"], int(packet["status1_ok"]), int(packet["status2_ok"])]
                + emg_channels
                + euler_values
                + accel_values
            )

    def add_imu(self, packet):
        """Record one parsed IMU packet and update the CSV when in IMU-only mode."""
        if self.process_imu:
            self.imu_total_count += 1

        with self.lock:
            if self.process_imu and packet["kind"] == "quaternion":
                self.imu_euler_sample_idx.append(packet["sample_idx"])
                for i, value in enumerate(packet["quat"]):
                    self.imu_quat_data[i].append(value)
                    self.imu_latest_quat[i] = value
                for i, value in enumerate(packet["euler"]):
                    self.imu_euler_data[i].append(value)
                    self.imu_latest_euler[i] = value
            elif self.process_imu and packet["kind"] == "acceleration":
                self.imu_accel_sample_idx.append(packet["sample_idx"])
                for i, value in enumerate(packet["accel"]):
                    self.imu_accel_data[i].append(value)
                    self.imu_latest_accel[i] = value

        # Write to CSV in IMU-only mode (no EMG packets to trigger writes).
        if self.csv_writer is not None and not self.process_emg:
            euler_values = [v if v is not None else "" for v in self.imu_latest_euler]
            accel_values = [v if v is not None else "" for v in self.imu_latest_accel]
            emg_channels = [""] * self.n_channels
            self.csv_writer.writerow(
                [packet["sample_idx"], "", ""]
                + emg_channels
                + euler_values
                + accel_values
            )

    # ------------------------------------------------------------------
    # Consumption (called from the GUI thread)
    # ------------------------------------------------------------------
    def snapshot(self):
        """Return an atomic copy of all plot-relevant state."""
        with self.lock:
            return {
                "sample_idx": list(self.sample_idx),
                "emg_raw": [list(d) for d in self.ch_data],
                "imu_euler_sample_idx": list(self.imu_euler_sample_idx),
                "euler_raw": [list(d) for d in self.imu_euler_data],
                "imu_accel_sample_idx": list(self.imu_accel_sample_idx),
                "accel_raw": [list(d) for d in self.imu_accel_data],
                "emg_total_count": self.emg_total_count,
                "imu_total_count": self.imu_total_count,
                "status1_bad_count": self.status1_bad_count,
                "status2_bad_count": self.status2_bad_count,
            }