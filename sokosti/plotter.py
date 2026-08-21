"""
plotter.py - Live matplotlib plot (LivePlotter).

This is the plotting code from emg_live_plot.py, unchanged in behavior.
It builds the figure/axes once, then a FuncAnimation calls ``update`` on
each refresh. The only difference from the original is that it reads its
data from a PacketSink.snapshot() instead of reaching into SerialReader
internals, so the same plotter works for USB and BLE alike.
"""

import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from ads1299 import code_to_volts


def convert_units(data_codes, vref, gain, unit):
    """Convert ADC codes to the selected unit."""
    volts = code_to_volts(data_codes, vref=vref, gain=gain)

    if unit == "v":
        return volts, "V"
    elif unit == "mv":
        return volts * 1e3, "mV"
    else:  # uv
        return volts * 1e6, "\u00b5V"


class LivePlotter:
    """Builds and drives the live matplotlib figure from a PacketSink."""

    def __init__(self, sink, *, mode, port_label, gain, vref, unit,
                 fs, imu_fs, window, channels, refresh_ms, max_plot_points,
                 imu_angle_ylim, imu_accel_ylim, ylim, outfile):
        self.sink = sink
        self.mode = mode
        self.port_label = port_label
        self.fs = fs
        self.imu_fs = imu_fs
        self.window = window
        self.channels = channels
        self.refresh_ms = refresh_ms
        self.max_plot_points = max_plot_points
        self.imu_angle_ylim = imu_angle_ylim
        self.imu_accel_ylim = imu_accel_ylim
        self.ylim = ylim
        self.gain = gain
        self.vref = vref
        self.unit = unit
        self.outfile = outfile

        self.show_emg = mode in ("both", "emg")
        self.show_imu = mode in ("both", "imu")
        self.start_time = time.time()

        self._build_figure()

    # ------------------------------------------------------------------
    def _build_figure(self):
        # IMU plots: 3 for Euler angles + 3 for linear acceleration
        imu_plot_channels = 6  # roll, pitch, yaw, accel_x, accel_y, accel_z
        n_rows = max(self.channels if self.show_emg else 0,
                     imu_plot_channels if self.show_imu else 0)
        self.fig = plt.figure(figsize=(16, 12))
        n_columns = int(self.show_emg) + int(self.show_imu)
        gs = self.fig.add_gridspec(n_rows, n_columns, width_ratios=[1] * n_columns,
                                   wspace=0.18, hspace=0.05)
        emg_column = 0 if self.show_emg else None
        imu_column = int(self.show_emg) if self.show_imu else None
        self.emg_axes = ([self.fig.add_subplot(gs[i, emg_column])
                          for i in range(self.channels)] if self.show_emg else [])
        self.imu_axes = ([self.fig.add_subplot(gs[i, imu_column])
                          for i in range(imu_plot_channels)] if self.show_imu else [])
        self.fig.suptitle(
            f"Live {self.mode.upper()} -- {self.port_label} -- "
            f"gain={self.gain:g}, {self.unit} -- logging to {self.outfile}"
        )

        self.emg_lines = []
        self.emg_range_labels = []
        for i, ax in enumerate(self.emg_axes):
            ln, = ax.plot([], [], lw=0.8, color="tab:blue")
            ax.set_ylabel(f"ch{i+1}")
            ax.set_xlim(0, self.window)
            if self.ylim is not None:
                ax.set_ylim(self.ylim[0], self.ylim[1])
            lbl = ax.text(0.005, 0.85, "", transform=ax.transAxes,
                          fontsize=7, family="monospace", va="top")
            self.emg_lines.append(ln)
            self.emg_range_labels.append(lbl)

        self.imu_lines = []
        self.imu_range_labels = []
        imu_labels = ["roll", "pitch", "yaw", "accel_x", "accel_y", "accel_z"]
        for i, ax in enumerate(self.imu_axes):
            ln, = ax.plot([], [], lw=0.8, color="tab:orange")
            ax.set_ylabel(imu_labels[i])
            ax.set_xlim(0, self.window)
            if i < 3:
                ax.set_ylim(self.imu_angle_ylim[0], self.imu_angle_ylim[1])
            else:
                ax.set_ylim(self.imu_accel_ylim[0], self.imu_accel_ylim[1])
            lbl = ax.text(0.005, 0.85, "", transform=ax.transAxes,
                          fontsize=7, family="monospace", va="top")
            self.imu_lines.append(ln)
            self.imu_range_labels.append(lbl)

        if self.emg_axes:
            self.emg_axes[-1].set_xlabel("Time (s)")
        if self.imu_axes:
            self.imu_axes[-1].set_xlabel("Time (s)")
        self.status_text = self.fig.text(0.01, 0.005, "", fontsize=9, family="monospace")

        self.ani = animation.FuncAnimation(
            self.fig, self.update, interval=self.refresh_ms,
            blit=False, cache_frame_data=False,
        )

    # ------------------------------------------------------------------
    def _trace_data(self, indices, values, fs):
        """Return matching, bounded numpy arrays for one trace group."""
        n = min(len(indices), len(values))
        if n < 1:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        indices = indices[-n:]
        values = values[-n:]
        if n > self.max_plot_points:
            step = int(np.ceil(n / self.max_plot_points))
            indices = indices[::step]
            values = values[::step]
        # Keep the newest sample at the right edge of the live window.
        t = ((np.asarray(indices, dtype=np.float64) - indices[-1]) / self.fs
             + self.window)
        return t, np.asarray(values, dtype=np.float64)

    # ------------------------------------------------------------------
    def update(self, frame):
        snap = self.sink.snapshot()
        sample_idx = snap["sample_idx"]
        emg_raw = snap["emg_raw"]
        imu_euler_sample_idx = snap["imu_euler_sample_idx"]
        euler_raw = snap["euler_raw"]
        imu_accel_sample_idx = snap["imu_accel_sample_idx"]
        accel_raw = snap["accel_raw"]
        emg_total_count = snap["emg_total_count"]
        imu_total_count = snap["imu_total_count"]
        status1_bad_count = snap["status1_bad_count"]
        status2_bad_count = snap["status2_bad_count"]

        if self.show_emg and (not sample_idx or len(emg_raw[0]) < 2):
            return (self.emg_lines + self.emg_range_labels
                    + self.imu_lines + self.imu_range_labels + [self.status_text])

        t_emg = np.array([], dtype=np.float64)
        emg_data = []
        if self.show_emg and sample_idx and emg_raw and emg_raw[0]:
            t_emg, _ = self._trace_data(sample_idx, emg_raw[0], self.fs)
            emg_data = [self._trace_data(sample_idx, values, self.fs)[1]
                        for values in emg_raw]
        t_euler = np.array([], dtype=np.float64)
        euler_data = []
        if self.show_imu and euler_raw and euler_raw[0]:
            t_euler, _ = self._trace_data(imu_euler_sample_idx, euler_raw[0], self.imu_fs)
            euler_data = [self._trace_data(imu_euler_sample_idx, values, self.imu_fs)[1]
                          for values in euler_raw]
        t_accel = np.array([], dtype=np.float64)
        accel_data = []
        if self.show_imu and accel_raw and accel_raw[0]:
            t_accel, _ = self._trace_data(imu_accel_sample_idx, accel_raw[0], self.imu_fs)
            accel_data = [self._trace_data(imu_accel_sample_idx, values, self.imu_fs)[1]
                          for values in accel_raw]

        if self.show_emg and emg_data:
            emg_data_array = np.asarray(emg_data, dtype=np.float64)
            emg_converted, unit_label = convert_units(
                emg_data_array, vref=self.vref, gain=self.gain, unit=self.unit
            )

            for ln, ch in zip(self.emg_lines, emg_converted):
                ln.set_data(t_emg, ch)

            for ax, ch, lbl in zip(self.emg_axes, emg_converted, self.emg_range_labels):
                if len(ch) > 1 and frame % 5 == 0:
                    lo, hi = np.min(ch), np.max(ch)
                    if self.ylim is None:
                        pad = max((hi - lo) * 0.1, 1)
                        ax.set_ylim(lo - pad, hi + pad)
                    lbl.set_text(f"{lo:.1f}..{hi:.1f} {unit_label}")

        # Euler angles (roll, pitch, yaw).
        for i in range(3):
            if not self.show_imu:
                break
            if i < len(euler_data) and len(euler_data[i]) > 0:
                self.imu_lines[i].set_data(t_euler[-len(euler_data[i]):], euler_data[i])
                if len(euler_data[i]) > 1 and frame % 5 == 0:
                    lo, hi = np.min(euler_data[i]), np.max(euler_data[i])
                    pad = max((hi - lo) * 0.1, 1.0)
                    self.imu_axes[i].set_ylim(lo - pad, hi + pad)
                    self.imu_range_labels[i].set_text(f"{lo:.1f}..{hi:.1f} \u00b0")

        # Linear acceleration - use its own sample timeline.
        if self.show_imu:
            has_accel_data = any(len(d) > 0 for d in accel_data)
            if has_accel_data:
                for i in range(3):
                    if i < len(accel_data) and len(accel_data[i]) > 0:
                        self.imu_lines[i + 3].set_data(
                            t_accel[-len(accel_data[i]):], accel_data[i]
                        )
                        if len(accel_data[i]) > 1 and frame % 5 == 0:
                            lo, hi = np.min(accel_data[i]), np.max(accel_data[i])
                            pad = max((hi - lo) * 0.1, 0.1)
                            self.imu_axes[i + 3].set_ylim(lo - pad, hi + pad)
                            self.imu_range_labels[i + 3].set_text(
                                f"{lo:.2f}..{hi:.2f} m/s\u00b2"
                            )
            else:
                # No acceleration data available
                for i in range(3):
                    self.imu_lines[i + 3].set_data([], [])
                    self.imu_range_labels[i + 3].set_text("no data")

        bad1_pct = 100.0 * status1_bad_count / max(emg_total_count, 1)
        bad2_pct = 100.0 * status2_bad_count / max(emg_total_count, 1)
        self.status_text.set_text(
            f"emg samples: {emg_total_count}  bad1-status: {status1_bad_count} "
            f"({bad1_pct:.1f}%) bad2-status: {status2_bad_count} ({bad2_pct:.1f}%)  "
            f"imu samples: {imu_total_count}"
        )

        if frame % 100 == 0:
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                print(f"EMG rate: {emg_total_count / elapsed:.1f} Hz, "
                      f"IMU rate: {imu_total_count / elapsed:.1f} Hz")

        return (self.emg_lines + self.emg_range_labels
                + self.imu_lines + self.imu_range_labels + [self.status_text])

    # ------------------------------------------------------------------
    def show(self):
        """Block until the plot window is closed."""
        plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.05)
        plt.show()