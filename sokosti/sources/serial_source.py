"""
serial_source.py - USB CDC-ACM byte transport (SerialSource).

A background thread reads bounded chunks from a pyserial port and feeds
them into a FrameParser. Complete packets are dispatched into a
PacketSink. This replaces the SerialReader class from emg_live_plot.py.
"""

import sys
import threading
import time

import serial

from ..logging import RunLogger


class SerialSource(threading.Thread):
    """Reads bytes from a serial port and pushes packets into a sink."""

    def __init__(self, ser, parser, sink, read_size=512, log_file=None):
        super().__init__(daemon=True)
        self.ser = ser
        self.parser = parser
        self.sink = sink
        self.read_size = read_size
        self.stop_flag = threading.Event()
        self.log_file = log_file
        self.logger = RunLogger(log_file) if log_file else None

        # Per-second EMG/IMU rate counters. Reset every second so the
        # reported rate is samples received in that second, not a
        # cumulative average over the whole run.
        self._sec_emg = 0
        self._sec_imu = 0
        self._last_rate = time.time()

    def _log(self, message):
        if self.logger is not None:
            self.logger.log(message)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr)

    def _report_rate(self, now=None):
        now = now or time.time()
        elapsed = now - self._last_rate
        if elapsed <= 0:
            elapsed = 1e-9
        self._log(
            f"EMG rate: {self._sec_emg / elapsed:.1f} Hz, "
            f"IMU rate: {self._sec_imu / elapsed:.1f} Hz"
        )
        self._sec_emg = 0
        self._sec_imu = 0

    def run(self):
        if self.logger is not None:
            self.logger.write_header(
                "Sokosti USB live capture",
                metadata={
                    "port": self.ser.port,
                    "baud": self.ser.baudrate,
                    "log_file": self.log_file,
                },
            )
        print("Serial reader started...", file=sys.stderr)
        while not self.stop_flag.is_set():
            try:
                # Read bounded chunks so a large host-side USB backlog does not
                # monopolize the GUI thread's companion reader thread.
                chunk = self.ser.read(min(self.ser.in_waiting or 128, self.read_size))
            except serial.SerialException as e:
                print(f"Serial error: {e}", file=sys.stderr)
                break

            if not chunk:
                continue

            for kind, packet in self.parser.feed(chunk):
                if kind == "emg":
                    self.sink.add_emg(packet)
                    self._sec_emg += 1
                else:
                    self.sink.add_imu(packet)
                    self._sec_imu += 1

            # Per-second EMG/IMU rate report. Counts are reset every second
            # so the reported rate is samples received in that second, not a
            # cumulative average over the whole run.
            now = time.time()
            if now - self._last_rate >= 1.0:
                self._report_rate(now)
                self._last_rate = now

    def stop(self):
        self.stop_flag.set()

    def close(self):
        """Close the run log file, if one was opened."""
        if self.logger is not None:
            self.logger.close()