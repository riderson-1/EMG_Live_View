"""
ble_source.py - Bluetooth Low Energy (NUS) byte transport (BleSource).

BleSource wraps a bleak BLE connection to the Nordic UART Service (NUS)
and feeds notification bytes into a FrameParser, exactly like the serial
source does. It also tracks connection/throughput/stability telemetry
(bytes, packets, checksum failures, reconnects) that can be printed to
the terminal or written to a log file -- separate from the plot window.
"""

import asyncio
import time

from bleak import BleakClient, BleakScanner

from ..logging import RunLogger

# Standard Nordic UART Service UUIDs
NUS_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
NUS_RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # write:  PC -> nRF
NUS_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # notify: nRF -> PC

DEFAULT_NAME = "Sokosti_BLE"


class BleSource:
    """
    Connects to the Sokosti nRF5340 over NUS and streams packets into a sink.

    Telemetry (bytes/packets received, checksum failures, reconnects) is
    accumulated on the instance and can be printed or logged separately.
    """

    def __init__(self, parser, sink, *, name=DEFAULT_NAME, address=None,
                 retry_delay=3.0, scan_timeout=10.0, log_file=None):
        self.parser = parser
        self.sink = sink
        self.name = name
        self.address = address
        self.retry_delay = retry_delay
        self.scan_timeout = scan_timeout
        self.log_file = log_file
        self.logger = RunLogger(log_file) if log_file else None

        # Session-cumulative counters (since connect start; used only for
        # the final summary printed on disconnect, NOT for the live rate --
        # a cumulative average hides real-time behavior and converges
        # slowly, which makes short test runs look misleadingly bad/good).
        self.bytes_rx = 0
        self.packets_rx = 0
        self.notifications_rx = 0
        self.checksum_failures = 0
        self.connect_count = 0
        self.disconnect_count = 0
        self.start_time = time.time()

        # Windowed counters: reset every report interval, so the printed
        # rate reflects the last ~5s only. This is the number to watch.
        self._window_bytes = 0
        self._window_packets = 0
        self._window_notifications = 0
        self._window_notif_min = None
        self._window_notif_max = None
        self._last_telemetry = time.time()

        # Per-second EMG/IMU rate counters. These are reset every second so
        # the reported rate is the actual samples received in that second,
        # not a cumulative average over the whole run.
        self._sec_emg = 0
        self._sec_imu = 0
        self._last_rate = time.time()

    # ------------------------------------------------------------------
    def _log(self, message):
        if self.logger is not None:
            self.logger.log(message)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] {message}")

    # ------------------------------------------------------------------
    def close(self):
        """Close the run log file, if one was opened."""
        if self.logger is not None:
            self.logger.close()

    # ------------------------------------------------------------------
    async def _find_device(self):
        print(f"Scanning for '{self.name}' ({self.scan_timeout:.0f}s)...")
        device = await BleakScanner.find_device_by_name(self.name, timeout=self.scan_timeout)
        if device is None:
            print(f"Device '{self.name}' not found. Retrying...")
            return None
        print(f"Found {device.name} [{device.address}]")
        return device.address

    # ------------------------------------------------------------------
    def _on_notify(self, _handle, data: bytearray):
        """BLE notification callback: feed bytes into the shared parser."""
        n = len(data)

        # Raw radio-level stats: one entry per actual BLE notification,
        # independent of how many samples the parser recovers from it.
        # This is what tells you whether notifications are still small
        # (fragmented) even after a larger ATT MTU is negotiated.
        self.bytes_rx += n
        self.notifications_rx += 1
        self._window_bytes += n
        self._window_notifications += 1
        self._window_notif_min = n if self._window_notif_min is None else min(self._window_notif_min, n)
        self._window_notif_max = n if self._window_notif_max is None else max(self._window_notif_max, n)

        for kind, packet in self.parser.feed(bytes(data)):
            self.packets_rx += 1
            self._window_packets += 1
            if kind == "emg":
                self.sink.add_emg(packet)
                self._sec_emg += 1
            else:
                self.sink.add_imu(packet)
                self._sec_imu += 1
        self.checksum_failures = self.parser.error_count

        # Periodic throughput/stability report to the terminal.
        now = time.time()
        if now - self._last_telemetry >= 5.0:
            self._report_telemetry(now)
            self._last_telemetry = now

        # Per-second EMG/IMU rate report. Counts are reset every second so
        # the reported rate is samples received in that second, not a
        # cumulative average over the whole run.
        if now - self._last_rate >= 1.0:
            self._report_rate(now)
            self._last_rate = now

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

    def _report_telemetry(self, now=None, final=False):
        now = now or time.time()

        if final:
            # Session summary: cumulative average since connect.
            elapsed = now - self.start_time
            if elapsed <= 0:
                return
            self._log(
                f"BLE session summary: {self.bytes_rx / elapsed:.0f} B/s avg, "
                f"{self.packets_rx / elapsed:.1f} samples/s avg, "
                f"{self.notifications_rx} notifications total, "
                f"checksum failures: {self.checksum_failures}, "
                f"connects: {self.connect_count}, disconnects: {self.disconnect_count}"
            )
            return

        window_elapsed = now - self._last_telemetry
        if window_elapsed <= 0:
            window_elapsed = 1e-9

        avg_notif_size = (
            self._window_bytes / self._window_notifications
            if self._window_notifications else 0.0
        )
        notif_min = self._window_notif_min if self._window_notif_min is not None else 0
        notif_max = self._window_notif_max if self._window_notif_max is not None else 0

        self._log(
            f"BLE: {self._window_bytes / window_elapsed:.0f} B/s, "
            f"{self._window_packets / window_elapsed:.1f} samples/s, "
            f"{self._window_notifications / window_elapsed:.1f} notifications/s "
            f"(size avg={avg_notif_size:.0f}B min={notif_min}B max={notif_max}B), "
            f"checksum failures: {self.checksum_failures}, "
            f"connects: {self.connect_count}, disconnects: {self.disconnect_count}"
        )

        # Reset the window for the next interval.
        self._window_bytes = 0
        self._window_packets = 0
        self._window_notifications = 0
        self._window_notif_min = None
        self._window_notif_max = None

    # ------------------------------------------------------------------
    async def _acquire_mtu(self, client):
        """
        Force the backend to report the real negotiated ATT MTU.

        bleak's default-23 report on `client.mtu_size` does NOT mean the
        Exchange MTU procedure failed -- it just means bleak never asked
        the backend for the negotiated value. The accessor for this has
        moved across bleak versions (client._acquire_mtu() in older
        releases, client._backend._acquire_mtu() in newer ones), so this
        tries both and falls back gracefully if neither exists.
        """
        for accessor in (
            getattr(client, "_acquire_mtu", None),
            getattr(getattr(client, "_backend", None), "_acquire_mtu", None),
        ):
            if accessor is None:
                continue
            try:
                await accessor()
                return True
            except Exception:
                continue
        return False

    # ------------------------------------------------------------------
    async def _connect_and_stream(self, address, disconnected_event):
        def on_disconnect(_client):
            self.disconnect_count += 1
            self._log("Device disconnected.")
            disconnected_event.set()

        async with BleakClient(address, disconnected_callback=on_disconnect) as client:
            self.connect_count += 1
            self._log(f"Connected: {client.is_connected}")

            nus = client.services.get_service(NUS_SERVICE_UUID)
            if nus is None:
                self._log("NUS service not found on this device.")
                return

            if await self._acquire_mtu(client):
                self._log(f"ATT MTU: {client.mtu_size}")
            else:
                self._log(f"Could not acquire real MTU, using default: {client.mtu_size}")

            await client.start_notify(NUS_TX_UUID, self._on_notify)
            self._log("Subscribed to NUS TX. Streaming (Ctrl+C to quit)...")

            # Block until the disconnected_callback fires, rather than polling
            # client.is_connected (which can lag the actual event).
            await disconnected_event.wait()

    # ------------------------------------------------------------------
    async def run(self):
        """Connect (with auto-reconnect) and stream until cancelled."""
        if self.logger is not None:
            self.logger.write_header(
                "Sokosti BLE live capture",
                metadata={
                    "name": self.name,
                    "address": self.address,
                    "retry_delay": self.retry_delay,
                    "scan_timeout": self.scan_timeout,
                    "log_file": self.log_file,
                },
            )

        fixed_address = self.address

        while True:
            addr = fixed_address
            while addr is None:
                addr = await self._find_device()
                if addr is None:
                    await asyncio.sleep(self.retry_delay)

            disconnected_event = asyncio.Event()
            try:
                await self._connect_and_stream(addr, disconnected_event)
            except Exception as exc:
                self._log(f"Connection error: {exc}")

            self._report_telemetry(final=True)
            self._log(f"Reconnecting in {self.retry_delay:.0f}s...")
            await asyncio.sleep(self.retry_delay)