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

        # Telemetry counters (transport-specific, not in the shared sink).
        self.bytes_rx = 0
        self.packets_rx = 0
        self.checksum_failures = 0
        self.connect_count = 0
        self.disconnect_count = 0
        self.start_time = time.time()
        self._last_telemetry = time.time()

    # ------------------------------------------------------------------
    def _log(self, message):
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(line + "\n")

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
        self.bytes_rx += len(data)
        for kind, packet in self.parser.feed(bytes(data)):
            self.packets_rx += 1
            if kind == "emg":
                self.sink.add_emg(packet)
            else:
                self.sink.add_imu(packet)
        self.checksum_failures = self.parser.error_count

        # Periodic throughput/stability report to the terminal.
        now = time.time()
        if now - self._last_telemetry >= 5.0:
            self._report_telemetry(now)
            self._last_telemetry = now

    def _report_telemetry(self, now=None):
        now = now or time.time()
        elapsed = now - self.start_time
        if elapsed <= 0:
            return
        self._log(
            f"BLE: {self.bytes_rx / elapsed:.0f} B/s, "
            f"{self.packets_rx / elapsed:.1f} pkts/s, "
            f"checksum failures: {self.checksum_failures}, "
            f"connects: {self.connect_count}, disconnects: {self.disconnect_count}"
        )

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

            await client.start_notify(NUS_TX_UUID, self._on_notify)
            self._log("Subscribed to NUS TX. Streaming (Ctrl+C to quit)...")

            # Block until the disconnected_callback fires, rather than polling
            # client.is_connected (which can lag the actual event).
            await disconnected_event.wait()

    # ------------------------------------------------------------------
    async def run(self):
        """Connect (with auto-reconnect) and stream until cancelled."""
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

            self._report_telemetry()
            self._log(f"Reconnecting in {self.retry_delay:.0f}s...")
            await asyncio.sleep(self.retry_delay)