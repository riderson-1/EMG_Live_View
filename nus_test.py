"""
nus_test.py - Minimal Nordic UART Service (NUS) receiver.

Proof-of-concept: scans for the Sokosti nRF5340, connects, subscribes to
NUS TX notifications, and prints whatever text arrives (e.g. "Hello World").

Requires: bleak
    pip install bleak --break-system-packages

Usage:
    python nus_test.py                      # scans for default name below
    python nus_test.py --name Sokosti_Test  # override device name
    python nus_test.py --address AA:BB:CC:DD:EE:FF   # skip scan, connect directly
"""

import argparse
import asyncio
import sys

from bleak import BleakClient, BleakScanner

# Standard Nordic UART Service UUIDs
NUS_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
NUS_RX_UUID       = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # write:  PC -> nRF
NUS_TX_UUID       = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # notify: nRF -> PC

DEFAULT_NAME = "Sokosti_Test"


def on_notify(_handle, data: bytearray):
    """Called whenever the nRF sends data over the NUS TX characteristic."""
    try:
        text = data.decode("utf-8").rstrip("\r\n")
        print(f"RX ({len(data)} bytes): {text}")
    except UnicodeDecodeError:
        print(f"RX ({len(data)} bytes, binary): {data.hex()}")


async def find_device(name: str, timeout: float = 10.0):
    print(f"Scanning for '{name}' ({timeout:.0f}s)...")
    device = await BleakScanner.find_device_by_name(name, timeout=timeout)
    if device is None:
        print(f"Device '{name}' not found. Retrying...")
        return None
    print(f"Found {device.name} [{device.address}]")
    return device.address


async def connect_and_stream(address: str, disconnected_event: asyncio.Event):
    """Connect once, subscribe to NUS TX, and block until disconnected."""

    def on_disconnect(_client: BleakClient):
        print("Device disconnected.")
        disconnected_event.set()

    async with BleakClient(address, disconnected_callback=on_disconnect) as client:
        print(f"Connected: {client.is_connected}")

        nus = client.services.get_service(NUS_SERVICE_UUID)
        if nus is None:
            print("NUS service not found on this device.")
            return

        await client.start_notify(NUS_TX_UUID, on_notify)
        print("Subscribed to NUS TX. Waiting for messages (Ctrl+C to quit)...\n")

        # Block here until the disconnected_callback fires, rather than
        # polling client.is_connected (which can lag the actual event).
        await disconnected_event.wait()


async def main(address: str | None, name: str, retry_delay: float):
    # If a fixed address was given, keep using it across reconnects.
    # Otherwise re-scan by name every time, in case the address changes
    # or multiple boards are around.
    fixed_address = address

    while True:
        addr = fixed_address
        while addr is None:
            addr = await find_device(name)
            if addr is None:
                await asyncio.sleep(retry_delay)

        disconnected_event = asyncio.Event()
        try:
            await connect_and_stream(addr, disconnected_event)
        except Exception as exc:
            print(f"Connection error: {exc}")

        print(f"Reconnecting in {retry_delay:.0f}s...\n")
        await asyncio.sleep(retry_delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal NUS receiver PoC with auto-reconnect")
    parser.add_argument("--name", default=DEFAULT_NAME,
                         help=f"BLE advertised name to scan for (default: {DEFAULT_NAME})")
    parser.add_argument("--address", default=None,
                         help="Connect directly by MAC/UUID, skipping the scan on every (re)connect")
    parser.add_argument("--retry-delay", type=float, default=3.0,
                         help="Seconds to wait before rescanning/reconnecting (default: 3)")
    args = parser.parse_args()

    try:
        asyncio.run(main(args.address, args.name, args.retry_delay))
    except KeyboardInterrupt:
        print("\nStopped.")