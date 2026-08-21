"""
sokosti.sources - byte transports for the shared pipeline.

Each source is a thin wrapper that produces bytes and feeds them into a
FrameParser, which dispatches complete packets into a PacketSink. USB and
BLE are just two different byte sources for the same pipeline.
"""

from .serial_source import SerialSource
from .ble_source import BleSource

__all__ = ["SerialSource", "BleSource"]