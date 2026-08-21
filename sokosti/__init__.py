"""
sokosti - shared live-plotting pipeline for the Sokosti nRF5340.

Four layers, each in its own module:

    protocol  -> pure packet decoding (no state)
    frames    -> byte stream framing (FrameParser)
    sink      -> thread-safe storage + CSV (PacketSink)
    plotter   -> matplotlib live plot (LivePlotter)

Transports live in sokosti/sources/ and feed bytes into a FrameParser,
which dispatches complete packets into a PacketSink. The LivePlotter
consumes snapshots from the sink. USB and BLE are just two different
byte sources for the same pipeline.
"""

from .protocol import (
    EMG_SYNC_MARKER,
    IMU_SYNC_MARKER,
    SAMPLE_PACKET_FORMAT,
    SAMPLE_PACKET_SIZE,
    IMU_PACKET_FORMAT,
    IMU_PACKET_SIZE,
    NUM_CHANNELS,
    extract_24bit_signed,
    find_sync_marker,
    parse_sample_packet,
    parse_imu_packet,
)
from .frames import FrameParser
from .sink import PacketSink
from .plotter import LivePlotter

__all__ = [
    "EMG_SYNC_MARKER",
    "IMU_SYNC_MARKER",
    "SAMPLE_PACKET_FORMAT",
    "SAMPLE_PACKET_SIZE",
    "IMU_PACKET_FORMAT",
    "IMU_PACKET_SIZE",
    "NUM_CHANNELS",
    "extract_24bit_signed",
    "find_sync_marker",
    "parse_sample_packet",
    "parse_imu_packet",
    "FrameParser",
    "PacketSink",
    "LivePlotter",
]