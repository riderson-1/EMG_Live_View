import struct

from sokosti.protocol import (
    parse_sample_packet,
    parse_imu_packet,
    SAMPLE_PACKET_SIZE,
    IMU_PACKET_SIZE,
)
from sokosti.frames import FrameParser


def _checksum(packet_bytes: bytes) -> int:
    """XOR of every byte except the trailing checksum byte."""
    checksum = 0
    for b in packet_bytes[:-1]:
        checksum ^= b
    return checksum


def _make_emg_packet(sample_idx=42, status1=1, status2=1):
    raw = bytearray(SAMPLE_PACKET_SIZE)
    raw[0] = 0xAA
    raw[1] = 0x55
    struct.pack_into("<I", raw, 2, sample_idx)
    raw[6] = status1
    raw[7] = status2
    # 16 channels x 3 bytes, big-endian 24-bit signed (MSB first),
    # matching extract_24bit_signed. Channel i holds (i + 1) * 1000.
    for i in range(16):
        val = (i + 1) * 1000
        off = 8 + i * 3
        raw[off] = (val >> 16) & 0xFF
        raw[off + 1] = (val >> 8) & 0xFF
        raw[off + 2] = val & 0xFF
    raw[56] = _checksum(raw)
    return bytes(raw)


def _make_imu_accel_packet(sample_idx=99, x=16, y=32, z=48):
    raw = bytearray(IMU_PACKET_SIZE)
    raw[0] = 0xBB
    raw[1] = 0x66
    struct.pack_into("<I", raw, 2, sample_idx)
    raw[6] = 4          # sensor_id: ACC
    raw[7] = 6          # data_len: 3 x int16
    struct.pack_into("<3h", raw, 8, x, y, z)
    raw[18] = _checksum(raw)
    return bytes(raw)


def test_parse_emg_packet():
    raw = _make_emg_packet(sample_idx=42)
    packet = parse_sample_packet(raw)

    assert packet is not None
    assert packet["sample_idx"] == 42
    assert packet["status1_ok"] is True
    assert packet["status2_ok"] is True
    # Channel i holds (i + 1) * 1000.
    assert packet["channels"][0] == 1000
    assert packet["channels"][1] == 2000
    assert packet["channels"][15] == 16000


def test_parse_imu_accel_packet():
    raw = _make_imu_accel_packet(sample_idx=99, x=16, y=32, z=48)
    packet = parse_imu_packet(raw)

    assert packet is not None
    assert packet["sample_idx"] == 99
    assert packet["sensor_id"] == 4
    assert packet["kind"] == "acceleration"
    # 1/16 m/s^2 per LSB
    assert packet["accel"] == [1.0, 2.0, 3.0]


def test_frame_parser_recovers_split_packets():
    """A packet split across feed() calls must still be parsed once."""
    raw = _make_emg_packet(sample_idx=7)
    parser = FrameParser()

    packets = []
    for i in range(len(raw)):
        packets.extend(parser.feed(raw[i:i + 1]))

    assert len(packets) == 1
    kind, packet = packets[0]
    assert kind == "emg"
    assert packet["sample_idx"] == 7


def test_frame_parser_recovers_from_garbage():
    """Leading garbage before a valid packet must be skipped."""
    raw = _make_emg_packet(sample_idx=3)
    parser = FrameParser()
    packets = parser.feed(b"\x00\x01\x02" + raw)

    assert len(packets) == 1
    kind, packet = packets[0]
    assert kind == "emg"
    assert packet["sample_idx"] == 3
