"""
protocol.py - Pure packet decoding for the Sokosti binary protocol.

This layer has NO state and NO I/O. It only knows how to interpret a
fixed-size byte buffer as either an EMG sample packet or an IMU packet.
The byte-stream framing (finding sync markers, buffering partial data)
lives in frames.py; the storage/CSV side lives in sink.py.

The packet layouts here must match the firmware's C++ structs exactly.
"""

import struct

import numpy as np
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# Packet geometry
# ---------------------------------------------------------------------------
ADS1299_NUM_BITS = 24
ADS1299_MAX_CODE = 2 ** (ADS1299_NUM_BITS - 1)
ADS1299_VREF = 4.5
ADS1299_GAIN = 8

EMG_SYNC_MARKER = b"\xAA\x55"
IMU_SYNC_MARKER = b"\xBB\x66"

# sync[2] | sample_idx(4) | status1(1) | status2(1) | ch_data(48) | checksum(1)
SAMPLE_PACKET_FORMAT = "<BB I BB 48s B"
SAMPLE_PACKET_SIZE = struct.calcsize(SAMPLE_PACKET_FORMAT)

# sync(2) | sample_idx(4) | sensor_id(1) | data_len(1) | data(10) | checksum(1)
IMU_PACKET_FORMAT = "<2sI B B 10s B"
IMU_PACKET_SIZE = struct.calcsize(IMU_PACKET_FORMAT)

NUM_CHANNELS = 16


def extract_24bit_signed(data, offset):
    """Extract a signed 24-bit little-endian value from ``data`` at ``offset``."""
    val = (data[offset] << 16) | (data[offset + 1] << 8) | data[offset + 2]
    if val & 0x800000:
        val -= 0x1000000
    return val


def find_sync_marker(data, start=0, marker=EMG_SYNC_MARKER):
    """Return the index of the first ``marker`` in ``data`` at/after ``start``, else -1."""
    marker_len = len(marker)
    for i in range(start, len(data) - marker_len + 1):
        if data[i:i + marker_len] == marker:
            return i
    return -1


def _xor_checksum(packet_bytes):
    """XOR of every byte except the trailing checksum byte."""
    computed = 0
    for b in packet_bytes[:-1]:
        computed ^= b
    return computed


def parse_sample_packet(data):
    """
    Parse one EMG sample packet.

    Returns a dict with keys ``sample_idx``, ``status1_ok``, ``status2_ok``,
    and ``channels`` (list of 16 signed ADC codes), or None if the packet is
    truncated, has a bad sync marker, or fails the XOR checksum.
    """
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

    # Reject false-positive sync matches: XOR all bytes except the checksum byte.
    if _xor_checksum(packet_bytes) != checksum:
        return None

    channels = [extract_24bit_signed(ch_data_raw, i * 3) for i in range(NUM_CHANNELS)]
    return {
        "sample_idx": sample_idx,
        "status1_ok": bool(status1_ok),
        "status2_ok": bool(status2_ok),
        "channels": channels,
    }


def parse_imu_packet(data):
    """
    Parse a single IMU packet.

    The driver sends two different BHI360 virtual-sensor payloads:
      sensor 37 (GAMERV): int16 x,y,z,w + uint16 accuracy = 10 bytes
      sensor  4 (ACC):    int16 x,y,z = 6 bytes
    Quaternion components are Q14; corrected acceleration is 1/16 m/s^2/LSB.

    Returns a dict with ``sample_idx``, ``sensor_id``, ``kind`` and either
    ``quat``/``accuracy``/``euler`` (quaternion) or ``accel`` (acceleration),
    or None if the packet is invalid.
    """
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

    if _xor_checksum(packet_bytes) != checksum:
        return None

    if sensor_id == 37 and data_len == 10:
        x, y, z, w, accuracy = struct.unpack("<4hH", data_raw[:10])
        quat = np.asarray([x, y, z, w], dtype=np.float64) / 16384.0
        norm = np.linalg.norm(quat)
        if norm == 0:
            return None
        quat /= norm
        euler = R.from_quat(quat).as_euler("xyz", degrees=True)
        return {
            "sample_idx": sample_idx,
            "sensor_id": sensor_id,
            "kind": "quaternion",
            "quat": quat.tolist(),
            "accuracy": accuracy / 16384.0,
            "euler": euler.tolist(),
        }

    if sensor_id == 4 and data_len == 6:
        accel_raw = struct.unpack("<3h", data_raw[:6])
        return {
            "sample_idx": sample_idx,
            "sensor_id": sensor_id,
            "kind": "acceleration",
            "accel": [value / 16.0 for value in accel_raw],
        }

    return None