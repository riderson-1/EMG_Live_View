"""
frames.py - Byte-stream framing (FrameParser).

A FrameParser turns an arbitrary byte stream (serial reads, BLE
notifications, file chunks, ...) into a sequence of complete, validated
packets. It owns the partial-buffer state and the sync-marker recovery
logic that used to live inside SerialReader.run().

Transports call ``parser.feed(chunk)`` and receive a list of
``(kind, packet)`` tuples where ``kind`` is ``"emg"`` or ``"imu"``.
"""

from .protocol import (
    EMG_SYNC_MARKER,
    IMU_SYNC_MARKER,
    SAMPLE_PACKET_SIZE,
    IMU_PACKET_SIZE,
    find_sync_marker,
    parse_sample_packet,
    parse_imu_packet,
)


class FrameParser:
    """Stateful byte-stream framer. Not thread-safe; call from one thread."""

    def __init__(self):
        self.buffer = b""
        self.error_count = 0

    def feed(self, chunk: bytes):
        """
        Append ``chunk`` to the internal buffer and extract any complete
        packets. Returns a list of ``(kind, packet)`` tuples.
        """
        self.buffer += chunk
        packets = []
        processed = 0

        while len(self.buffer) >= 2:
            emg_idx = find_sync_marker(self.buffer, marker=EMG_SYNC_MARKER)
            imu_idx = find_sync_marker(self.buffer, marker=IMU_SYNC_MARKER)

            if emg_idx == -1 and imu_idx == -1:
                break

            if emg_idx == -1:
                sync_idx, packet_size, packet_type = imu_idx, IMU_PACKET_SIZE, "imu"
            elif imu_idx == -1:
                sync_idx, packet_size, packet_type = emg_idx, SAMPLE_PACKET_SIZE, "emg"
            elif emg_idx < imu_idx:
                sync_idx, packet_size, packet_type = emg_idx, SAMPLE_PACKET_SIZE, "emg"
            else:
                sync_idx, packet_size, packet_type = imu_idx, IMU_PACKET_SIZE, "imu"

            if sync_idx > 0:
                self.buffer = self.buffer[sync_idx:]

            if len(self.buffer) < packet_size:
                break

            if packet_type == "emg":
                packet = parse_sample_packet(self.buffer[:packet_size])
            else:
                packet = parse_imu_packet(self.buffer[:packet_size])

            if packet is not None:
                packets.append((packet_type, packet))
                self.buffer = self.buffer[packet_size:]
                processed += 1
            else:
                # Invalid packet at the sync marker: drop one byte and resync.
                self.buffer = self.buffer[1:]
                self.error_count += 1

        # Prevent unbounded buffer growth when no complete packet is found.
        if processed == 0 and len(self.buffer) > SAMPLE_PACKET_SIZE * 4:
            self.buffer = self.buffer[-SAMPLE_PACKET_SIZE * 2:]

        return packets