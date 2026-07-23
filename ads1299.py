#!/usr/bin/env python3
"""
ADS1299 conversion utilities.

The ADS1299 outputs 24-bit two's-complement ADC codes.

For normal channel measurements, the input-referred differential voltage is:

    volts = code * VREF / gain / 2^23

because the ADS1299 full-scale differential input range is:

    -VREF/gain to +VREF/gain

"""

import numpy as np
import struct

ADS1299_NUM_BITS = 24
ADS1299_MAX_CODE = 2 ** (ADS1299_NUM_BITS - 1)  # 2^23


def code_to_volts(code, vref, gain):
    """
    Convert signed ADS1299 ADC code(s) to input-referred volts.

    Parameters
    ----------
    code : int, float, list, tuple, or numpy array
        Signed 24-bit ADS1299 ADC code.
        This assumes the value is already converted from two's complement
        into a signed integer range:

            -8388608 to +8388607

    vref : float
        ADS1299 reference voltage in volts.
        Common ADS1299 internal reference is 4.5 V.

    gain : int or float
        PGA gain. Valid ADS1299 gains are typically:
        1, 2, 4, 6, 8, 12, 24.

    Returns
    -------
    float or numpy array
        Input-referred voltage in volts.
    """
    return np.asarray(code, dtype=np.float64) * (vref / gain) / ADS1299_MAX_CODE


def code_to_microvolts(code, vref=4.5, gain=24):
    """
    Convert signed ADS1299 ADC code(s) to input-referred microvolts.
    """
    return code_to_volts(code, vref=vref, gain=gain) * 1e6


def twos_complement_24_to_int(value):
    """
    Convert an unsigned 24-bit integer to signed integer.

    Use this only if your firmware sends raw unsigned 24-bit values.
    If your CSV already contains signed values, you do not need this.
    """
    value = int(value) & 0xFFFFFF

    if value & 0x800000:
        value -= 0x1000000

    return value

# SamplePacket binary format (must match the C++ struct exactly)
SAMPLE_PACKET_FORMAT = '!BB I BB 48s B'  # sync[2], sample_idx, status1_ok, status2_ok, ch_data[48], checksum
SAMPLE_PACKET_SIZE = struct.calcsize(SAMPLE_PACKET_FORMAT)

def parse_sample_packet(data):
    """
    Unpack a binary SamplePacket from USB.
    
    Parameters
    ----------
    data : bytes
        Raw 57-byte packet from the device.
    
    Returns
    -------
    dict or None
        Parsed packet with keys: sample_idx, status1_ok, status2_ok, channels (list of 16 values in volts)
        Returns None if sync marker is invalid or checksum fails.
    """
    if len(data) < SAMPLE_PACKET_SIZE:
        return None
    
    sync0, sync1, sample_idx, status1_ok, status2_ok, ch_data_raw, checksum = \
        struct.unpack(SAMPLE_PACKET_FORMAT, data[:SAMPLE_PACKET_SIZE])
    
    # Validate sync marker
    if sync0 != 0xAA or sync1 != 0x55:
        return None
    
    # TODO: validate checksum if needed
    
    # Extract 24-bit channel values and convert to volts
    channels = []
    for i in range(16):
        offset = i * 3
        raw_bytes = ch_data_raw[offset:offset+3]
        # Unpack big-endian 24-bit value
        val = (raw_bytes[0] << 16) | (raw_bytes[1] << 8) | raw_bytes[2]
        # Sign-extend from 24-bit
        if val & 0x800000:
            val -= 0x1000000
        # Convert to volts using your existing function (gain=8 for SingleChannelTest)
        volts = code_to_volts(val, vref=4.5, gain=8)
        channels.append(volts)
    
    return {
        'sample_idx': sample_idx,
        'status1_ok': status1_ok,
        'status2_ok': status2_ok,
        'channels': channels,
    }


def find_sync_marker(data, start=0):
    """
    Scan for 0xAA55 sync marker in a byte stream.
    Useful for recovering from a dropped/corrupted packet.
    """
    for i in range(start, len(data) - 1):
        if data[i] == 0xAA and data[i+1] == 0x55:
            return i
    return -1