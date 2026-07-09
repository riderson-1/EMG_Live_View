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


ADS1299_NUM_BITS = 24
ADS1299_MAX_CODE = 2 ** (ADS1299_NUM_BITS - 1)  # 2^23


def code_to_volts(code, vref=4.5, gain=24):
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
    return np.asarray(code, dtype=float) * (vref / gain) / ADS1299_MAX_CODE


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

