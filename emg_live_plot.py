#!/usr/bin/env python3
"""
Live EMG plotter - USB CDC-ACM transport.

Entry point that wires the shared sokosti pipeline together for a USB
serial connection. The plotting behavior is identical to the original
emg_live_plot.py; only the byte source differs from the BLE variant.

Usage:
    python emg_live_plot.py --port /dev/ttyACM0 --gain 1 --mode both
"""

import argparse
import os
import time
from datetime import datetime

import serial

from sokosti import FrameParser, PacketSink, LivePlotter, NUM_CHANNELS
from sokosti.sources import SerialSource

# Create captures folder if it doesn't exist
os.makedirs("captures", exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Live EMG plot from ADS1299 binary packets (USB)")
    p.add_argument("--port", default="/dev/ttyACM0", help="Serial port")
    p.add_argument("--baud", type=int, default=115200, help="Baud rate")
    p.add_argument("--fs", type=float, default=1000.0, help="EMG sample rate in Hz")
    p.add_argument("--imu-fs", type=float, default=200.0,
                   help="IMU sample rate in Hz (quaternion + acceleration combined)")
    p.add_argument("--window", type=float, default=5.0, help="Rolling window length in seconds")
    p.add_argument("--channels", type=int, default=16, help="Number of EMG channels")
    p.add_argument("--outfile", default=None, help="CSV log path")
    p.add_argument("--refresh-ms", type=int, default=100, help="Plot refresh interval in ms")
    p.add_argument("--max-plot-points", type=int, default=1200,
                   help="Maximum points per trace to draw per refresh")
    p.add_argument("--imu-angle-ylim", nargs=2, type=float, default=(-180.0, 180.0),
                   metavar=("MIN", "MAX"),
                   help="Fixed Euler-angle axis limits in degrees")
    p.add_argument("--imu-accel-ylim", nargs=2, type=float, default=(-20.0, 20.0),
                   metavar=("MIN", "MAX"),
                   help="Fixed linear-acceleration axis limits in m/s^2")
    p.add_argument("--imu-channels", type=int, default=8, help="Number of IMU data traces to plot")
    p.add_argument("--mode", choices=("both", "emg", "imu"), default="both",
                   help="What to display: both sensors, EMG only, or IMU only")

    p.add_argument("--gain", type=float, default=8.0, help="ADS1299 PGA gain")
    p.add_argument("--vref", type=float, default=4.5, help="ADS1299 reference voltage")
    p.add_argument("--unit", choices=["v", "mv", "uv"], default="mv", help="Display unit")
    p.add_argument("--ylim", nargs=2, type=float, default=None, metavar=("YMIN", "YMAX"),
                   help="Fixed y-axis limits")

    return p.parse_args()


def main():
    args = parse_args()
    maxlen = int(args.fs * args.window)
    show_emg = args.mode in ("both", "emg")
    show_imu = args.mode in ("both", "imu")

    outfile = args.outfile or os.path.join(
        "captures", f"sokosti_capture_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    csv_header = (
        ["sample", "status1_ok", "status2_ok"]
        + [f"ch{i+1}" for i in range(NUM_CHANNELS)]
        + ["roll", "pitch", "yaw", "accel_x", "accel_y", "accel_z"]
    )

    print(f"Opening {args.port} @ {args.baud} baud")
    ser = serial.Serial(args.port, args.baud, timeout=0.01)
    time.sleep(0.5)
    # Discard bytes that accumulated before this run. This prevents an old
    # host-side USB buffer from appearing as a several-second live delay.
    ser.reset_input_buffer()

    parser = FrameParser()
    # The protocol always decodes NUM_CHANNELS per packet, so the sink must
    # store all of them. --channels only controls how many the plotter shows.
    sink = PacketSink(
        NUM_CHANNELS, maxlen, csv_header=csv_header,
        process_emg=show_emg, process_imu=show_imu,
    )
    source = SerialSource(ser, parser, sink)
    source.start()

    plotter = LivePlotter(
        sink,
        mode=args.mode,
        port_label=f"{args.port} @ {args.baud} baud",
        fs=args.fs,
        imu_fs=args.imu_fs,
        window=args.window,
        channels=args.channels,
        refresh_ms=args.refresh_ms,
        max_plot_points=args.max_plot_points,
        imu_angle_ylim=args.imu_angle_ylim,
        imu_accel_ylim=args.imu_accel_ylim,
        ylim=args.ylim,
        gain=args.gain,
        vref=args.vref,
        unit=args.unit,
        outfile=outfile,
    )

    try:
        plotter.show()
    finally:
        source.stop()
        source.join(timeout=2)
        ser.close()
        print(f"\nReceived {sink.emg_total_count} emg samples and "
              f"{sink.imu_total_count} imu samples")


if __name__ == "__main__":
    main()
