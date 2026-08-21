#!/usr/bin/env python3
"""
Live EMG plotter - Bluetooth Low Energy (NUS) transport.

Entry point that wires the shared sokosti pipeline together for a BLE
connection to the Nordic UART Service. The plotting behavior is identical
to the USB variant (emg_live_plot.py); only the byte source differs.

BLE connection/throughput/stability telemetry is printed to the terminal
(and optionally a log file), separate from the plot window.

Usage:
    python emg_live_ble.py --name Sokosti_BLE
    python emg_live_ble.py --address AA:BB:CC:DD:EE:FF
"""

import argparse
import asyncio
import csv
import os
import threading
from datetime import datetime

from sokosti import FrameParser, PacketSink, LivePlotter
from sokosti.sources import BleSource

# Create captures folder if it doesn't exist
os.makedirs("captures", exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Live EMG plot from ADS1299 binary packets (BLE)")
    p.add_argument("--name", default="Sokosti_BLE",
                   help="BLE advertised name to scan for (default: Sokosti_BLE)")
    p.add_argument("--address", default=None,
                   help="Connect directly by MAC/UUID, skipping the scan on every (re)connect")
    p.add_argument("--retry-delay", type=float, default=3.0,
                   help="Seconds to wait before rescanning/reconnecting (default: 3)")
    p.add_argument("--scan-timeout", type=float, default=10.0,
                   help="Seconds to scan for the device (default: 10)")
    p.add_argument("--log-file", default=None,
                   help="Optional path to append BLE telemetry/connection events")

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
        "captures", f"sokosti_ble_capture_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    f = open(outfile, "w", newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow(
        ["sample", "status1_ok", "status2_ok"]
        + [f"ch{i+1}" for i in range(args.channels)]
        + ["roll", "pitch", "yaw", "accel_x", "accel_y", "accel_z"]
    )

    parser = FrameParser()
    sink = PacketSink(
        args.channels, maxlen, csv_writer,
        process_emg=show_emg, process_imu=show_imu,
    )
    source = BleSource(
        parser, sink,
        name=args.name,
        address=args.address,
        retry_delay=args.retry_delay,
        scan_timeout=args.scan_timeout,
        log_file=args.log_file,
    )

    plotter = LivePlotter(
        sink,
        mode=args.mode,
        port_label=f"BLE {args.name or args.address}",
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

    # Run the BLE event loop on a background thread so the matplotlib
    # plot can run on the main thread (matplotlib must own the GUI loop).
    ble_thread = threading.Thread(target=lambda: asyncio.run(source.run()), daemon=True)
    ble_thread.start()

    try:
        plotter.show()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        f.close()
        print(f"\nSaved {sink.emg_total_count} emg samples and "
              f"{sink.imu_total_count} imu samples to {outfile}")


if __name__ == "__main__":
    main()