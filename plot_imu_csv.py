#!/usr/bin/env python3
"""Plot IMU axes (roll, pitch, yaw) and accelerometer (accel_x/y/z) from a Sokosti capture CSV."""

import argparse
import sys

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Plot IMU axes and accelerometer data from a CSV capture.")
    parser.add_argument("csv", nargs="?", help="Path to the CSV file (defaults to the most recent capture).")
    args = parser.parse_args()

    path = args.csv
    # if path is None:
    #     import glob
    #     import os
    #     files = sorted(glob.glob(os.path.join("captures", "sokosti_capture_*.csv")))
    #     if not files:
    #         print("No CSV file given and none found in captures/.", file=sys.stderr)
    #         sys.exit(1)
    #     path = files[-1]
    #     print(f"Using most recent capture: {path}")

    df = pd.read_csv(path)

    # IMU columns may be empty for some rows -> coerce to numeric and drop NaNs
    imu_cols = ["roll", "pitch", "yaw", "accel_x", "accel_y", "accel_z"]
    for col in imu_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Use sample as x-axis (or row index if missing)
    x = df["sample"] if "sample" in df.columns else df.index

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: orientation (roll, pitch, yaw) in degrees
    axes[0].plot(x, df["roll"], label="roll", linewidth=0.8)
    axes[0].plot(x, df["pitch"], label="pitch", linewidth=0.8)
    axes[0].plot(x, df["yaw"], label="yaw", linewidth=0.8)
    axes[0].set_ylabel("Degrees")
    axes[0].set_title(f"Orientation (roll / pitch / yaw) — {path}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bottom: acceleration (accel_x/y/z)
    axes[1].plot(x, df["accel_x"], label="accel_x", linewidth=0.8)
    axes[1].plot(x, df["accel_y"], label="accel_y", linewidth=0.8)
    axes[1].plot(x, df["accel_z"], label="accel_z", linewidth=0.8)
    axes[1].set_ylabel("Acceleration")
    axes[1].set_xlabel("Sample")
    axes[1].set_title("Accelerometer (accel_x / accel_y / accel_z)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.suptitle(f"FFT - {path}", fontsize=12)
    plt.show()


if __name__ == "__main__":
    main()
