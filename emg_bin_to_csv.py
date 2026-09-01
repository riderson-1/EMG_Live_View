#!/usr/bin/env python3
"""
Convert an on-board SD-card binary capture into a CSV identical in format
to the live BLE/USB captures.

The SD writer (threads.cpp) interleaves two record types into one file:

    EmgSamplePacket  -> 8 x SamplePacket (57 B each) = 456 B, sync 0xAA55
    ImuSamplePacket  -> 19 B, sync 0xBB66

Every packet still carries its own sync marker, so we can feed the whole
file through the same FrameParser -> PacketSink pipeline used by the live
plotters. The CSV layout (sample, status, ch1..ch16, roll/pitch/yaw,
accel_x/y/z) is therefore identical to the BLE captures.

Usage:
    python emg_bin_to_csv.py on_board_captures/session_0001.bin
    python emg_bin_to_csv.py session_0001.bin -o out.csv
"""

import argparse
import csv
import os
from datetime import datetime

from sokosti import FrameParser, PacketSink


def parse_args():
    p = argparse.ArgumentParser(
        description="Parse an SD-card binary capture into a BLE-style CSV"
    )
    p.add_argument("binfile", help="Path to the on-board .bin capture file")
    p.add_argument("-o", "--outfile", default=None,
                   help="Output CSV path (default: captures/sokosti_sd_capture_<ts>.csv)")
    p.add_argument("--channels", type=int, default=16, help="Number of EMG channels")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.binfile):
        raise SystemExit(f"File not found: {args.binfile}")

    outfile = args.outfile or os.path.join(
        "captures", f"sokosti_sd_capture_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)

    with open(outfile, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            ["sample", "status1_ok", "status2_ok"]
            + [f"ch{i+1}" for i in range(args.channels)]
            + ["roll", "pitch", "yaw", "accel_x", "accel_y", "accel_z"]
        )

        parser = FrameParser()
        sink = PacketSink(args.channels, maxlen=1, csv_writer=csv_writer,
                          process_emg=True, process_imu=True)

        total_bytes = os.path.getsize(args.binfile)
        print(f"Parsing {args.binfile} ({total_bytes} bytes) -> {outfile}")

        with open(args.binfile, "rb") as fh:
            while True:
                chunk = fh.read(65536)
                if not chunk:
                    break
                for kind, packet in parser.feed(chunk):
                    if kind == "emg":
                        sink.add_emg(packet)
                    else:
                        sink.add_imu(packet)

    print(f"Done: {sink.emg_total_count} emg samples, "
          f"{sink.imu_total_count} imu packets, "
          f"{parser.error_count} checksum/resync errors")
    print(f"Saved to {outfile}")


if __name__ == "__main__":
    main()