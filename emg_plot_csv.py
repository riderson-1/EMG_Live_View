#!/usr/bin/env python3
"""
Offline EMG CSV plotter.

Expects CSV columns:

    sample,status1_ok,status2_ok,ch1,...,ch16

The channel values are assumed to be signed ADS1299 ADC codes unless
--unsigned-24bit is used.

Examples
--------
Plot all channels in microvolts:

    python3 emg_plot_csv.py emg_capture.csv --fs 250 --gain 24 --unit uv

Plot channels 1, 2, 3 only:

    python3 emg_plot_csv.py emg_capture.csv --channels 1,2,3

Plot first 10 seconds:

    python3 emg_plot_csv.py emg_capture.csv --xlim 0 10 --unit uv

"""

import argparse
import csv
import sys

import numpy as np
import matplotlib.pyplot as plt

from ads1299 import code_to_volts, code_to_microvolts, twos_complement_24_to_int


def parse_channel_list(s):
    """Parse channel selection string."""
    if s is None or s.lower() == "all":
        return None

    result = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            result.extend(range(a - 1, b))
        else:
            result.append(int(part) - 1)

    return result


def parse_args():
    p = argparse.ArgumentParser(description="Plot stored EMG CSV file")

    p.add_argument("csvfile", help="CSV file to plot")
    p.add_argument("--fs", type=float, default=250.0, help="Sample rate in Hz")
    p.add_argument("--channels", default="all", help="Channels to plot, e.g. 'all', '1,2,3', '1-8'")
    p.add_argument("--gain", type=float, default=1, help="ADS1299 PGA gain")
    p.add_argument("--vref", type=float, default=5, help="ADS1299 reference voltage in volts")
    p.add_argument("--unit", choices=["v", "mv", "uv"], default="uv", help="Display unit")
    p.add_argument("--xlim", nargs=2, type=float, default=None, help="Fixed x-axis limits in seconds")
    p.add_argument("--ylim", nargs=2, type=float, default=None, help="Fixed y-axis limits in selected unit")
    p.add_argument("--offset", type=float, default=0.0, help="Vertical offset between channels (0 = separate subplots)")
    p.add_argument("--bad-status", choices=["keep", "drop", "mark"], default="mark", help="How to handle bad status rows")
    p.add_argument("--unsigned-24bit", action="store_true", help="Interpret as unsigned 24-bit codes")
    p.add_argument("--title", default=None, help="Plot title")
    p.add_argument("--save", default=None, help="Save figure to path (optional)")

    return p.parse_args()


def read_emg_csv(path, unsigned_24bit=False):
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("CSV has no header")

        required = ["sample", "status1_ok", "status2_ok"]
        for col in required:
            if col not in reader.fieldnames:
                raise ValueError(f"Missing required column: {col}")

        ch_names = [name for name in reader.fieldnames if name.startswith("ch")]
        if not ch_names:
            raise ValueError("No channel columns found")

        samples, status1, status2, channels = [], [], [], [[] for _ in ch_names]

        for row_num, row in enumerate(reader, start=2):
            try:
                samples.append(int(row["sample"]))
                status1.append(int(row["status1_ok"]))
                status2.append(int(row["status2_ok"]))

                for i, ch_name in enumerate(ch_names):
                    val = int(row[ch_name])
                    if unsigned_24bit:
                        val = twos_complement_24_to_int(val)
                    channels[i].append(val)

            except ValueError:
                print(f"Warning: skipping malformed row {row_num}", file=sys.stderr)

    return (
        np.asarray(samples, dtype=np.int64),
        np.asarray(status1, dtype=bool),
        np.asarray(status2, dtype=bool),
        ch_names,
        np.asarray(channels, dtype=np.int64)
    )


def convert_units(data_codes, vref, gain, unit):
    volts = code_to_volts(data_codes, vref=vref, gain=gain)
    if unit == "v":
        return volts, "V"
    elif unit == "mv":
        return volts * 1e3, "mV"
    else:  # uv
        return volts * 1e6, "µV"


def main():
    args = parse_args()

    samples, status1, status2, ch_names, data_codes = read_emg_csv(
        args.csvfile,
        unsigned_24bit=args.unsigned_24bit,
    )

    if len(samples) == 0:
        raise RuntimeError("No samples found in CSV")

    selected = parse_channel_list(args.channels)
    if selected is None:
        selected = list(range(len(ch_names)))

    for idx in selected:
        if idx < 0 or idx >= len(ch_names):
            raise ValueError(f"Channel index out of range: ch{idx + 1}")

    data_codes = data_codes[selected, :]
    selected_names = [ch_names[i] for i in selected]

    good = status1 & status2

    if args.bad_status == "drop":
        samples = samples[good]
        data_codes = data_codes[:, good]
        good = np.ones(len(samples), dtype=bool)

    data, unit_label = convert_units(data_codes, vref=args.vref, gain=args.gain, unit=args.unit)
    t = (samples - samples[0]) / args.fs
    n_channels = data.shape[0]

    # --- SIMPLIFIED PLOTTING ---
    if args.offset != 0:
        # Overlaid mode
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(n_channels):
            y = data[i] + i * args.offset
            ax.plot(t, y, lw=0.8, label=selected_names[i])

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Voltage + offset ({unit_label})")
        ax.legend(fontsize=8)

    else:
        # Subplots mode - reasonable figure size
        fig_height = min(12, 2 + n_channels * 0.8)  # Cap at 12 inches
        fig, axes = plt.subplots(n_channels, 1, sharex=True, figsize=(12, fig_height))

        if n_channels == 1:
            axes = [axes]

        for ax, name, y in zip(axes, selected_names, data):
            ax.plot(t, y, lw=0.8)
            ax.set_ylabel(f"{name}\n({unit_label})")

            if args.ylim is not None:
                ax.set_ylim(args.ylim[0], args.ylim[1])

            if args.bad_status == "mark":
                bad_t = t[~good]
                for x in bad_t:
                    ax.axvline(x, color="red", alpha=0.05, lw=0.5)

            lo, hi = np.nanmin(y), np.nanmax(y)
            ax.text(0.005, 0.85, f"{lo:.1f}..{hi:.1f} {unit_label}",
                    transform=ax.transAxes, fontsize=8, family="monospace", va="top")

        axes[-1].set_xlabel("Time (s)")

    # Apply axis limits
    if args.xlim is not None:
        fig.axes[0].set_xlim(args.xlim[0], args.xlim[1])

    # Title and stats
    title = args.title or f"{args.csvfile} | gain={args.gain:g}, vref={args.vref:g}V"
    fig.suptitle(title)

    bad_count = np.count_nonzero(~good)
    total_count = len(good)
    bad_pct = 100.0 * bad_count / max(total_count, 1)
    fig.text(0.01, 0.005, f"samples: {total_count}  bad-status: {bad_count} ({bad_pct:.1f}%)",
             fontsize=9, family="monospace")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"Saved to {args.save}")

    plt.show()  # Interactive window with zoom/pan


if __name__ == "__main__":
    main()