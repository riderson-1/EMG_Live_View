#!/usr/bin/env python3
"""
Clean an EMG capture CSV by removing corrupted / duplicate / out-of-order rows.

Expects CSV columns:

    sample,status1_ok,status2_ok,ch1,...,ch16

IMPORTANT: this script does NOT drop rows just because of a large gap in the
'sample' counter. A big jump (e.g. samples simply weren't captured for a
while) is normal and expected -- it just means missing data, which the
plotting script already handles correctly by leaving a blank gap (see
insert_gap_breaks() in emg_plot_csv.py). Dropping those rows would be wrong
and, worse, would desynchronize 'sample' from real elapsed time.

Only two kinds of genuinely bad rows are detected and dropped, using the
*raw* sample counter column (no conversion needed, works regardless of
gain/vref/unit):

1. Non-increasing sample number (duplicate or backward jump)
   -> almost always a re-transmitted / out-of-order row.

2. A "spike and snap-back": the sample counter jumps forward by an
   implausibly large amount on one row, and then the *very next* row jumps
   back down to close to where the sequence should have continued from.
   This is the signature of a corrupted row where two CSV rows got glued
   together on the wire (e.g. a dropped delimiter/newline during
   UART/USB-CDC streaming), producing a garbled counter value like
   "99199217" instead of "99199". Only the single glued row is dropped.

A large jump that is NOT followed by a snap-back (i.e. the counter keeps
counting up normally from the new value) is treated as a genuine gap and
is kept as-is -- nothing is removed, no NaNs are inserted here. Gap
handling for plotting stays entirely in emg_plot_csv.py's
insert_gap_breaks(), which works directly off the (now corruption-free)
'sample' column.

Rows are evaluated against the last row that was *kept* (not simply the
previous row in the file), so one dropped row can't cause a cascade of
otherwise-good rows to also get dropped.

Examples
--------
Basic cleanup, auto-named output:

    python3 emg_clean_csv.py emg_capture.csv

Explicit output path and a tighter jump threshold:

    python3 emg_clean_csv.py emg_capture.csv -o emg_capture_clean.csv --max-sample-jump 5000

Also drop rows where either status flag is bad:

    python3 emg_clean_csv.py emg_capture.csv --drop-bad-status

Just report what would be removed, without writing a file:

    python3 emg_clean_csv.py emg_capture.csv --dry-run
"""

import argparse
import csv
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Clean an EMG capture CSV")

    p.add_argument("csvfile", help="Input CSV file to clean")
    p.add_argument("-o", "--output", default=None,
                    help="Output CSV path (default: <input>_clean.csv)")
    p.add_argument("--max-sample-jump", type=int, default=10000,
                    help="Forward jump in the 'sample' counter larger than this "
                         "is only treated as corruption (and dropped) if the very "
                         "next row snaps back down close to the expected value. "
                         "Otherwise it's treated as a genuine gap and kept. "
                         "(default: 10000)")
    p.add_argument("--drop-bad-status", action="store_true",
                    help="Also drop rows where status1_ok or status2_ok is 0/false "
                         "(default: keep them, matching the plotting script's 'mark' mode)")
    p.add_argument("--dry-run", action="store_true",
                    help="Only print the cleaning report, do not write an output file")

    return p.parse_args()


def clean_csv(path, max_sample_jump, drop_bad_status):
    """
    Read `path`, decide which rows to keep, and return:

        fieldnames, kept_rows (list of dict), report (dict of stats)

    Row values are kept as raw strings (not converted to int/float) so the
    output CSV is a byte-for-byte-equivalent, just with bad rows removed.
    """
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("CSV has no header")

        required = ["sample", "status1_ok", "status2_ok"]
        for col in required:
            if col not in reader.fieldnames:
                raise ValueError(f"Missing required column: {col}")

        fieldnames = reader.fieldnames

        # First pass: parse every row, tracking malformed ones separately.
        # We need random access with one-row lookahead, so this can't be
        # done in a single streaming pass.
        parsed = []  # list of (row_num, row_dict, sample, status1, status2)
        n_malformed = 0

        for row_num, row in enumerate(reader, start=2):
            try:
                sample = int(row["sample"])
                status1 = int(row["status1_ok"])
                status2 = int(row["status2_ok"])
            except (ValueError, KeyError):
                print(f"Warning: skipping malformed row {row_num}", file=sys.stderr)
                n_malformed += 1
                continue

            parsed.append((row_num, row, sample, status1, status2))

    n_total = n_malformed + len(parsed)
    kept_rows = []
    n_nonincreasing = 0
    n_corrupt = 0
    n_gap = 0
    n_bad_status = 0

    last_good_sample = None
    i = 0

    while i < len(parsed):
        row_num, row, sample, status1, status2 = parsed[i]

        if last_good_sample is None:
            # First row in the file: always accepted as the starting point.
            keep = True
        else:
            diff = sample - last_good_sample

            if diff <= 0:
                # Duplicate or out-of-order sample counter -> drop.
                n_nonincreasing += 1
                i += 1
                continue

            elif diff > max_sample_jump:
                # Large forward jump. Check the *next* row (if any) to tell
                # a corrupted/merged row apart from a genuine gap: if the
                # sequence snaps back down close to where it should have
                # continued from, this row is the corrupted one.
                if i + 1 < len(parsed):
                    next_sample = parsed[i + 1][2]
                    next_diff_from_last_good = next_sample - last_good_sample

                    if 0 < next_diff_from_last_good <= max_sample_jump:
                        # Snap-back detected -> this row is corrupted, drop it.
                        # last_good_sample is left unchanged so the next row
                        # is compared against the same reference point.
                        n_corrupt += 1
                        i += 1
                        continue

                # No snap-back (or no next row) -> genuine gap, keep the row.
                n_gap += 1
                keep = True

            else:
                # Normal increment.
                keep = True

        if keep:
            if drop_bad_status and not (status1 and status2):
                n_bad_status += 1
            else:
                kept_rows.append(row)

            last_good_sample = sample

        i += 1

    report = {
        "rows_read": n_total,
        "rows_kept": len(kept_rows),
        "dropped_malformed": n_malformed,
        "dropped_nonincreasing": n_nonincreasing,
        "dropped_corrupt": n_corrupt,
        "gaps_kept": n_gap,
        "dropped_bad_status": n_bad_status,
    }

    return fieldnames, kept_rows, report


def print_report(path, report):
    dropped_total = report["rows_read"] - report["rows_kept"]
    pct = 100.0 * dropped_total / max(report["rows_read"], 1)

    print(f"--- {path} ---")
    print(f"  rows read:                 {report['rows_read']}")
    print(f"  rows kept:                 {report['rows_kept']}")
    print(f"  dropped (malformed):       {report['dropped_malformed']}")
    print(f"  dropped (non-increasing):  {report['dropped_nonincreasing']}")
    print(f"  dropped (corrupted/merged rows): {report['dropped_corrupt']}")
    print(f"  genuine gaps kept as-is:   {report['gaps_kept']}")
    print(f"  dropped (bad status):      {report['dropped_bad_status']}")
    print(f"  total dropped:             {dropped_total} ({pct:.3f}%)")


def main():
    args = parse_args()

    fieldnames, kept_rows, report = clean_csv(
        args.csvfile,
        max_sample_jump=args.max_sample_jump,
        drop_bad_status=args.drop_bad_status,
    )

    print_report(args.csvfile, report)

    if args.dry_run:
        print("\nDry run: no output file written.")
        return

    if args.output:
        out_path = args.output
    else:
        base, ext = os.path.splitext(args.csvfile)
        out_path = f"{base}_clean{ext or '.csv'}"

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    print(f"\nWrote cleaned CSV: {out_path}")


if __name__ == "__main__":
    main()