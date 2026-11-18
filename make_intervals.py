#!/usr/bin/env python3
"""Create shared time intervals for all beam measurement sets."""
import argparse
import csv
from math import floor
from pathlib import Path

from casacore.tables import table


def make_intervals(input_dir, prefix, output, seconds=900, beams=36, ms_pattern=None):
    ms_pattern = ms_pattern or f"{prefix}.{{beam}}_averaged_cal.leakage.ms"
    if ms_pattern.count("{beam}") != 1 or Path(ms_pattern).name != ms_pattern:
        raise ValueError("The MS filename pattern must contain exactly one {beam} placeholder.")
    bounds = []
    for beam in range(beams):
        path = input_dir / ms_pattern.replace("{beam}", f"beam{beam:02d}")
        with table(str(path), readonly=True, ack=False) as ms:
            with ms.query(columns="gmin(TIME) AS TMIN, gmax(TIME) AS TMAX", style="Python") as limits:
                bounds.append((float(limits.getcell("TMIN", 0)), float(limits.getcell("TMAX", 0))))

    start = min(lo for lo, hi in bounds)
    count = floor((max(hi for lo, hi in bounds) - start) / seconds) + 1
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["folder", "start_mjd_seconds_inclusive", "end_mjd_seconds_exclusive"])
        for index in range(count):
            writer.writerow([f"bin-{index:03d}", start + index * seconds, start + (index + 1) * seconds])
    print(f"Wrote {count} shared intervals to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--ms-pattern", help="Optional filename pattern containing {beam}")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=900)
    parser.add_argument("--beams", type=int, default=36)
    args = parser.parse_args()
    if args.seconds <= 0 or args.beams < 1:
        parser.error("seconds and beams must be positive")
    make_intervals(args.input_dir, args.prefix, args.output, args.seconds, args.beams, args.ms_pattern)
