#!/usr/bin/env python3
"""Export measurement coordinates and raw roll, azimuth and elevation from an MS."""
import argparse
import csv
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
from casacore.tables import table, taql


def export_pointing(obs_id, ms, intervals, beam_directions, output):
    with intervals.open(newline="") as handle:
        bins = list(csv.DictReader(handle))
    starts = np.array([float(row["start_mjd_seconds_inclusive"]) for row in bins])
    ends = np.array([float(row["end_mjd_seconds_exclusive"]) for row in bins])
    origin = starts.min()

    beam_ra, beam_dec = np.full(36, np.nan), np.full(36, np.nan)
    for line in beam_directions.read_text().splitlines():
        beam, direction = line.split(maxsplit=1)
        ra, dec, frame = [part.strip() for part in direction.strip("[]").split(",")]
        if frame != "J2000":
            raise ValueError("Expected J2000 beam directions.")
        if "d" not in dec and len(dec.split(".")) >= 3:
            degrees, minutes, seconds = dec.split(".", 2)
            dec = f"{degrees}d{minutes}m{seconds}s"
        index = int(beam.removeprefix("beam"))
        beam_ra[index] = Angle(ra, unit=u.hourangle).deg
        beam_dec[index] = Angle(dec, unit=u.deg).deg
    if not np.isfinite(beam_ra).all() or not np.isfinite(beam_dec).all():
        raise ValueError("Expected one direction for every beam 00–35.")

    with table(str(ms / "ANTENNA"), readonly=True, ack=False) as antenna:
        names = np.asarray(antenna.getcol("NAME"))
        positions = antenna.getcol("POSITION")
    if len(names) != 36:
        raise ValueError("This ASKAP analysis expects 36 antennas.")
    if "'" in str(ms):
        raise ValueError("Single quotes in MS paths are unsupported.")
    with taql(f"SELECT DISTINCT TIME FROM '{ms}' ORDERBY TIME") as times:
        integration_times = times.getcol("TIME")

    with table(str(ms / "POINTING"), readonly=True, ack=False) as pointing:
        time = pointing.getcol("TIME")
        ids = pointing.getcol("ANTENNA_ID")
        tracking = pointing.getcol("TRACKING").astype(bool)
        poly = pointing.getcol("NUM_POLY")
        direction = pointing.getcol("DIRECTION")
        roll = pointing.getcol("POLANGLE")
        azimuth = pointing.getcol("AZIMUTH")
        elevation = pointing.getcol("ELEVATION")
        frame = pointing.getcolkeyword("DIRECTION", "MEASINFO").get("Ref", "")
    if frame != "J2000" or np.any(poly[tracking] != 0):
        raise ValueError("Expected constant J2000 POINTING directions for tracked records.")
    if direction.shape[1:] != (1, 2):
        raise ValueError(f"Unexpected POINTING DIRECTION shape: {direction.shape}")
    tracked_ra = np.degrees(direction[tracking, 0, 0])
    tracked_dec = np.degrees(direction[tracking, 0, 1])

    output.mkdir(parents=True, exist_ok=True)
    coordinate_file = output / f"pointing-{obs_id}.npz"
    np.savez_compressed(coordinate_file, sbid=np.array(obs_id), direction_frame=np.array("FK5 J2000"), time_scale=np.array("UTC"), bin_names=np.array([row["folder"] for row in bins]), bin_start=starts, bin_end=ends, antenna_names=names, antenna_xyz_m=positions, integration_time_s=integration_times, beam_ra_deg=beam_ra, beam_dec_deg=beam_dec, dish_time_s=time[tracking], dish_antenna=ids[tracking], dish_ra_deg=tracked_ra, dish_dec_deg=tracked_dec)

    # Preserve every raw angle record, including TRACKING=false. Do not unwrap angles.
    order = np.lexsort((time, ids))
    angle_file = output / f"SB{obs_id}_pointing_angles.npz"
    np.savez_compressed(angle_file, obs_id=obs_id, source_ms=str(ms), observation_start_mjd_seconds=origin, raw_elapsed_seconds=time[order] - origin, raw_hours=(time[order] - origin) / 3600, raw_antenna_ids=ids[order], raw_polangle=roll[order], raw_azimuth=azimuth[order], raw_elevation=elevation[order], raw_tracking=tracking[order], angle_units="Raw stored POINTING values; no angular folding or unwrapping applied")
    print(f"Saved {coordinate_file}")
    print(f"Saved {angle_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs-id", required=True)
    parser.add_argument("--ms", required=True, type=Path, help="One input beam MS containing ANTENNA and POINTING")
    parser.add_argument("--intervals", required=True, type=Path)
    parser.add_argument("--beam-directions", required=True, type=Path, help="The workflow's pointings.txt")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export_pointing(args.obs_id, args.ms, args.intervals, args.beam_directions, args.output)
