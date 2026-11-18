#!/usr/bin/env python3
"""Read ASKAP MS metadata and prepare a checked observation without submitting jobs."""
import argparse
import csv
import json
import math
import re
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
from astropy.io import fits
from casacore.tables import table


def read_footprint(path):
    centres = {}
    pattern = re.compile(r"^\s*(\d+)\s+\(\s*[-+0-9.eE]+\s+[-+0-9.eE]+\s*\)\s+([^,\s]+)\s*,\s*(\S+)\s*$")
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = pattern.fullmatch(line)
        if match is None:
            raise ValueError(f"Unrecognised footprint row in {path}: {line}")
        beam = int(match[1])
        if beam in centres:
            raise ValueError(f"Duplicate beam {beam} in footprint file {path}")
        ra = Angle(match[2], unit=u.hourangle).wrap_at(360 * u.deg).rad
        dec = Angle(match[3], unit=u.deg).rad
        if not np.isfinite([ra, dec]).all() or abs(dec) > np.pi / 2:
            raise ValueError(f"Invalid coordinates for beam {beam} in {path}")
        centres[beam] = (float(ra), float(dec))
    if set(centres) != set(range(36)) or len(set(centres.values())) != 36:
        raise ValueError(f"Footprint must contain 36 distinct beam centres numbered 0–35: {path}")
    return centres


def check_inputs(args):
    if not args.obs_id.isdigit() or not re.fullmatch(r"[A-Za-z0-9_.-]+", args.field):
        raise ValueError("Use a numeric observation ID and a filename-safe field name.")
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("BIN_SECONDS must be finite and positive.")
    for value in (args.full_beam, args.snapshot_beam):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Restoring beam sizes must be finite and positive.")
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input MS directory: {args.input_dir}")
    if not args.pb.is_file():
        raise FileNotFoundError("PB_IMAGE must point to a readable FITS image.")
    header = fits.getheader(args.pb)
    if header.get("NAXIS", 0) < 2:
        raise ValueError("PB_IMAGE is not an image FITS file.")
    discovered = {}
    for path in sorted(args.input_dir.glob("*.ms")):
        matches = re.findall(r"(?:^|[._-])beam([0-9]+)(?=[._-]|$)", path.name)
        if not matches:
            continue
        if len(matches) != 1:
            raise ValueError(f"Ambiguous beam number in MS filename: {path.name}")
        beam = int(matches[0])
        if beam in discovered:
            raise ValueError(f"More than one MS for beam {beam}: {discovered[beam]} and {path}. Choose an INPUT_DIR containing one input MS per beam.")
        discovered[beam] = path
    if set(discovered) != set(range(36)):
        missing = sorted(set(range(36)) - set(discovered))
        extra = sorted(set(discovered) - set(range(36)))
        raise ValueError(f"This imaging profile requires beams 00–35. Missing: {missing}; unexpected: {extra}.")

    metadata_dirs = {args.work_dir / "metadata", args.work_dir.parent / "metadata", args.input_dir / "metadata", args.input_dir.parent / "metadata", args.input_dir.parent.parent / "metadata"}
    filename = f"footprintOutput-sb{args.obs_id}-src*-{args.field}.txt"
    footprints = sorted({path.resolve() for folder in metadata_dirs if folder.is_dir() for path in folder.glob(filename) if path.is_file()})
    if len(footprints) != 1:
        raise ValueError(f"Expected one pipeline {filename}; found {len(footprints)}: {footprints}. Keep the original footprint file in {args.work_dir / 'metadata'}. FIELD/PHASE_DIR is not a substitute for individual beam centres.")
    footprint = footprints[0]
    beam_centres = read_footprint(footprint)

    rows = []
    field_centre = None
    antenna_names = None
    frequency_grid = None
    for beam, path in sorted(discovered.items()):
        if any(character.isspace() or character == "'" for character in str(path)):
            raise ValueError(f"MS paths containing whitespace or single quotes are not supported: {path}")
        with table(str(path), readonly=True, ack=False) as ms:
            required = {"TIME", "INTERVAL", "UVW", "DATA", "FLAG", "FLAG_ROW", "ANTENNA1", "ANTENNA2", "FIELD_ID", "DATA_DESC_ID"}
            if ms.nrows() == 0 or not required.issubset(ms.colnames()):
                raise ValueError(f"Empty MS or missing required columns: {path}")
            with ms.query(columns="gmin(TIME) AS LO, gmax(TIME) AS HI", style="Python") as limits:
                start, end = float(limits.getcell("LO", 0)), float(limits.getcell("HI", 0))
            if not all(map(math.isfinite, (start, end))) or end < start:
                raise ValueError(f"Invalid TIME coverage: {path}")
        with table(str(path / "ANTENNA"), readonly=True, ack=False) as antennas:
            names = list(map(str, antennas.getcol("NAME")))
            if len(names) != 36:
                raise ValueError(f"This imaging profile requires 36 antennas, found {len(names)}: {path}")
            if antenna_names is not None and names != antenna_names:
                raise ValueError(f"Antenna ordering differs between input beams: {path}")
            antenna_names = names
        with table(str(path / "SPECTRAL_WINDOW"), readonly=True, ack=False) as spectral:
            if spectral.nrows() != 1:
                raise ValueError(f"This profile requires a single spectral window: {path}")
            channels = int(spectral.getcell("NUM_CHAN", 0))
            frequencies = np.asarray(spectral.getcell("CHAN_FREQ", 0), dtype=float)
            if channels != 288 or frequencies.shape != (channels,) or not np.isfinite(frequencies).all():
                raise ValueError(f"This imaging profile requires 288 frequency channels, found {channels}: {path}")
            if frequency_grid is not None and not np.allclose(frequencies, frequency_grid, atol=1, rtol=0):
                raise ValueError(f"Frequency channels differ between input beams: {path}")
            frequency_grid = frequencies
        with table(str(path / "POLARIZATION"), readonly=True, ack=False) as polarisation:
            if polarisation.nrows() != 1 or list(polarisation.getcell("CORR_TYPE", 0)) != [9, 10, 11, 12]:
                raise ValueError(f"IQUV imaging requires XX, XY, YX, YY correlations in that order: {path}")
        with table(str(path / "DATA_DESCRIPTION"), readonly=True, ack=False) as description:
            if description.nrows() != 1 or int(description.getcell("SPECTRAL_WINDOW_ID", 0)) != 0 or int(description.getcell("POLARIZATION_ID", 0)) != 0:
                raise ValueError(f"Unsupported DATA_DESCRIPTION mapping: {path}")
        with table(str(path / "FIELD"), readonly=True, ack=False) as field:
            if field.nrows() != 1 or int(field.getcell("NUM_POLY", 0)) != 0:
                raise ValueError(f"Expected a single, constant FIELD phase centre: {path}")
            if field.getcolkeyword("PHASE_DIR", "MEASINFO").get("Ref") != "J2000":
                raise ValueError(f"Expected J2000 FIELD/PHASE_DIR metadata: {path}")
            direction = np.asarray(field.getcell("PHASE_DIR", 0), dtype=float).squeeze()
            if direction.shape != (2,) or not np.isfinite(direction).all() or abs(direction[1]) > np.pi / 2:
                raise ValueError(f"Unexpected FIELD/PHASE_DIR: {path}")
        with table(str(path / "POINTING"), readonly=True, ack=False) as pointing:
            required = {"TIME", "ANTENNA_ID", "TRACKING", "DIRECTION", "NUM_POLY", "POLANGLE", "AZIMUTH", "ELEVATION"}
            if pointing.nrows() == 0 or not required.issubset(pointing.colnames()):
                raise ValueError(f"POINTING is empty or missing the ASKAP angle columns: {path}")
            if pointing.getcolkeyword("DIRECTION", "MEASINFO").get("Ref") != "J2000":
                raise ValueError(f"Expected J2000 POINTING directions: {path}")
            if beam == 0:
                centre_column = "TARGET" if "TARGET" in pointing.colnames() else "DIRECTION"
                if pointing.getcolkeyword(centre_column, "MEASINFO").get("Ref") != "J2000":
                    raise ValueError("Cannot read a J2000 observation centre from POINTING metadata.")
                tracked = pointing.getcol("TRACKING").astype(bool)
                centres = np.asarray(pointing.getcol(centre_column), dtype=float)
                if centres.ndim < 2 or np.prod(centres.shape[1:]) != 2 or np.any(pointing.getcol("NUM_POLY")[tracked] != 0):
                    raise ValueError("Expected one constant RA/Dec direction per POINTING record.")
                centres = centres.reshape(len(centres), 2)[tracked]
                centres = centres[np.isfinite(centres).all(axis=1)]
                if not len(centres):
                    raise ValueError("No finite tracked POINTING records for the observation centre.")
                ra = float(np.angle(np.mean(np.exp(1j * centres[:, 0]))) % (2 * np.pi))
                dec = float(np.median(centres[:, 1]))
                east = np.angle(np.exp(1j * (centres[:, 0] - ra))) * np.cos(dec)
                separation = np.hypot(east, centres[:, 1] - dec)
                if np.max(separation) > np.deg2rad(1 / 60):
                    raise ValueError("Tracked POINTING metadata span more than one arcminute; cannot safely infer a single field centre.")
                field_centre = (ra, dec, centre_column)
        rows.append({"beam": f"beam{beam:02d}", "ms": str(path), "start": start, "end": end, "ra_rad": beam_centres[beam][0], "dec_rad": beam_centres[beam][1], "phase_centre_ra_rad": float(direction[0]), "phase_centre_dec_rad": float(direction[1])})
        print(f"  beam{beam:02d}: 36 antennas, {channels} channels; metadata OK", flush=True)

    directions = {}
    for row in rows:
        delta_ra = np.angle(np.exp(1j * (row["ra_rad"] - field_centre[0])))
        cosine = np.sin(row["dec_rad"]) * np.sin(field_centre[1]) + np.cos(row["dec_rad"]) * np.cos(field_centre[1]) * np.cos(delta_ra)
        if np.arccos(np.clip(cosine, -1, 1)) > np.deg2rad(10):
            raise ValueError(f"Footprint beam {row['beam']} is more than 10 degrees from the tracked field centre; check that the footprint belongs to these MSs.")
        ra = Angle(row["ra_rad"] * u.rad).wrap_at(360 * u.deg).to_string(unit=u.hourangle, sep="hms", precision=9, pad=True).removesuffix("s")
        dec = Angle(row["dec_rad"] * u.rad).to_string(unit=u.deg, sep=".", precision=8, pad=True, alwayssign=True)
        directions[row["beam"]] = f"[{ra}, {dec}, J2000]"

    start = min(row["start"] for row in rows)
    end = max(row["end"] for row in rows)
    count = math.floor((end - start) / args.seconds) + 1
    windows = [{"folder": f"bin-{index:03d}", "start_mjd_seconds_inclusive": start + index * args.seconds, "end_mjd_seconds_exclusive": start + (index + 1) * args.seconds} for index in range(count)]
    covered = [index for index, window in enumerate(windows) if float(window["end_mjd_seconds_exclusive"]) <= end]
    if args.last_bin is None:
        if not covered:
            raise ValueError("No complete snapshot interval; choose shorter BIN_SECONDS.")
        last_bin = covered[-1]
    else:
        last_bin = args.last_bin
    if last_bin < 0 or last_bin >= len(windows) or float(windows[last_bin]["start_mjd_seconds_inclusive"]) > end:
        raise ValueError("LAST_BIN is outside the observation's time coverage.")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "pointings.txt").write_text("".join(f"{row['beam']} {directions[row['beam']]}\n" for row in rows))
    with (args.output / "intervals.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["folder", "start_mjd_seconds_inclusive", "end_mjd_seconds_exclusive"])
        writer.writeheader()
        writer.writerows(windows)
    field_ra = Angle(field_centre[0] * u.rad).to_string(unit=u.hourangle, sep="hms", precision=9, pad=True)
    field_dec = Angle(field_centre[1] * u.rad).to_string(unit=u.deg, sep="dms", precision=8, pad=True, alwayssign=True)
    (args.output / "measurement-sets.txt").write_text("".join(f"{row['beam']} {row['ms']}\n" for row in rows))
    (args.output / "dataset.sh").write_text(f'NCHAN=288\nLAST_BIN={last_bin}\nFIELD_RA="{field_ra}"\nFIELD_DEC="{field_dec}"\n')
    report = {"profile": "ASKAP 36 beams / 36 antennas / 288 channels / XX XY YX YY", "input_dir": str(args.input_dir), "primary_beam": str(args.pb), "beam_directions": "pipeline footprint metadata", "footprint_file": str(footprint), "field_centre": {"ra": field_ra, "dec": field_dec, "source": f"tracked POINTING/{field_centre[2]}"}, "first_time_mjd_seconds": start, "last_time_mjd_seconds": end, "last_bin": last_bin, "beams": rows}
    (args.output / "input-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Field centre from POINTING/{field_centre[2]}: {field_ra}, {field_dec}")
    print(f"36 distinct beam centres from: {footprint}")
    print(f"Input check passed. Selected bins: bin-000 through bin-{last_bin:03d}.")
    print("The beam file opened.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs-id", required=True)
    parser.add_argument("--field", required=True)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--pb", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=900)
    parser.add_argument("--last-bin", type=int, help="Default: last full interval")
    parser.add_argument("--full-beam", type=float, required=True)
    parser.add_argument("--snapshot-beam", type=float, required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        check_inputs(args)
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        parser.exit(1, f"Input check failed: {error}\n")
