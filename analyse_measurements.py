#!/usr/bin/env python3
"""Measure completed full-track and snapshot products.

Run python analyse_measurements.py --help for inputs and examples.
"""
import argparse
import gc
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from workflow_inputs import resolve_inputs

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import astropy
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, FK5, SkyCoord, search_around_sky
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.utils import iers
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

MIN_SNR = 10.0
ISOLATION_ARCSEC = 45.0
RACS_FILE = None
TAQL = None
LOG = logging.getLogger("time_analysis")


@lru_cache(maxsize=1)
def read_racs(path):
    return Table.read(path)


def median_rms(rms_file, plot=False, field = None, bins = 100):
    rms_file = Path(rms_file)

    with fits.open(rms_file, memmap=True) as hdul:
        hdu = next(item for item in hdul if item.data is not None)
        rms = np.squeeze(np.asarray(hdu.data, dtype=float))
        header = hdu.header.copy()

    if rms.ndim != 2:
        raise ValueError(f"Expected a 2D RMS image, got {rms.shape}")

    rms[~np.isfinite(rms) | (rms <= 0)] = np.nan
    unit = header.get("BUNIT", "Jy/beam")
    u.Unit(unit)
    if not u.Unit(unit).is_equivalent(u.Jy / u.beam):
        raise ValueError(f"Unsupported RMS unit: {unit}")
    
    rms *= u.Unit(unit).to(u.mJy / u.beam)
    unit = "mJy/beam"
    
    values = rms[np.isfinite(rms)]

    if values.size == 0:
        raise ValueError(f"No valid RMS pixels in {rms_file}")

    median = float(np.median(values))


    return median


def measure_dynamic_range(catalogue_file, median_rms, reference_position=None, match_radius_arcsec=2):
    catalogue_file = Path(catalogue_file)
    columns = None
    sources = []

    for line in catalogue_file.read_text().splitlines():
        if line.lstrip().startswith("#"):
            header = line.lstrip("#").split()

            if "ra_deg_cont" in header and "flux_peak" in header:
                columns = header

            continue

        if columns is None:
            continue

        row = dict(zip(columns, line.split()))

        try:
            source = {
                "name": row.get("component_name", row.get("component_id", "source")),
                "ra": float(row["ra_deg_cont"]),
                "dec": float(row["dec_deg_cont"]),
                "peak": float(row["flux_peak"]),
            }
        except (KeyError, ValueError):
            continue

        if math.isfinite(source["peak"]) and source["peak"] > 0:
            sources.append(source)

    if not sources:
        raise ValueError(f"No usable sources in {catalogue_file}")

    if not median_rms > 0:
        raise ValueError("median_rms must be positive")

    positions = SkyCoord(
        [source["ra"] for source in sources] * u.deg,
        [source["dec"] for source in sources] * u.deg,
    )

    if reference_position is None:
        index = max(range(len(sources)), key=lambda i: sources[i]["peak"])
        separation_arcsec = None
    else:
        separations = reference_position.separation(positions)
        index = int(separations.argmin())
        separation_arcsec = separations[index].arcsec

        if separation_arcsec > match_radius_arcsec:
            raise ValueError(f"No source found within {match_radius_arcsec} arcsec")

    source = sources[index]
    return {
        "name": source["name"],
        "position": positions[index],
        "peak": source["peak"],
        "dynamic_range": source["peak"] / median_rms,
        "separation_arcsec": separation_arcsec,
    }


def isolated(coords):
    first, second, _, _ = search_around_sky(coords, coords, ISOLATION_ARCSEC * u.arcsec)
    neighbors = np.bincount(first[first != second], minlength=len(coords))
    return neighbors == 0


def prepare_position_catalogues(selavy_file):
    selavy_file = Path(selavy_file)
    lines = selavy_file.read_text().splitlines()
    header = next(line[1:].split() for line in lines if line.startswith("#") and "component_id" in line)
    all_sources = []

    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        row = dict(zip(header, line.split()))

        try:
            peak = float(row["flux_peak"])
            rms = float(row["rms_image"])

            if rms <= 0:
                continue

            all_sources.append({
                "name": row["component_name"],
                "ra": float(row["ra_deg_cont"]),
                "dec": float(row["dec_deg_cont"]),
                "peak": peak,
                "flux": float(row["flux_int"]),
                "rms": rms,
                "snr": peak / rms,
                "siblings": row["has_siblings"] == "1",
            })
        except (KeyError, ValueError):
            continue

    if not all_sources:
        raise ValueError(f"No usable Selavy sources in {selavy_file}")

    all_coords = SkyCoord(
        [source["ra"] for source in all_sources] * u.deg,
        [source["dec"] for source in all_sources] * u.deg,
    )
    keep = np.array([
        source["snr"] >= MIN_SNR and not source["siblings"]
        for source in all_sources
    ])
    keep &= isolated(all_coords)
    selavy_sources = [source for source, selected in zip(all_sources, keep) if selected]
    selavy_coords = all_coords[keep]

    if not selavy_sources:
        raise ValueError("No Selavy sources pass the SNR and isolation cuts")

    racs = read_racs(str(RACS_FILE))
    ra = np.asarray(racs["RA"], dtype=float)
    dec = np.asarray(racs["Dec"], dtype=float)
    pad = ISOLATION_ARCSEC / 3600
    ra_pad = pad / np.cos(np.deg2rad(np.max(np.abs(selavy_coords.dec.deg))))
    nearby = (
        (dec >= selavy_coords.dec.deg.min() - pad)
        & (dec <= selavy_coords.dec.deg.max() + pad)
        & (ra >= selavy_coords.ra.deg.min() - ra_pad)
        & (ra <= selavy_coords.ra.deg.max() + ra_pad)
    )
    racs_indices = np.flatnonzero(nearby)
    nearby_coords = SkyCoord(ra[nearby] * u.deg, dec[nearby] * u.deg)
    racs_keep = isolated(nearby_coords)
    racs_coords = nearby_coords[racs_keep]
    racs_sources = [
        {
            "name": str(racs["Name"][index]),
            "ra": ra[index],
            "dec": dec[index],
            "flux": float(racs["Total_flux"][index]),
        }
        for index, selected in zip(racs_indices, racs_keep)
        if selected
    ]

    return {
        "racs": racs_coords,
        "racs_sources": racs_sources,
        "selavy": selavy_coords,
        "selavy_sources": selavy_sources,
        "catalogue_name": selavy_file.stem,
        "branch": selavy_file.parent.name,
    }


def find_unique_matches(prepared, match_radius_arcsec):
    racs_coords = prepared["racs"]
    selavy_coords = prepared["selavy"]

    racs_index, selavy_index, separation, _ = search_around_sky(
        racs_coords,
        selavy_coords,
        match_radius_arcsec * u.arcsec,
    )

    racs_counts = np.bincount(racs_index, minlength=len(racs_coords))
    selavy_counts = np.bincount(selavy_index, minlength=len(selavy_coords))
    unique = (racs_counts[racs_index] == 1) & (selavy_counts[selavy_index] == 1)

    return racs_index[unique], selavy_index[unique], separation.arcsec[unique]


def measure_position_offsets(prepared, match_radius_arcsec=10.0, plot=False):
    racs_index, selavy_index, _ = find_unique_matches(prepared, match_radius_arcsec)

    if len(racs_index) == 0:
        raise ValueError(f"No unique matches within {match_radius_arcsec} arcsec")

    racs = prepared["racs"][racs_index]
    selavy = prepared["selavy"][selavy_index]
    ra_offsets = (selavy.ra - racs.ra).wrap_at(180 * u.deg).arcsec
    dec_offsets = (selavy.dec - racs.dec).arcsec
    median_ra = float(np.median(ra_offsets))
    median_dec = float(np.median(dec_offsets))


    return {
        "median_ra_arcsec": median_ra,
        "median_dec_arcsec": median_dec,
        "n_matches": len(ra_offsets),
        "ra_offsets": ra_offsets,
        "dec_offsets": dec_offsets,
    }


def compare_flux_scales(prepared, match_radius_arcsec=1.5, plot=False):
    racs_index, selavy_index, separations = find_unique_matches(prepared, match_radius_arcsec)
    matches = []

    for ri, si, separation in zip(racs_index, selavy_index, separations):
        racs_source = prepared["racs_sources"][ri]
        selavy_source = prepared["selavy_sources"][si]

        if racs_source["flux"] <= 0 or selavy_source["flux"] <= 0:
            continue

        matches.append({
            "racs_name": racs_source["name"],
            "selavy_name": selavy_source["name"],
            "separation_arcsec": float(separation),
            "racs_flux": racs_source["flux"],
            "selavy_flux": selavy_source["flux"],
            "snr": selavy_source["snr"],
            "flux_ratio": selavy_source["flux"] / racs_source["flux"],
        })

    if not matches:
        raise ValueError("No unique positive-flux matches found")

    ratios = np.array([row["flux_ratio"] for row in matches])
    median_ratio = float(np.median(ratios))


    return {"n_matches": len(matches), "median_ratio": median_ratio, "snr": np.array([row["snr"] for row in matches]), "ratios": ratios}


def measure_leakage(prepared, stokes_i_image, stokes, aperture_arcsec=13.0, plot=False, n_bins=45, min_per_bin=5):
    stokes_i_image = Path(stokes_i_image)
    stokes = stokes.upper()

    if stokes not in ("Q", "U", "V"):
        raise ValueError("stokes must be Q, U, or V")

    image_path = stokes_i_image.with_name(stokes_i_image.name.replace(".i.", f".{stokes.lower()}.", 1))

    if not image_path.is_file():
        raise FileNotFoundError(f"Stokes {stokes} image not found: {image_path}")

    with fits.open(image_path, memmap=True) as hdul:
        hdu = next(item for item in hdul if item.data is not None)
        image = np.squeeze(np.asarray(hdu.data, dtype=float)).copy()
        header = hdu.header.copy()

    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got {image.shape}")

    unit = header.get("BUNIT", "Jy/beam").lower()
    if "mjy" in unit:
        image *= 1
    elif "jy" in unit:
        image *= 1000
    else:
        raise ValueError(f"Unrecognised image unit: {header.get('BUNIT')}")

    wcs = WCS(header).celestial
    pixel_scale = np.sqrt(np.prod(proj_plane_pixel_scales(wcs) * 3600))
    radius_pix = aperture_arcsec / pixel_scale
    snr, fraction = [], []

    for source in prepared["selavy_sources"]:
        if source["flux"] <= 0 or source["snr"] <= 0:
            continue

        x, y = wcs.world_to_pixel(SkyCoord(source["ra"] * u.deg, source["dec"] * u.deg))
        if not np.isfinite(x) or not np.isfinite(y):
            continue

        x0, x1 = int(x - radius_pix - 1), int(x + radius_pix + 2)
        y0, y1 = int(y - radius_pix - 1), int(y + radius_pix + 2)
        if x0 < 0 or y0 < 0 or x1 > image.shape[1] or y1 > image.shape[0]:
            continue

        yy, xx = np.mgrid[y0:y1, x0:x1]
        aperture = np.hypot(xx - x, yy - y) <= radius_pix
        pixels = image[y0:y1, x0:x1][aperture]
        pixels = pixels[np.isfinite(pixels)]

        if pixels.size:
            snr.append(source["snr"])
            fraction.append(np.max(np.abs(pixels)) / source["flux"])

    snr = np.asarray(snr)
    fraction = np.asarray(fraction)
    valid = np.isfinite(snr) & np.isfinite(fraction) & (snr > 0) & (fraction > 0)
    snr, fraction = snr[valid], fraction[valid]

    if len(snr) < 2:
        raise ValueError(f"Not enough valid sources in {image_path}")

    edges = np.logspace(np.log10(snr.min()), np.log10(snr.max()), n_bins + 1)
    bin_snr, bin_fraction, bin_error = [], [], []

    for i in range(n_bins):
        in_bin = (snr >= edges[i]) & (snr < edges[i + 1])
        if in_bin.sum() < min_per_bin:
            continue

        median = np.median(fraction[in_bin])
        error = np.std(fraction[in_bin]) / np.sqrt(in_bin.sum())

        if median > 0 and error > 0:
            bin_snr.append(np.sqrt(edges[i] * edges[i + 1]))
            bin_fraction.append(median)
            bin_error.append(error)

    bin_snr = np.asarray(bin_snr)
    bin_fraction = np.asarray(bin_fraction)
    bin_error = np.asarray(bin_error)

    if len(bin_snr) < 8:
        raise ValueError("Too few populated SNR bins to fit a transition.")

    x = np.log10(bin_snr)
    y = np.log10(bin_fraction)
    y_error = bin_error / (bin_fraction * np.log(10))
    best = None

    for split in range(3, len(x) - 2):
        break_x = (x[split - 1] + x[split]) / 2
        shape = np.maximum(break_x - x, 0)
        level = np.average(y - shape, weights=1 / y_error**2)
        chi2 = np.sum(((y - level - shape) / y_error)**2)

        if best is None or chi2 < best["chi2"]:
            best = {"break_x": break_x, "level": level, "chi2": chi2}

    transition_snr = 10**best["break_x"]
    leakage_percent = 100 * 10**best["level"]


    return {
        "stokes": stokes,
        "leakage_percent": float(leakage_percent),
        "snr": snr,
        "fraction": fraction,
        "bin_snr": bin_snr,
        "bin_fraction": bin_fraction,
        "transition_snr": transition_snr,
        "level": best["level"],
    }


def summarize_bandpass_amplitudes(cal_dir, reference=None, plot=False, plot_dir=None, freq_mhz=None, gains=None, reference_gains=None, return_gains=False):
    cal_dir = Path(cal_dir).expanduser().resolve()
    ref_dir = Path(reference).expanduser().resolve() if reference is not None else None
    directories = (cal_dir,) if ref_dir is None else (cal_dir, ref_dir)
    taql = TAQL

    n_beams = 36
    n_antennas = 36
    n_channels = 288
    n_gains = 2 * n_channels
    row_pattern = re.compile(r"\[0,\s*(\d+),\s*0\]\[(.*)\]")
    pair_pattern = re.compile(r"\(([^(),]+),\s*([^(),]+)\)")
    if gains is None:
        gains = np.full((n_beams, n_antennas, n_channels, 2), complex(np.nan, np.nan))
    
        for beam in range(n_beams):
            beam_gains = []
    
            for directory in directories:
                table = directory / f"beam{beam:02d}" / f"beam{beam:02d}.bandpass.tab"
                cell = f"[{beam},0:{n_antennas},0:{n_gains}]"
                query = f"SELECT BANDPASS{cell} AS G, BANDPASS_VALID{cell} AS V FROM '{table.name}' LIMIT 1"
                output = subprocess.check_output([taql, query], cwd=table.parent, text=True)
    
                packed = np.full((n_antennas, n_gains), complex(np.nan, np.nan))
                valid = np.zeros((n_antennas, n_gains), dtype=bool)
                gain_rows = 0
                flag_rows = 0
    
                for line in output.splitlines():
                    row = row_pattern.fullmatch(line.strip())
                    if row is None:
                        continue
    
                    antenna = int(row[1])
                    pairs = pair_pattern.findall(row[2])
    
                    if pairs:
                        packed[antenna] = [complex(float(real), float(imag)) for real, imag in pairs]
                        gain_rows += 1
                    else:
                        valid[antenna] = np.fromstring(row[2], sep=",", dtype=int).astype(bool)
                        flag_rows += 1
    
                if gain_rows != n_antennas or flag_rows != n_antennas:
                    raise ValueError(f"Unexpected TaQL output for {table}")
    
                packed[~valid] = np.nan
                beam_gains.append(packed.reshape(n_antennas, n_channels, 2))
    
            target = beam_gains[0]
    
            if ref_dir is None:
                gains[beam] = target
            else:
                ref = beam_gains[1]
                usable = np.isfinite(target) & np.isfinite(ref) & (ref != 0)
                np.divide(target, ref, out=gains[beam], where=usable)
    
    if reference_gains is not None:
        target = gains
        gains = np.full_like(target, complex(np.nan, np.nan))
        usable = np.isfinite(target) & np.isfinite(reference_gains) & (reference_gains != 0)
        np.divide(target, reference_gains, out=gains, where=usable)

    amplitude = np.abs(gains)
    medians = np.ma.median(np.ma.masked_invalid(amplitude), axis=2).filled(np.nan)
    median_x = medians[:, :, 0]
    median_y = medians[:, :, 1]
    x_over_y = np.full_like(median_x, np.nan)
    usable = np.isfinite(median_x) & np.isfinite(median_y) & (median_y > 0)
    np.divide(median_x, median_y, out=x_over_y, where=usable)

    results = {"median_x": median_x, "median_y": median_y, "x_over_y": x_over_y}

    if return_gains:
        results["gains"] = gains

    return results


def read_coords(path, plot=False, *, field_ra, field_dec):
    with np.load(Path(path).expanduser()) as archive:
        data = dict(archive)

    starts = np.round(data["bin_start"], 3)
    ends = np.round(data["bin_end"], 3)
    hours = ((starts + ends) / 2 - starts[0]) / 3600
    frame = FK5(equinox=Time("J2000"))

    def median_altaz(seconds, ra, dec, location):
        days = np.floor(seconds / 86400)
        day_fraction = (seconds - days * 86400) / 86400
        times = Time(days, day_fraction, format="mjd", scale="utc")
        sky = SkyCoord(ra, dec, unit="deg", frame=frame)
        horizon = sky.transform_to(AltAz(obstime=times, location=location, pressure=0 * u.hPa))
        result = np.full((len(starts), 2), np.nan)

        for index, (start, end) in enumerate(zip(starts, ends)):
            use = (seconds >= start) & (seconds < end)
            if use.any():
                azimuth = np.unwrap(horizon.az.rad[use])
                result[index, 0] = np.degrees(np.median(azimuth)) % 360
                result[index, 1] = np.median(horizon.alt.deg[use])

        return result

    xyz = data["antenna_xyz_m"]
    dish_locations = EarthLocation.from_geocentric(*xyz.T, unit=u.m)
    beam_location = EarthLocation.from_geocentric(*xyz.mean(axis=0), unit=u.m)
    field = SkyCoord(field_ra, field_dec, frame=frame)
    field_altaz = median_altaz(data["integration_time_s"], field.ra.deg, field.dec.deg, beam_location)
    n_beams = len(data["beam_ra_deg"])
    n_dishes = len(xyz)
    beam_altaz = np.full((len(starts), n_beams, 2), np.nan)
    dish_altaz = np.full((len(starts), n_dishes, 2), np.nan)

    for beam in range(n_beams):
        ra = data["beam_ra_deg"][beam]
        dec = data["beam_dec_deg"][beam]
        beam_altaz[:, beam] = median_altaz(data["integration_time_s"], ra, dec, beam_location)

    for antenna in range(n_dishes):
        use = data["dish_antenna"] == antenna
        if use.any():
            seconds = data["dish_time_s"][use]
            ra = data["dish_ra_deg"][use]
            dec = data["dish_dec_deg"][use]
            dish_altaz[:, antenna] = median_altaz(seconds, ra, dec, dish_locations[antenna])

    keep = np.isfinite(dish_altaz).all(axis=2).any(axis=1)
    hours = hours[keep]
    beam_altaz = beam_altaz[keep]
    dish_altaz = dish_altaz[keep]


    return {
        "bin_names": data["bin_names"][keep],
        "hours": hours,
        "beam_altaz": beam_altaz,
        "dish_altaz": dish_altaz,
        "field_altaz": field_altaz[keep],
    }

BRANCHES = ("before", "after")
POLARISATIONS = ("Q", "U", "V")
FIELDS = {
    "86292": ("05h24m00.000s", "-27d58m10.204s"),
    "61078": ("11h24m00.000s", "-27d58m10.204s"),
}


def one_file(paths, description):
    matches = sorted(set(path.resolve() for path in paths if path.is_file()))
    if len(matches) != 1:
        raise ValueError(f"Expected one {description}, found {len(matches)}: {matches}")
    if matches[0].stat().st_size == 0:
        raise ValueError(f"Empty {description}: {matches[0]}")
    return matches[0]


def product_files(folder, obs_id):
    products = {}
    for branch in BRANCHES:
        branch_dir = folder / branch
        products[branch] = {}
        for pol in "iquv":
            name = f"image.{pol}.SB{obs_id}.linmos.taylor.0.restored.conv.fits"
            products[branch][pol] = one_file([branch_dir / name], f"{branch} Stokes {pol.upper()} mosaic")
        # Accept the cluster layout and the flattened layout used by the notebook.
        products[branch]["catalogue"] = one_file(
            list(branch_dir.glob("*.components.txt")) + list(branch_dir.glob("selavy-*/*.components.txt")),
            f"{branch} Selavy component catalogue",
        )
        products[branch]["noise"] = one_file(
            list(branch_dir.glob("noiseMap*.fits")) + list(branch_dir.glob("selavy-*/noiseMap*.fits")),
            f"{branch} Selavy noise FITS",
        )
        # Opening headers also rejects unfinished/truncated FITS products.
        for key in ("i", "q", "u", "v", "noise"):
            path = products[branch][key]
            with fits.open(path, memmap=True, lazy_load_hdus=False) as hdul:
                hdu = next(hdu for hdu in hdul if hdu.is_image and hdu.header.get("NAXIS", 0) >= 2)
                info = hdu.fileinfo()
                if path.stat().st_size < info["datLoc"] + info["datSpan"]:
                    raise ValueError(f"Incomplete FITS file: {path}")
    for beam in range(36):
        table = folder / "cal" / f"beam{beam:02d}" / f"beam{beam:02d}.bandpass.tab"
        if not (table / "table.dat").is_file():
            raise ValueError(f"Missing calibration table: {table}")
    return products


def collect_inputs(args, coords):
    full = product_files(args.full, args.obs_id)
    available = {}
    skipped = {}
    requested = set(args.bins.split(",")) if args.bins else None
    for name in coords["bin_names"]:
        name = str(name)
        if requested is not None and name not in requested:
            continue
        folder = args.snapshots / name
        if not folder.is_dir():
            skipped[name] = "Folder not present"
            continue
        try:
            available[name] = product_files(folder, args.obs_id)
        except (ValueError, OSError, StopIteration) as error:
            skipped[name] = str(error)
            LOG.info("Skipping %s: %s", name, error)
    if requested is not None:
        unknown = requested - set(map(str, coords["bin_names"]))
        if unknown:
            raise ValueError(f"Requested bins absent from usable pointing data: {sorted(unknown)}")
    if not available:
        raise ValueError("No complete snapshot bins with usable pointing were found.")
    LOG.info("Analysing %s snapshot bins, %s through %s", len(available), min(available), max(available))
    return full, available, skipped


def save_checkpoint(output, arrays, report):
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "measurements.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, output / "measurements.npz")
    temporary = output / "analysis_report.tmp.json"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, output / "analysis_report.json")


def measurement(function, report, context, **kwargs):
    try:
        return function(**kwargs)
    except ValueError as error:
        # An unmeasurable quantity is recorded, rather than stopping all other bins.
        LOG.warning("%s: %s", context, error)
        report["unmeasured"].append({"context": context, "reason": str(error)})
        return None


def plot_elevation(snapshot, full, elevations, ylabel, path, scaled=False, dec_snapshot=None, dec_full=None):
    series = [(snapshot, full, "o", ("C0", "C1"), None)]
    if dec_snapshot is not None:
        series = [(snapshot, full, "o", ("C0", "C1"), "ΔRA"), (dec_snapshot, dec_full, "s", ("C2", "C3"), "ΔDec")]
    fig, ax = plt.subplots(figsize=(8, 4))
    handles = []
    for points, levels, marker, colors, component in series:
        for branch, color in enumerate(colors):
            ax.scatter(elevations, points[:, branch], color=color, marker=marker)
            if np.isfinite(levels[branch]):
                ax.axhline(levels[branch], color=color, linestyle="--")
            label = BRANCHES[branch].capitalize() if component is None else f"{component} {BRANCHES[branch]}"
            handles.append(Line2D([], [], color=color, marker=marker, linestyle="none", label=label))
    duration_handles = [
        Line2D([], [], color="0.35", marker="o", linestyle="none", label="Scaled snapshots" if scaled else "15-minute snapshots"),
        Line2D([], [], color="0.35", linestyle="--", label="Full observation"),
    ]
    ax.set_xlabel("Median field elevation (deg)")
    ax.set_ylabel(ylabel)
    fig.legend(handles=handles, loc="upper left", ncol=2, frameon=False)
    fig.legend(handles=duration_handles, loc="upper right", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path)
    plt.close(fig)


def bandpass_elevation_plots(values, elevations, output, obs_id, relative=False, detail_beams=(0,), detail_antennas=(1,)):
    output.mkdir(parents=True, exist_ok=True)
    mode = "snapshot_over_full" if relative else "absolute"
    amplitude_label = "Median snapshot / full amplitude" if relative else "Median bandpass amplitude"
    ratio_label = "Median X ratio / median Y ratio" if relative else "Median X / median Y"
    # values shape is (bin, beam, antenna, X/Y/ratio).
    series = [("amplitude", amplitude_label, ((0, "X", "C0"), (1, "Y", "C1"))), ("x_over_y", ratio_label, ((2, ratio_label, "C0"),))]
    _, n_beams, n_antennas, _ = values.shape
    angle_min = float(np.nanmin(elevations))
    angle_max = float(np.nanmax(elevations))
    angle_span = angle_max - angle_min or 1.0
    beam_columns = np.arange(n_beams)[None, :, None]
    rows = np.arange(n_antennas - 1, -1, -1)[None, None, :]
    x = beam_columns + 0.06 + 0.88 * (elevations[:, None, None] - angle_min) / angle_span
    grid = [[(i, 0), (i, n_antennas)] for i in range(n_beams + 1)]
    grid += [[(0, i), (n_beams, i)] for i in range(n_antennas + 1)]

    for quantity, ylabel, curves in series:
        finite = np.concatenate([values[..., index][np.isfinite(values[..., index])] for index, _, _ in curves])
        if finite.size == 0:
            LOG.warning("No finite bandpass values for %s %s", mode, quantity)
            continue
        low, high = min(float(finite.min()), 1.0), max(float(finite.max()), 1.0)
        if low == high:
            low -= 0.01
            high += 0.01
        fig, ax = plt.subplots(figsize=(32, 26))
        ax.add_collection(LineCollection(grid, colors="0.35", linewidths=0.4))
        for y in rows.ravel() + 0.06 + 0.88 * (1 - low) / (high - low):
            ax.axhline(y, color="black", linestyle=":", linewidth=0.6, alpha=0.75, zorder=1)
        for index, label, color in curves:
            data = values[..., index]
            y = rows + 0.06 + 0.88 * (data - low) / (high - low)
            xp = np.broadcast_to(x, data.shape)
            keep = np.isfinite(xp) & np.isfinite(y)
            ax.scatter(xp[keep], y[keep], color=color, s=8, label=label, zorder=2)
        ax.set_xlim(0, n_beams)
        ax.set_ylim(0, n_antennas)
        ax.set_xticks(np.arange(n_beams) + 0.5)
        ax.set_xticklabels([f"B{i:02d}" for i in range(n_beams)])
        ax.set_yticks(n_antennas - 0.5 - np.arange(n_antennas))
        ax.set_yticklabels([f"AK{i + 1:02d}" for i in range(n_antennas)])
        ax.xaxis.tick_top()
        ax.tick_params(length=0, labelsize=8)
        ax.set_xlabel(f"Median field elevation ({angle_min:.2f}° to {angle_max:.2f}° per panel)")
        ax.set_ylabel(f"{ylabel} ({low:.4g} to {high:.4g} per panel)")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=len(curves), frameon=False)
        fig.subplots_adjust(left=0.05, right=0.995, bottom=0.04, top=0.94)
        fig.savefig(output / f"SB{obs_id}_{mode}_{quantity}_vs_elevation_36x36.pdf")
        plt.close(fig)

        selections = [(f"B{beam:02d}", values[:, beam, :, :], [f"AK{i + 1:02d}" for i in range(36)]) for beam in detail_beams]
        selections += [(f"AK{ant:02d}", values[:, :, ant - 1, :], [f"B{i:02d}" for i in range(36)]) for ant in detail_antennas]
        for label, data, panels in selections:
            fig, axes = plt.subplots(6, 6, figsize=(14, 11), sharex=True, sharey=True)
            for panel, ax in enumerate(axes.flat):
                ax.axhline(1, color="black", linestyle=":", linewidth=0.8, alpha=0.75)
                for index, curve_label, color in curves:
                    ax.scatter(elevations, data[:, panel, index], color=color, s=14, label=curve_label)
                ax.set_title(panels[panel], fontsize=9)
                ax.tick_params(labelsize=8)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=len(curves), frameon=False)
            fig.suptitle(f"SB{obs_id} · {label} · {mode}")
            fig.supxlabel("Median field elevation (deg)")
            fig.supylabel(ylabel)
            fig.tight_layout(rect=(0.02, 0.02, 1, 0.93))
            fig.savefig(output / f"SB{obs_id}_{mode}_{label}_{quantity}_vs_elevation_6x6.pdf")
            plt.close(fig)


def read_mosaic_preview(path, max_side):
    with fits.open(path, memmap=True) as hdul:
        hdu = next(hdu for hdu in hdul if hdu.is_image and hdu.data is not None)
        image = np.squeeze(hdu.data)
        if image.ndim != 2:
            raise ValueError(f"Expected a 2D mosaic: {path}")
        step = max(1, int(np.ceil(max(image.shape) / max_side)))
        view = (slice(None, None, step), slice(None, None, step))
        preview = image[view].copy()
        projection = WCS(hdu.header).celestial.slice(view)
        unit = hdu.header.get("BUNIT", "")
    return preview, projection, unit


def mosaic_plots(full, snapshots, output, obs_id, report, max_side=3200):
    output.mkdir(parents=True, exist_ok=True)
    # Bounded sampling avoids holding every large mosaic in memory.
    samples = []
    units = set()
    for products in snapshots.values():
        for branch in BRANCHES:
            sample, _, unit = read_mosaic_preview(products[branch]["i"], 400)
            samples.append(sample[np.isfinite(sample)])
            units.add(unit)
    if len(units) != 1:
        raise ValueError(f"Snapshot image units differ: {units}")
    pixels = np.concatenate(samples)
    if pixels.size == 0:
        raise ValueError("All snapshot Stokes I samples are blank.")
    snapshot_limits = np.percentile(pixels, [1, 99])
    report["mosaic_display"] = {"snapshot_limits": snapshot_limits.tolist(), "unit": next(iter(units)), "max_side_pixels": max_side, "percentiles": [1, 99], "scale_sampling_max_side": 400}
    for name, products in [("full", full), *list(snapshots.items())]:
        previews = [read_mosaic_preview(products[branch]["i"], max_side) for branch in BRANCHES]
        images, projections, pair_units = zip(*previews)
        if pair_units[0] != pair_units[1]:
            raise ValueError(f"Mosaic units differ in {name}")
        if name == "full":
            pixels = np.concatenate([image[np.isfinite(image)] for image in images])
            if pixels.size == 0:
                LOG.warning("Full Stokes I pair is blank")
                continue
            low, high = np.percentile(pixels, [1, 99])
            report["mosaic_display"]["full_limits"] = [float(low), float(high)]
            report["mosaic_display"]["full_unit"] = pair_units[0]
        else:
            low, high = snapshot_limits
        fig = plt.figure(figsize=(13, 6), constrained_layout=True)
        axes = []
        for index, branch in enumerate(BRANCHES):
            ax = fig.add_subplot(1, 2, index + 1, projection=projections[index])
            axes.append(ax)
            panel = ax.imshow(images[index], origin="lower", cmap="magma", vmin=low, vmax=high, interpolation="nearest")
            ax.set_title(branch.capitalize())
            ax.set_xlabel("Right ascension")
            ax.set_ylabel("Declination")
        fig.colorbar(panel, ax=axes, label=pair_units[0])
        fig.suptitle(f"SB{obs_id} · Stokes I · {'Full observation' if name == 'full' else name}")
        fig.savefig(output / f"SB{obs_id}_{name}_stokes_I_before_after.pdf", dpi=300)
        plt.close(fig)
        LOG.info("Saved %s Stokes I preview", name)


def result_triplet(result):
    return np.stack([result["median_x"], result["median_y"], result["x_over_y"]], axis=-1)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, help="Completed workflow folder; supplies the default inputs")
    parser.add_argument("--show-inputs", action="store_true", help="Print inferred inputs without extracting measurements")
    parser.add_argument("--obs-id", help="Override the observation ID")
    parser.add_argument("--field-ra", help="J2000 field RA; defaults to the recorded value for SB86292/SB61078")
    parser.add_argument("--field-dec", help="J2000 field Dec; defaults to the recorded value for SB86292/SB61078")
    parser.add_argument("--full", type=Path, help="Completed full-workflow directory with before/, after/, and cal/")
    parser.add_argument("--snapshots", type=Path, help="Existing snapshot-workflow directory containing bin-NNN/")
    parser.add_argument("--pointing", type=Path, help="pointing-SBID.npz exported for this observation")
    parser.add_argument("--racs", type=Path, help="External reference catalogue for flux and position comparisons")
    parser.add_argument("--output", type=Path, help="Default: snapshots/plots-SBID")
    parser.add_argument("--taql", help="TaQL executable, otherwise found on PATH")
    parser.add_argument("--bins", help="Optional comma-separated bin names, otherwise all complete bins")
    parser.add_argument("--resume-from", metavar="BIN", help="Restore the full observation and earlier bins from the output checkpoint, then measure from BIN onward")
    plots = parser.add_mutually_exclusive_group()
    plots.add_argument("--check-plots", dest="no_check_plots", action="store_false", help="Also make per-bin diagnostic plots")
    plots.add_argument("--no-check-plots", dest="no_check_plots", action="store_true", help="Skip per-bin diagnostic plots (default)")
    parser.set_defaults(no_check_plots=True)
    parser.add_argument("--detail-beams", default="0", help="Comma-separated zero-based beam IDs for additional 6x6 plots")
    parser.add_argument("--detail-antennas", default="1", help="Comma-separated antenna numbers 1–36 for additional 6x6 plots")
    parser.add_argument("--image-max-side", type=int, default=3200)
    parser.add_argument("--iers", type=Path, help="Optional local IERS-A Earth-orientation table for offline coordinate transforms")
    return resolve_inputs(parser.parse_args(), parser)


def main():
    global RACS_FILE, TAQL
    args = arguments()
    for key in ("full", "snapshots", "pointing", "racs"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    args.output = args.output.expanduser().resolve() if args.output else args.snapshots / f"plots-{args.obs_id}"
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(args.output / "analysis.log")])
    TAQL = shutil.which(args.taql or "taql")
    if TAQL is None:
        raise ValueError("TaQL not found. Load the ASKAPsoft module or provide --taql.")
    RACS_FILE = args.racs
    if not RACS_FILE.is_file():
        raise FileNotFoundError(RACS_FILE)
    if args.image_max_side < 1:
        raise ValueError("--image-max-side must be positive")
    beams = [int(value) for value in args.detail_beams.split(",") if value]
    antennas = [int(value) for value in args.detail_antennas.split(",") if value]
    if any(not 0 <= beam < 36 for beam in beams) or any(not 1 <= ant <= 36 for ant in antennas):
        raise ValueError("Detail beams are 0–35; antennas are 1–36.")

    iers.conf.auto_download = False
    if args.iers:
        iers.earth_orientation_table.set(iers.IERS_A.open(args.iers.expanduser()))
    with np.load(args.pointing) as archive:
        if "sbid" in archive and str(archive["sbid"].item()) != args.obs_id:
            raise ValueError("Pointing archive belongs to a different observation.")
        starts = dict(zip(map(str, archive["bin_names"]), archive["bin_start"]))
        ends = dict(zip(map(str, archive["bin_names"]), archive["bin_end"]))
    ra, dec = FIELDS.get(args.obs_id, (None, None))
    ra, dec = args.field_ra or ra, args.field_dec or dec
    if ra is None or dec is None:
        raise ValueError("Supply --field-ra and --field-dec for this observation.")
    coords = read_coords(args.pointing, plot=False, field_ra=ra, field_dec=dec)
    full, snapshots, skipped = collect_inputs(args, coords)
    names = list(snapshots)
    coordinate_index = {str(name): index for index, name in enumerate(coords["bin_names"])}
    indices = [coordinate_index[name] for name in names]
    elevations = coords["field_altaz"][indices, 1]
    # Match the notebook's 15-min-to-10-hour normalisation, using actual bin edges.
    durations = np.array([ends[name] - starts[name] for name in names])
    scales = np.sqrt(durations / 36000.0)
    n = len(names)
    report = {
        "observation": args.obs_id,
        "inputs": {"full": str(args.full), "snapshots": str(args.snapshots), "pointing": str(args.pointing), "racs": str(RACS_FILE)},
        "included_bins": names,
        "skipped_bins": skipped,
        "bins_without_usable_pointing": sorted(set(starts) - set(map(str, coords["bin_names"]))),
        "unmeasured": [],
        "settings": {"min_snr": MIN_SNR, "isolation_arcsec": ISOLATION_ARCSEC, "position_match_arcsec": 10.0, "flux_match_arcsec": 1.5, "leakage_aperture_arcsec": 13.0, "leakage_bins": 45, "leakage_min_per_bin": 5, "reference_duration_seconds": 36000, "field_ra": ra, "field_dec": dec, "taql": TAQL},
        "units": {"rms": "mJy/beam", "peak": "mJy/beam (Selavy flux_peak)", "offsets": "arcsec, direct RA difference; no cos(dec)", "leakage": "percent", "bandpass": "dimensionless"},
        "diagnostics_enabled": not args.no_check_plots,
        "bandpass_channel_axes": ["beam", "antenna", "channel", "receptor_X_Y"],
        "reference_definition": "abs(snapshot complex gain / full complex gain), then median across frequency; X/Y is the ratio of these medians",
        "python": sys.version,
        "versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__, "astropy": astropy.__version__},
    }
    arrays = {
        "obs_id": np.array(args.obs_id),
        "bin_names": np.array(names),
        "branches": np.array(BRANCHES),
        "polarisations": np.array(POLARISATIONS),
        "field_altaz_deg": coords["field_altaz"][indices],
        "beam_altaz_deg": coords["beam_altaz"][indices],
        "dish_altaz_deg": coords["dish_altaz"][indices],
        "hours": coords["hours"][indices],
        "bin_start_mjd_seconds": np.array([starts[name] for name in names]),
        "bin_end_mjd_seconds": np.array([ends[name] for name in names]),
        "snapshot_to_10h_noise_factor": scales,
        "bandpass_metrics": np.array(["median_x", "median_y", "x_over_y"]),
    }
    for key in ("rms_mjy_beam", "dynamic_range", "peak_mjy_beam", "median_ra_arcsec", "median_dec_arcsec", "position_n_matches", "median_flux_ratio", "flux_n_matches"):
        arrays["full_" + key] = np.full(2, np.nan)
        arrays["snapshot_" + key] = np.full((n, 2), np.nan)
    arrays["full_leakage_percent"] = np.full((2, 3), np.nan)
    arrays["snapshot_leakage_percent"] = np.full((n, 2, 3), np.nan)
    arrays["full_bandpass"] = np.full((36, 36, 3), np.nan)
    arrays["snapshot_bandpass"] = np.full((n, 36, 36, 3), np.nan)
    arrays["snapshot_bandpass_relative_full"] = np.full((n, 36, 36, 3), np.nan)
    reference_position = None
    completed = set()
    if args.resume_from:
        if args.resume_from not in names:
            raise ValueError(f"Resume bin {args.resume_from} is not among the ready input bins.")
        earlier = names[:names.index(args.resume_from)]
        completed = {"full", *earlier}
        previous_report = json.loads((args.output / "analysis_report.json").read_text())
        if previous_report["observation"] != args.obs_id or previous_report["inputs"] != report["inputs"]:
            raise ValueError("Checkpoint inputs do not match this analysis.")
        if previous_report["diagnostics_enabled"] != report["diagnostics_enabled"]:
            raise ValueError("Resume with the same diagnostic plot setting as the original run.")
        channel_dir = args.output / "values" / "bandpass"
        with np.load(args.output / "measurements.npz") as saved:
            old_names = list(map(str, saved["bin_names"]))
            if any(name not in old_names for name in earlier):
                raise ValueError("An earlier bin is missing from the checkpoint.")
            old_rows = [old_names.index(name) for name in earlier]
            if not np.isfinite(saved["full_rms_mjy_beam"]).all() or not np.isfinite(saved["snapshot_rms_mjy_beam"][old_rows]).all():
                raise ValueError("The full observation or an earlier bin has unfinished image measurements.")
            for name in completed:
                if not (channel_dir / f"{name}.npz").is_file():
                    raise ValueError(f"Saved bandpass results are missing for {name}.")
            for key in arrays:
                if key.startswith("full_"):
                    arrays[key] = saved[key]
                elif key.startswith("snapshot_") and key != "snapshot_to_10h_noise_factor":
                    arrays[key][:len(earlier)] = saved[key][old_rows]
            for key in saved.files:
                if any(key.startswith(f"check_{name}_") for name in completed):
                    arrays[key] = saved[key]
            arrays["dynamic_range_reference_ra_dec_deg"] = saved["dynamic_range_reference_ra_dec_deg"]
        reference_position = SkyCoord(*arrays["dynamic_range_reference_ra_dec_deg"], unit="deg")
        with np.load(channel_dir / "full.npz") as saved:
            reference_gains = saved["gains"]
        report["unmeasured"] = [item for item in previous_report["unmeasured"] if item["context"].split("/", 1)[0] in completed]
        LOG.info("Restored the full observation and %s bins. Resuming at %s.", len(earlier), args.resume_from)
    for name, products in [("full", full), *list(snapshots.items())]:
        if name in completed:
            continue
        LOG.info("Measuring %s", name)
        row = None if name == "full" else names.index(name)
        for branch_index, branch in enumerate(BRANCHES):
            files = products[branch]
            diagnostic_dir = args.output / "checks" / name / branch
            diagnostic_dir.mkdir(parents=True, exist_ok=True)
            context = f"{name}/{branch}"
            check = not args.no_check_plots
            measured = {}
            rms = median_rms(files["noise"], plot=False, bins=1000)
            measured["rms_mjy_beam"] = rms
            if check:
                plot_noise_checks(files["noise"], rms, diagnostic_dir)
            dynamic = measurement(measure_dynamic_range, report, context + "/dynamic_range", catalogue_file=files["catalogue"], median_rms=rms, reference_position=reference_position)
            if dynamic:
                if name == "full" and branch == "before":
                    reference_position = dynamic["position"]
                    arrays["dynamic_range_reference_ra_dec_deg"] = np.array([reference_position.ra.deg, reference_position.dec.deg])
                measured["dynamic_range"] = dynamic["dynamic_range"]
                measured["peak_mjy_beam"] = dynamic["peak"]
            if reference_position is None:
                raise ValueError("No full-before dynamic-range reference source could be selected.")
            prepared = measurement(prepare_position_catalogues, report, context + "/catalogue", selavy_file=files["catalogue"])
            if prepared:
                offsets = measurement(measure_position_offsets, report, context + "/position", prepared=prepared, match_radius_arcsec=10.0, plot=False)
                if offsets:
                    measured.update({key: offsets[key] for key in ("median_ra_arcsec", "median_dec_arcsec")})
                    measured["position_n_matches"] = offsets["n_matches"]
                    arrays[f"check_{name}_{branch}_ra_offsets_arcsec"] = offsets["ra_offsets"]
                    arrays[f"check_{name}_{branch}_dec_offsets_arcsec"] = offsets["dec_offsets"]
                    if check:
                        plot_position_check(offsets, diagnostic_dir)
                flux = measurement(compare_flux_scales, report, context + "/flux", prepared=prepared, match_radius_arcsec=1.5, plot=False)
                if flux:
                    measured["median_flux_ratio"] = flux["median_ratio"]
                    measured["flux_n_matches"] = flux["n_matches"]
                    arrays[f"check_{name}_{branch}_flux_snr"] = flux["snr"]
                    arrays[f"check_{name}_{branch}_flux_ratios"] = flux["ratios"]
                    if check:
                        plot_flux_check(flux, diagnostic_dir)
                for pol_index, pol in enumerate(POLARISATIONS):
                    leak = measurement(measure_leakage, report, context + "/" + pol, prepared=prepared, stokes_i_image=files["i"], stokes=pol, plot=False)
                    if leak:
                        for key in ("snr", "fraction", "bin_snr", "bin_fraction", "transition_snr", "level"):
                            arrays[f"check_{name}_{branch}_{pol}_{key}"] = np.asarray(leak[key])
                        if check:
                            plot_leakage_check(leak, pol, diagnostic_dir)
                        if row is None:
                            arrays["full_leakage_percent"][branch_index, pol_index] = leak["leakage_percent"]
                        else:
                            arrays["snapshot_leakage_percent"][row, branch_index, pol_index] = leak["leakage_percent"]
            for key, value in measured.items():
                if row is None:
                    arrays["full_" + key][branch_index] = value
                else:
                    arrays["snapshot_" + key][row, branch_index] = value
            save_checkpoint(args.output, arrays, report)
            gc.collect()

        cal = args.full / "cal" if row is None else args.snapshots / name / "cal"
        LOG.info("Reading bandpass tables for %s", name)
        result = summarize_bandpass_amplitudes(cal, plot=False, plot_dir=args.output / "checks" / name / "bandpass", return_gains=True)
        if not args.no_check_plots:
            plot_bandpass_frequency(result["gains"], args.output / "checks" / name / "bandpass", name)
        if row is None:
            arrays["full_bandpass"] = result_triplet(result)
            reference_gains = result["gains"]
        else:
            arrays["snapshot_bandpass"][row] = result_triplet(result)
            relative = summarize_bandpass_amplitudes(cal, reference=args.full / "cal", plot=False, plot_dir=args.output / "checks" / name / "bandpass_reference", gains=result["gains"], reference_gains=reference_gains, return_gains=True)
            arrays["snapshot_bandpass_relative_full"][row] = result_triplet(relative)
            if not args.no_check_plots:
                plot_bandpass_frequency(relative["gains"], args.output / "checks" / name / "bandpass_reference", name + " / full")
            del relative
        channel_dir = args.output / "values" / "bandpass"
        channel_dir.mkdir(parents=True, exist_ok=True)
        temporary = channel_dir / f"{name}.tmp.npz"
        np.savez_compressed(temporary, gains=result["gains"])
        os.replace(temporary, channel_dir / f"{name}.npz")
        del result
        save_checkpoint(args.output, arrays, report)

    arrays["snapshot_rms_10h_mjy_beam"] = arrays["snapshot_rms_mjy_beam"] * scales[:, None]
    arrays["snapshot_dynamic_range_10h"] = arrays["snapshot_dynamic_range"] / scales[:, None]
    elevation_dir = args.output / "elevation"
    elevation_dir.mkdir(exist_ok=True)
    plot_elevation(arrays["snapshot_rms_10h_mjy_beam"], arrays["full_rms_mjy_beam"], elevations, "10-hour equivalent RMS (mJy/beam)", elevation_dir / "rms_vs_elevation.pdf", scaled=True)
    plot_elevation(arrays["snapshot_dynamic_range_10h"], arrays["full_dynamic_range"], elevations, "10-hour equivalent dynamic range", elevation_dir / "dynamic_range_vs_elevation.pdf", scaled=True)
    plot_elevation(arrays["snapshot_median_ra_arcsec"], arrays["full_median_ra_arcsec"], elevations, "Median Selavy offset relative to RACS (arcsec)", elevation_dir / "position_offsets_vs_elevation.pdf", dec_snapshot=arrays["snapshot_median_dec_arcsec"], dec_full=arrays["full_median_dec_arcsec"])
    plot_elevation(arrays["snapshot_median_flux_ratio"], arrays["full_median_flux_ratio"], elevations, "Median Selavy / RACS integrated flux", elevation_dir / "flux_scale_vs_elevation.pdf")
    for pol_index, pol in enumerate(POLARISATIONS):
        plot_elevation(arrays["snapshot_leakage_percent"][:, :, pol_index], arrays["full_leakage_percent"][:, pol_index], elevations, f"Stokes I to {pol} leakage (%)", elevation_dir / f"leakage_{pol}_vs_elevation.pdf")

    bandpass_elevation_plots(arrays["snapshot_bandpass"], elevations, args.output / "bandpass", args.obs_id, detail_beams=beams, detail_antennas=antennas)
    bandpass_elevation_plots(arrays["snapshot_bandpass_relative_full"], elevations, args.output / "bandpass_reference", args.obs_id, relative=True, detail_beams=beams, detail_antennas=antennas)
    mosaic_plots(full, snapshots, args.output / "mosaics", args.obs_id, report, max_side=args.image_max_side)
    report["finished"] = True
    save_checkpoint(args.output, arrays, report)
    LOG.info("Finished. %s unmeasured quantities; %s skipped bins. Results: %s", len(report["unmeasured"]), len(skipped), args.output)


def plot_noise_checks(path, median, output):
    with fits.open(path, memmap=True) as hdul:
        hdu = next(hdu for hdu in hdul if hdu.is_image and hdu.data is not None)
        image = np.squeeze(np.asarray(hdu.data, dtype=float))
        unit = u.Unit(hdu.header.get("BUNIT", "Jy/beam"))
        image = image * unit.to(u.mJy / u.beam)
        image[~np.isfinite(image) | (image <= 0)] = np.nan
        wcs = WCS(hdu.header).celestial
    values = image[np.isfinite(image)]
    low, high = np.percentile(values, [2, 98])
    step = max(1, int(np.ceil(max(image.shape) / 1600)))
    view = (slice(None, None, step), slice(None, None, step))
    fig, ax = plt.subplots(figsize=(9, 7), subplot_kw={"projection": wcs.slice(view)})
    panel = ax.imshow(image[view], origin="lower", cmap="viridis", vmin=low, vmax=high)
    ax.set_xlabel("Right ascension")
    ax.set_ylabel("Declination")
    fig.colorbar(panel, ax=ax, label="mJy/beam")
    ax.set_title(f"RMS map · median {median:.3g} mJy/beam")
    fig.savefig(output / "rms_map.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=1000)
    ax.axvline(median, color="red", linestyle="--", label=f"Median = {median:.3g}")
    ax.set_xlabel("mJy/beam")
    ax.set_ylabel("Number of pixels")
    ax.set_title("RMS pixel distribution")
    ax.set_xlim(np.percentile(values, 0), np.percentile(values, 99.9))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "rms_histogram.pdf")
    plt.close(fig)


def plot_position_check(result, output):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(result["ra_offsets"], result["dec_offsets"], s=18, alpha=0.6)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(result["median_ra_arcsec"], color="red", linestyle=":", label=f"Median ΔRA = {result['median_ra_arcsec']:.3f}″")
    ax.axhline(result["median_dec_arcsec"], color="blue", linestyle=":", label=f"Median ΔDec = {result['median_dec_arcsec']:.3f}″")
    ax.set_xlabel("ΔRA (arcsec)")
    ax.set_ylabel("ΔDec (arcsec)")
    ax.set_title(f"RACS–Selavy offsets · {result['n_matches']} matches")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "position_offsets.pdf")
    plt.close(fig)


def plot_flux_check(result, output):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(result["snr"], result["ratios"], alpha=0.6)
    ax.axhline(result["median_ratio"], color="red", linestyle="--", label=f"Median = {result['median_ratio']:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("Selavy Stokes I S/N")
    ax.set_ylabel("Selavy / RACS integrated flux")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "flux_ratio.pdf")
    plt.close(fig)


def plot_leakage_check(result, stokes, output):
    line_snr = np.geomspace(result["bin_snr"].min(), result["bin_snr"].max(), 300)
    line_fraction = 10**result["level"] * np.maximum(result["transition_snr"] / line_snr, 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(result["snr"], 100 * result["fraction"], s=10, alpha=0.25, label="Sources")
    ax.scatter(result["bin_snr"], 100 * result["bin_fraction"], s=24, color="black", label="Binned medians")
    ax.plot(line_snr, 100 * line_fraction, color="red", label="1/SNR then flat")
    ax.axvline(result["transition_snr"], color="gray", linestyle=":", label=f"Transition SNR = {result['transition_snr']:.1f}")
    ax.axhline(result["leakage_percent"], color="red", linestyle="--", label=f"Plateau = {result['leakage_percent']:.3g}%")
    ax.set(xscale="log", yscale="log", xlabel="Stokes I SNR", ylabel=f"Maximum |Stokes {stokes}| / integrated Stokes I (%)", title=f"Stokes I to {stokes} leakage")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / f"leakage_{stokes}.pdf")
    plt.close(fig)


def plot_bandpass_frequency(gains, output, label):
    output.mkdir(parents=True, exist_ok=True)
    n_beams, n_antennas, n_channels, _ = gains.shape
    x = np.linspace(0.04, 0.96, n_channels)
    grid = [[(i, 0), (i, n_antennas)] for i in range(n_beams + 1)]
    grid += [[(0, i), (n_beams, i)] for i in range(n_antennas + 1)]
    plots = [("amplitude", "Gain amplitude", np.abs(gains), 0.65, 1.40), ("phase", "Gain phase (deg)", np.angle(gains, deg=True), -50, 50)]
    for quantity, ylabel, values, low, high in plots:
        traces = [[], []]
        for beam in range(n_beams):
            for antenna in range(n_antennas):
                for pol in range(2):
                    curve = values[beam, antenna, :, pol]
                    y = n_antennas - 1 - antenna + 0.04 + 0.92 * (curve - low) / (high - low)
                    y[(curve < low) | (curve > high)] = np.nan
                    traces[pol].append(np.column_stack((beam + x, y)))
        fig, ax = plt.subplots(figsize=(32, 26))
        ax.add_collection(LineCollection(grid, colors="0.35", linewidths=0.4))
        for pol in range(2):
            ax.add_collection(LineCollection(traces[pol], colors=f"C{pol}", linewidths=0.55))
        ax.set_xlim(0, n_beams)
        ax.set_ylim(0, n_antennas)
        ax.set_xticks(np.arange(n_beams) + 0.5)
        ax.set_xticklabels([f"B{i:02d}" for i in range(n_beams)])
        ax.set_yticks(n_antennas - 0.5 - np.arange(n_antennas))
        ax.set_yticklabels([f"AK{i + 1:02d}" for i in range(n_antennas)])
        ax.xaxis.tick_top()
        ax.tick_params(length=0, labelsize=8)
        ax.set_xlabel(f"Channel index (0–{n_channels - 1} per panel)")
        ax.set_ylabel(f"{ylabel} ({low:g} to {high:g} per panel)")
        fig.suptitle(f"{label} · {ylabel}")
        handles = [Line2D([], [], color="C0", label="X"), Line2D([], [], color="C1", label="Y")]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.967), ncol=2)
        fig.subplots_adjust(left=0.05, right=0.995, bottom=0.035, top=0.94)
        fig.savefig(output / f"bandpass_{quantity}_vs_channel_36x36.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
