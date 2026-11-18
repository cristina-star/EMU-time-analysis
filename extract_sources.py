#!/usr/bin/env python3
"""Detect bin-000 image peaks and save their fixed-position fluxes through time."""
import argparse
from pathlib import Path

from workflow_inputs import resolve_inputs

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.ndimage import find_objects, label, maximum_filter
from scipy.spatial import cKDTree

OBS_ID = None
FIELD = None
BRANCH = "before"
dy, dx = np.mgrid[-2:3, -2:3].reshape(2, -1)
PEAK_DESIGN = np.column_stack((np.ones(dx.size), dx, dy, dx**2, dx * dy, dy**2))

def image_path(snapshots, bin_number, beam):
    folder = snapshots / f"bin-{bin_number:03d}" / BRANCH
    if beam is None:
        return folder / f"image.i.SB{OBS_ID}.linmos.taylor.0.restored.conv.fits"
    name = f"beam{beam:02d}"
    filename = f"image.i.{FIELD}.SB{OBS_ID}.cont.{FIELD}.{name}.taylor.0.restored.conv.fits"
    return folder / name / filename

def read_image(path):
    image, header = fits.getdata(path, header=True, memmap=True)
    to_mjy = u.Unit(header["BUNIT"]).to(u.mJy / u.beam)
    image = image.squeeze()
    if image.ndim != 2:
        raise ValueError(f"Expected a two-dimensional Stokes I image: {path}")
    return image, WCS(header).celestial, to_mjy, header

def racs_anchors(wcs, shape, catalogue, limit=250):
    height, width = shape
    corners = wcs.pixel_to_world(
        [0, width - 1, width - 1, 0], [0, 0, height - 1, height - 1],
    ).icrs
    ra, dec, flux = catalogue
    pad = 0.2
    nearby = (
        (ra >= corners.ra.deg.min() - pad) & (ra <= corners.ra.deg.max() + pad)
        & (dec >= corners.dec.deg.min() - pad) & (dec <= corners.dec.deg.max() + pad)
    )
    coords = SkyCoord(ra[nearby], dec[nearby], unit="deg", frame="icrs")
    flux = flux[nearby]
    if len(coords) < 2:
        raise ValueError("Too few RACS sources in the beam image.")

    _, nearest, _ = coords.match_to_catalog_sky(coords, nthneighbor=2)
    x, y = wcs.world_to_pixel(coords)
    candidates = np.flatnonzero(
        (nearest > 60 * u.arcsec)
        & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        & np.isfinite(flux) & (flux > 0)
    )
    ranked = candidates[np.argsort(flux[candidates])[::-1]]
    return coords[ranked if limit is None else ranked[:limit]]

def measure_peaks(image, wcs, to_mjy, anchors, args, min_peak_mjy=None, return_flux=False):
    predicted = np.column_stack(wcs.world_to_pixel(anchors))
    measured = np.full_like(predicted, np.nan)
    flux = np.full(len(anchors), np.nan)
    threshold = args.min_alignment_mjy if min_peak_mjy is None else min_peak_mjy
    pixel_arcsec = 3600 * np.min(proj_plane_pixel_scales(wcs))
    radius = int(np.ceil(args.max_match_arcsec / pixel_arcsec))
    margin = radius + 2
    height, width = image.shape

    for i, (xi, yi) in enumerate(predicted):
        if not (np.isfinite(xi) and np.isfinite(yi)):
            continue
        cx, cy = round(xi), round(yi)
        if not (margin <= cx < width - margin and margin <= cy < height - margin):
            continue

        box = image[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1]
        finite = np.isfinite(box)
        if not finite.any():
            continue
        row, col = np.unravel_index(np.argmax(np.where(finite, box, -np.inf)), box.shape)
        px, py = cx + col - radius, cy + row - radius
        patch = image[py - 2:py + 3, px - 2:px + 3]
        if image[py, px] * to_mjy < threshold or not np.isfinite(patch).all():
            continue

        constant, bx, by, cxx, cxy, cyy = np.linalg.lstsq(PEAK_DESIGN, patch.ravel(), rcond=None)[0]
        curvature = np.array([[2 * cxx, cxy], [cxy, 2 * cyy]])
        if np.linalg.eigvalsh(curvature)[-1] >= 0:
            continue
        subpixel = np.linalg.solve(curvature, -np.array([bx, by]))
        if np.max(np.abs(subpixel)) > 1.5:
            continue

        peak = np.array([px, py]) + subpixel
        if wcs.pixel_to_world(*peak).separation(anchors[i]).arcsec <= args.max_match_arcsec:
            sx, sy = subpixel
            height_mjy = to_mjy * (
                constant + bx * sx + by * sy + cxx * sx**2 + cxy * sx * sy + cyy * sy**2
            )
            if np.isfinite(height_mjy) and height_mjy >= threshold:
                measured[i] = peak
                flux[i] = height_mjy

    if return_flux:
        return flux
    return measured, measured - predicted

def read_windows(path, first, last):
    windows = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        number = int(fields[0].split("-")[-1])
        start, end = map(float, fields[1].split(","))
        if not np.isfinite([start, end]).all() or end <= start or number in windows:
            raise ValueError(f"Invalid or duplicate time window: {line}")
        windows[number] = (start, end)
    if not windows:
        raise ValueError(f"No time windows in {path}")
    numbers = np.array(sorted(number for number in windows if first <= number <= last))
    if numbers.size < 2:
        raise ValueError("Need at least two planned bins in the requested range.")
    starts, ends = np.array([windows[number] for number in numbers]).T
    origin = min(start for start, _ in windows.values())
    return numbers, starts, ends, origin

def available_range(numbers, starts, ends, snapshots, beams):
    """Trim unavailable leading/trailing bins; retain missing interior bins as gaps."""
    counts = np.array([
        sum(image_path(snapshots, int(number), beam).is_file() for beam in beams)
        for number in numbers
    ])
    present = np.flatnonzero(counts > 0)
    if present.size < 2:
        raise ValueError(f"Need at least two bins with an available {BRANCH} Stokes I beam image.")
    selected = slice(present[0], present[-1] + 1)
    for number, count in zip(numbers[selected], counts[selected]):
        print(f"bin-{number:03d}: {count}/{len(beams)} {BRANCH} Stokes I beam image files available", flush=True)
    return numbers[selected], starts[selected], ends[selected]

def align_snapshot(image, wcs, factor, anchors, args):
    _, offsets = measure_peaks(image, wcs, factor, anchors, args)
    candidates = np.flatnonzero(np.isfinite(offsets).all(axis=1))
    if candidates.size < 5:
        raise ValueError(f"Only {candidates.size} RACS matches; need at least 5.")
    values = offsets[candidates]
    deviation = np.abs(values - np.median(values, axis=0))
    scatter = 1.4826 * np.median(deviation, axis=0)
    accepted = (deviation <= 4 * np.maximum(scatter, 0.25)).all(axis=1)
    if accepted.sum() < 5:
        raise ValueError("Fewer than 5 reliable RACS matches after clipping.")
    corrected = wcs.deepcopy()
    corrected.wcs.crpix = wcs.wcs.crpix + np.median(values[accepted], axis=0)
    corrected.wcs.set()
    return corrected, int(accepted.sum())


def max_source_flux(image, wcs, factor, sources, search_arcsec, batch_size=256):
    """Literal maximum finite pixel in a 10-arcsec sky circle; no later flux cut."""
    flux = np.full(len(sources), np.nan, dtype=np.float32)
    if not len(sources):
        return flux
    x, y = wcs.world_to_pixel(sources)
    usable = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    radius = int(np.ceil(search_arcsec / (3600 * np.min(proj_plane_pixel_scales(wcs))))) + 2
    dy, dx = np.mgrid[-radius:radius+1, -radius:radius+1].reshape(2, -1)
    height, width = image.shape
    for first in range(0, len(usable), batch_size):
        take = usable[first:first+batch_size]
        xx = np.floor(x[take]).astype(int)[:, None] + dx
        yy = np.floor(y[take]).astype(int)[:, None] + dy
        inside = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
        sky = wcs.pixel_to_world(xx, yy)
        within = sources[take, None].separation(sky).arcsec <= search_arcsec
        values = image[np.clip(yy, 0, height-1), np.clip(xx, 0, width-1)]
        valid = inside & within & np.isfinite(values)
        maxima = np.max(np.where(valid, values, -np.inf), axis=1)
        has_pixels = valid.any(axis=1)
        flux[take[has_pixels]] = maxima[has_pixels] * factor
    return flux


def detect_sources(image, wcs, factor, header, args):
    """Find bright local maxima in the image; keep the brightest per beam-sized neighbourhood."""
    separation = args.min_separation_arcsec
    if separation is None:
        separation = 3600 * float(header.get("BMAJ", np.nan))
    if not np.isfinite(separation) or separation <= 0:
        raise ValueError("Missing restoring-beam BMAJ; supply --min-separation-arcsec.")
    pixel_arcsec = 3600 * np.min(proj_plane_pixel_scales(wcs))
    half_width = max(1, int(np.ceil(separation / (2 * pixel_arcsec))))
    values = np.asarray(image * factor, dtype=np.float32)
    values[~np.isfinite(values)] = -np.inf
    local_max = maximum_filter(values, size=2*half_width+1, mode="constant", cval=-np.inf)
    peaks = (values >= args.min_reference_mjy) & (values == local_max)
    # Flat-topped peaks contribute one coordinate, rather than one per pixel.
    labels, _ = label(peaks, structure=np.ones((3, 3), dtype=int))
    x, y = [], []
    for number, bounds in enumerate(find_objects(labels), start=1):
        if bounds is None:
            continue
        patch = np.where(labels[bounds] == number, values[bounds], -np.inf)
        row, col = np.unravel_index(np.argmax(patch), patch.shape)
        y.append(bounds[0].start + row)
        x.append(bounds[1].start + col)
    x, y = np.asarray(x, dtype=int), np.asarray(y, dtype=int)
    coordinates = wcs.pixel_to_world(x, y).icrs
    good = np.isfinite(coordinates.ra.deg) & np.isfinite(coordinates.dec.deg)
    x, y, coordinates = x[good], y[good], coordinates[good]
    if not len(coordinates):
        return coordinates, x, y, separation
    # Angular separation handles rotated/distorted pixel grids correctly.
    tree = cKDTree(coordinates.cartesian.xyz.value.T)
    chord = 2 * np.sin(np.deg2rad(separation / 3600) / 2)
    blocked = np.zeros(len(coordinates), dtype=bool)
    kept = []
    for index in np.argsort(-values[y, x], kind="stable"):
        if blocked[index]:
            continue
        kept.append(index)
        blocked[tree.query_ball_point(coordinates[index].cartesian.xyz.value, chord)] = True
    kept = np.asarray(kept, dtype=int)
    return coordinates[kept], x[kept], y[kept], separation


def measure_beam(beam, numbers, catalogue, args):
    path = image_path(args.snapshots, args.reference_bin, beam)
    image, wcs, factor, header = read_image(path)
    anchors = racs_anchors(wcs, image.shape, catalogue)
    corrected, reference_matches = align_snapshot(image, wcs, factor, anchors, args)
    sources, x, y, separation = detect_sources(image, corrected, factor, header, args)
    reference_flux = max_source_flux(image, corrected, factor, sources, args.source_search_arcsec)
    good = np.isfinite(reference_flux) & (reference_flux >= args.min_reference_mjy)
    sources, x, y, reference_flux = sources[good], x[good], y[good], reference_flux[good]
    print(f"  {len(sources)} detected image peaks at >= {args.min_reference_mjy:g} mJy/beam; "
          f"minimum separation {separation:.2f} arcsec; {reference_matches} alignment matches", flush=True)
    flux = np.full((len(numbers), len(sources)), np.nan, dtype=np.float32)
    match_counts = np.zeros(len(numbers), dtype=int)
    result = {
        "ra_deg": np.asarray(sources.ra.deg), "dec_deg": np.asarray(sources.dec.deg),
        "reference_pixel_x": x, "reference_pixel_y": y,
        "reference_peak": reference_flux, "flux": flux,
        "alignment_matches": match_counts, "min_separation_arcsec": separation,
    }
    if not len(sources):
        return result
    for i, number in enumerate(numbers):
        if number == args.reference_bin:
            flux[i] = reference_flux
            match_counts[i] = reference_matches
            continue
        path = image_path(args.snapshots, number, beam)
        if not path.is_file():
            print(f"  bin-{number:03d}: missing image; leaving a gap", flush=True)
            continue
        try:
            current, current_wcs, factor, _ = read_image(path)
            aligned, matches = align_snapshot(current, current_wcs, factor, anchors, args)
            flux[i] = max_source_flux(current, aligned, factor, sources, args.source_search_arcsec)
            match_counts[i] = matches
            print(f"  bin-{number:03d}: {np.isfinite(flux[i]).sum()}/{len(sources)} peaks measured; "
                  f"{matches} alignment matches", flush=True)
        except (ValueError, OSError, np.linalg.LinAlgError) as exc:
            print(f"  bin-{number:03d}: {exc}; leaving a gap", flush=True)
    return result


def save_points(results, processed, numbers, starts, ends, origin, args):
    beams = np.asarray(args.beams, dtype=int)
    available = [beam for beam in beams if beam in results]
    def concatenate(key, dtype=float):
        arrays = [results[beam][key] for beam in available]
        return np.concatenate(arrays).astype(dtype) if arrays else np.empty(0, dtype=dtype)
    source_beams = np.concatenate([
        np.full(len(results[beam]["ra_deg"]), beam, dtype=int) for beam in available
    ]) if available else np.empty(0, dtype=int)
    indices = np.concatenate([
        np.arange(len(results[beam]["ra_deg"])) for beam in available
    ]) if available else np.empty(0, dtype=int)
    fluxes = [results[beam]["flux"] for beam in available]
    flux = np.concatenate(fluxes, axis=1) if fluxes else np.empty((len(numbers), 0), dtype=np.float32)
    alignment = np.zeros((len(numbers), len(beams)), dtype=int)
    separations = np.full(len(beams), np.nan)
    counts = np.zeros(len(beams), dtype=int)
    for j, beam in enumerate(beams):
        if beam in results:
            alignment[:, j] = results[beam]["alignment_matches"]
            separations[j] = results[beam]["min_separation_arcsec"]
            counts[j] = len(results[beam]["ra_deg"])
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"SB{OBS_ID}_all_beam_source_peaks_bins{numbers[0]:03d}-{numbers[-1]:03d}.npz"
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary, obs_id=OBS_ID, branch=BRANCH, source_snapshots=str(args.snapshots.resolve()), reference_bin=args.reference_bin,
        min_reference_mjy_beam=args.min_reference_mjy,
        source_search_radius_arcsec=args.source_search_arcsec,
        source_detection="Image local maxima in reference bin; brightest retained within one minimum-separation radius; no RACS source selection, spatial strip or cross-beam uniqueness cut.",
        flux_definition="Maximum finite image pixel within source_search_radius_arcsec of each fixed ICRS coordinate; mJy/beam; no later flux cut.",
        coordinate_frame="ICRS", observation_start_mjd_seconds=origin,
        bin_numbers=numbers, bin_names=np.array([f"bin-{n:03d}" for n in numbers]),
        bin_start_mjd_seconds=starts, bin_end_mjd_seconds=ends,
        hours=((starts+ends)/2-origin)/3600,
        beam_ids=beams, processed_beams=np.isin(beams, processed),
        beam_succeeded=np.isin(beams, available), beam_source_count=counts,
        beam_detection_min_separation_arcsec=separations,
        alignment_match_count=alignment,
        beam_centre_ra_deg=args.beam_ra_deg[beams], beam_centre_dec_deg=args.beam_dec_deg[beams],
        source_beam_ids=source_beams, source_index_within_beam=indices,
        source_ra_deg=concatenate("ra_deg"), source_dec_deg=concatenate("dec_deg"),
        source_reference_pixel_x=concatenate("reference_pixel_x", int),
        source_reference_pixel_y=concatenate("reference_pixel_y", int),
        source_reference_peak_mjy_beam=concatenate("reference_peak", np.float32),
        peak_mjy_beam=flux,
    )
    temporary.replace(path)
    print(f"Saved {path} ({len(source_beams)} source/beam entries)", flush=True)


def main():
    global OBS_ID, FIELD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, help="Completed workflow folder; supplies the default inputs")
    parser.add_argument("--show-inputs", action="store_true", help="Print inferred inputs without extracting sources")
    parser.add_argument("--obs-id")
    parser.add_argument("--field", help="Field identifier used in the image filenames, e.g. EMU_0524-28")
    parser.add_argument("--pointing", type=Path, help="pointing-ID.npz containing beam_ra_deg and beam_dec_deg")
    parser.add_argument("--snapshots", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--racs", type=Path, help="Catalogue for astrometric alignment only.")
    parser.add_argument("--beams", type=int, nargs="+", default=list(range(36)))
    parser.add_argument("--reference-bin", type=int, default=0)
    parser.add_argument("--first-bin", type=int, default=0)
    parser.add_argument("--last-bin", type=int, help="Default: last bin in snapshots/windows.txt")
    parser.add_argument("--min-reference-mjy", type=float, default=10)
    parser.add_argument("--source-search-arcsec", type=float, default=10)
    parser.add_argument("--min-separation-arcsec", type=float,
                        help="Minimum separation of detected reference peaks; default is image BMAJ.")
    parser.add_argument("--min-alignment-mjy", type=float, default=10)
    parser.add_argument("--max-match-arcsec", type=float, default=10)
    args = resolve_inputs(parser.parse_args(), parser, sources=True)
    args.beams = sorted(set(args.beams))
    if any(beam < 0 or beam > 35 for beam in args.beams):
        parser.error("Beam numbers must be 0 through 35.")
    if args.first_bin < 0 or args.last_bin <= args.first_bin or args.reference_bin < 0:
        parser.error("Need nonnegative bin numbers and last-bin greater than first-bin.")
    cuts = [args.min_reference_mjy, args.source_search_arcsec,
            args.min_alignment_mjy, args.max_match_arcsec]
    if args.min_separation_arcsec is not None:
        cuts.append(args.min_separation_arcsec)
    if not np.isfinite(cuts).all() or min(cuts) <= 0:
        parser.error("Flux cuts and angular radii must be finite and positive.")
    if not args.racs.is_file():
        parser.error(f"Missing alignment catalogue: {args.racs}")
    print("Reading the alignment catalogue. Science sources will be detected in the images.", flush=True)
    table = Table.read(args.racs)
    catalogue = tuple(np.asarray(table[name], dtype=float) for name in ("RA", "Dec", "Total_flux"))
    del table
    OBS_ID, FIELD = args.obs_id, args.field
    with np.load(args.pointing, allow_pickle=False) as pointing:
        args.beam_ra_deg = pointing["beam_ra_deg"].copy()
        args.beam_dec_deg = pointing["beam_dec_deg"].copy()
    centres = np.column_stack((args.beam_ra_deg[args.beams], args.beam_dec_deg[args.beams]))
    if not np.isfinite(centres).all() or len(np.unique(centres, axis=0)) != len(args.beams):
        raise ValueError("Expected distinct, finite beam centres from the pipeline footprint; the shared MS phase centre is not a beam centre.")
    numbers, starts, ends, origin = read_windows(args.snapshots / "windows.txt", args.first_bin, args.last_bin)
    numbers, starts, ends = available_range(numbers, starts, ends, args.snapshots, args.beams)
    results, processed = {}, []
    failed = False
    for beam in args.beams:
        print(f"SB{OBS_ID} · Beam {beam:02d}", flush=True)
        try:
            results[beam] = measure_beam(beam, numbers, catalogue, args)
        except (ValueError, OSError, np.linalg.LinAlgError) as exc:
            print(f"  Skipping beam {beam:02d}: {exc}", flush=True)
            failed = True
        processed.append(beam)
        save_points(results, processed, numbers, starts, ends, origin, args)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

