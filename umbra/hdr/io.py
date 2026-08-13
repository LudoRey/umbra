"""Metadata read from the per-exposure stacks that feed the composition."""
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import astropy.io.fits
import numpy as np

from umbra.common import coords, fits, imageio
from umbra.common.disk import binary_disk
from umbra.common.terminal import cprint


# Read in order; the first keyword a stack carries wins.
EXPOSURE_KEYWORDS = ("EXPTIME", "EXPOSURE")

MOON_KEYWORDS = ("MOON-X", "MOON-Y", "MOON-R")


def single_filepaths(
    grouped_filepaths: dict[tuple[str, ...], list[Path]],
    group_keywords: Sequence[str],
    stacks_dir: str,
) -> list[Path]:
    """
    The one stack of each group, in the order :func:`fits.group_filepaths` returned them.

    Integration writes exactly one stack per group, so a group holding anything else means stale
    files are sitting next to the real ones.
    """
    filepaths = []
    for group_values, group_filepaths in grouped_filepaths.items():
        if len(group_filepaths) != 1:
            group_name = fits.format_group_name(group_values, group_keywords)
            raise ValueError(
                f"Expected one stack per group in {stacks_dir}, found {len(group_filepaths)} for group "
                f"{group_name or '(ungrouped)'}: {', '.join(p.name for p in group_filepaths)}.")
        filepaths.append(group_filepaths[0])
    return filepaths


def measure_black_and_white_points(filepaths: Sequence[Path], moon_interior_factor: float = 0.5) -> tuple[float, float]:
    """
    The two levels the stacks span, read off the ends of ``filepaths``, longest exposure first.

    Only the values in between mean anything: the white point is where the imaging system clips,
    the black point is where the images sit when they hold no signal. The white point is the
    maximum of the longest exposure, which clips over the inner corona. The black point is the
    median inside the moon's disk of the shortest exposure, the one place that holds nothing to
    record: no corona by construction, and no earthshine at that exposure. The frame at large
    would not do, since registration pads it with zeros, and neither would a minimum, which lands
    on the deepest noise excursion or on a dead pixel.

    The disk is shrunk by ``moon_interior_factor`` to keep out the limb, where scattered light and
    any residual registration error sit.
    """
    white_point = float(imageio.read(filepaths[0])[0].max())

    img, header = imageio.read(filepaths[-1])
    center = np.array([header["MOON-X"], header["MOON-Y"]])
    radius = moon_interior_factor * cast(float, header["MOON-R"])
    interior = binary_disk(center, radius, coords.Region.from_shape(img.shape))
    black_point = float(np.median(img[interior]))
    return black_point, white_point


def read_exposure_times(headers: Sequence[astropy.io.fits.Header], filepaths: Sequence[Path]) -> list[float]:
    """
    Read the exposure time of each stack, falling back to equal times when any stack lacks one.

    Exposure time alone, not the exposure-gain product: raising the gain amplifies signal and
    noise together and buys no photons.
    """
    exposure_times = []
    for header, filepath in zip(headers, filepaths):
        value = next((header[keyword] for keyword in EXPOSURE_KEYWORDS if keyword in header), None)
        if value is None:
            cprint(f"{filepath.name} records no exposure time ({' or '.join(EXPOSURE_KEYWORDS)}): "
                   "weighting every stack equally instead.", color="yellow")
            return [1.0] * len(headers)
        exposure_times.append(float(cast(float, value)))
    return exposure_times


def read_moon_geometry(headers: Sequence[astropy.io.fits.Header], stacks_dir: str) -> astropy.io.fits.Header:
    """
    Average the moon's position and radius across the stacks.

    They are all moon-registered against the same reference, so the moon sits at one position for
    the whole set: averaging only settles the differences between per-group means.
    """
    geometry = fits.aggregate(headers, {keyword: np.mean for keyword in MOON_KEYWORDS})
    missing = [keyword for keyword in MOON_KEYWORDS if keyword not in geometry]
    if missing:
        raise ValueError(f"The stacks in {stacks_dir} carry no moon geometry ({', '.join(missing)}); "
                         "they were not produced by the integration step.")
    return geometry
