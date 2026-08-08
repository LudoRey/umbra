"""Metadata read from the per-exposure stacks that feed the composition."""
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import astropy.io.fits
import numpy as np

from umbra.common import fits
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

    The integration step writes exactly one stack per group, so a group holding anything else
    means stale files are sitting next to the real ones.
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


def read_exposure_times(headers: Sequence[astropy.io.fits.Header], filepaths: Sequence[Path]) -> list[float]:
    """
    Read the exposure time of each stack, falling back to equal times when any stack lacks one.

    Weights are made proportional to exposure time, the inverse-variance weighting of a
    photon-noise-limited signal once the fit has rescaled every stack onto a common brightness.
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
