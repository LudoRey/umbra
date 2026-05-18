from collections.abc import Sequence
from typing import cast
from pathlib import Path

import astropy.io.fits
import numpy as np

from umbra.common import fits
from umbra.common.terminal import cprint

def select_reference(
    grouped_filepaths: dict[tuple[str, ...], list[Path]],
    filepath_headers: dict[Path, astropy.io.fits.Header],
    group_keywords: Sequence[str]
) -> str:
    """Returns the middle image (by timestamp) from the darkest exposure group."""
    cprint("Selecting reference image:", style="bold")
    group_keyword_values, group_filepaths = next(iter(grouped_filepaths.items()))
    group_identifier = ', '.join([f'{k}={v}' for k, v in zip(group_keywords, group_keyword_values)])
    sorted_filepaths = sorted(group_filepaths, key=lambda p: cast(str, filepath_headers[p]["DATE-OBS"]))
    middle_filepath = sorted_filepaths[len(sorted_filepaths) // 2]
    reference_filename = middle_filepath.name
    print(f"Image from group {group_identifier}:")
    print(f"- {reference_filename}")
    return reference_filename

def select_anchors(
    grouped_filepaths: dict[tuple[str, ...], list[Path]],
    filepath_headers: dict[Path, astropy.io.fits.Header],
    group_keywords: Sequence[str],
    num_bright_pixels: float,
    bright_relative_threshold: float = 0.8
) -> list[str]:
    """Selects two anchor images: the first and last images (by timestamp) from the exposure group that satisfies the following criteria:
    - Images from that group must contain at least a certain number of "bright" pixels. (`num_bright_pixels`)
    - A pixel is considered "bright" if its value is above `bright_relative_threshold` of the maximum pixel value in the image.
    Basically, it's a measure of how saturated the image is. Typically, num_bright_pixels is taken from num_clipped_pixels (see moon module)
    """
    cprint("Selecting anchor images:", style="bold")
    anchor_filenames = []
    for group_keyword_values, group_filepaths in grouped_filepaths.items():
        group_identifier = ', '.join([f'{k}={v}' for k, v in zip(group_keywords, group_keyword_values)])
        sorted_filepaths = sorted(group_filepaths, key=lambda p: cast(str, filepath_headers[p]["DATE-OBS"]))
        first_filepath, last_filepath = sorted_filepaths[0], sorted_filepaths[-1]
        img, _ = fits.read_fits_as_float(first_filepath, verbose=False)
        bright_pixel_count = np.sum(img >= bright_relative_threshold * img.max())
        if bright_pixel_count >= num_bright_pixels:
            anchor_filepaths = [first_filepath, last_filepath]
            anchor_filenames = [p.name for p in anchor_filepaths]
            print(f"Images from group {group_identifier}:")
            print("\n".join(f"- {p}" for p in anchor_filenames))
            break
    if not anchor_filenames:
        raise RuntimeError("Automatic anchor selection failed: no group found with a sufficient number of bright pixels. Please select anchors manually.")
    return anchor_filenames
