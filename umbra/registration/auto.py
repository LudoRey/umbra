from collections.abc import Sequence
from pathlib import Path

import numpy as np

from umbra.common import fits
from umbra.common.terminal import cprint

def select_reference(
    input_dir: Path,
    group_keywords: Sequence[str]
) -> str:
    """Returns the middle image (by timestamp) from the darkest exposure group."""
    cprint("Selecting reference image:", style="bold", color="cyan")
    filepath_to_header = {p: fits.read_fits_header(p) for p in fits.list_fits_filepaths(input_dir)}
    grouped_filepaths = fits.get_grouped_filepaths(filepath_to_header, group_keywords)
    group_keyword_values, group_filepaths = next(iter(grouped_filepaths.items()))
    group_identifier = ', '.join([f'{k}={v}' for k, v in zip(group_keywords, group_keyword_values)])
    if group_identifier:
        cprint(f"Selected group: {group_identifier}")
    sorted_filepaths = sorted(group_filepaths, key=lambda p: fits.extract_timestamp(filepath_to_header[p]))
    middle_filepath = sorted_filepaths[len(sorted_filepaths) // 2]
    reference_filename = middle_filepath.name
    print(f"Selected reference image: {reference_filename}")
    cprint(f"Reference image selected successfully.", color="green")
    return reference_filename

def select_anchors(
    input_dir: Path,
    group_keywords: Sequence[str],
    num_bright_pixels: float,
    bright_relative_threshold: float = 0.8
) -> list[str]:
    """Selects two anchor images: the first and last images (by timestamp) from the exposure group that satisfies the following criteria:
    - Images from that group must contain at least a certain number of "bright" pixels. (`num_bright_pixels`)
    - A pixel is considered "bright" if its value is above `bright_relative_threshold` of the maximum pixel value in the image.
    Basically, it's a measure of how saturated the image is. Typically, num_bright_pixels is taken from num_clipped_pixels (see moon module)
    """
    cprint("Selecting anchor images:", style="bold", color="cyan")
    filepath_to_header = {p: fits.read_fits_header(p) for p in fits.list_fits_filepaths(input_dir)}
    grouped_filepaths = fits.get_grouped_filepaths(filepath_to_header, group_keywords)
    anchor_filenames = []
    for group_keyword_values, group_filepaths in grouped_filepaths.items():
        group_identifier = ', '.join([f'{k}={v}' for k, v in zip(group_keywords, group_keyword_values)])
        sorted_filepaths = sorted(group_filepaths, key=lambda p: fits.extract_timestamp(filepath_to_header[p]))
        first_filepath, last_filepath = sorted_filepaths[0], sorted_filepaths[-1]
        img, _ = fits.read_fits_as_float(first_filepath, verbose=False)
        bright_pixel_count = np.sum(img >= bright_relative_threshold * img.max())
        if bright_pixel_count >= num_bright_pixels:
            if group_identifier:
                cprint(f"Selected group: {group_identifier}")
            anchor_filepaths = [first_filepath, last_filepath]
            anchor_filenames = [p.name for p in anchor_filepaths]
            print(f"Selected anchor images:")
            print("\n".join(f"- {p}" for p in anchor_filenames))
            break
    if not anchor_filenames:
        raise RuntimeError("Automatic anchor selection failed: no group found with a sufficient number of bright pixels. Please select anchors manually.")
    cprint(f"Anchor images selected successfully.", color="green")
    return anchor_filenames
