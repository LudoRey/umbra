import os
from collections.abc import Sequence
from typing import cast

import numpy as np

from umbra.common import context, coords, fits, imageio
from umbra.common.disk import binary_disk
from umbra.common.polar import angle_map, radius_map
from umbra.common.terminal import cprint
from umbra.hdr import equalization, io, weighting


def main(
    # IO
    stacks_dir: str,
    hdr_dir: str,
    group_keywords: Sequence[str],
    # Weighting
    low_threshold: float,
    low_smoothness: float,
    high_threshold: float,
    high_smoothness: float,
    # Diagnostics
    save_weights: bool,
    # Output
    output_format: imageio.Format = "float32",
) -> None:
    """
    Combine the per-exposure stacks produced by integration.py into a single HDR image.

    Each stack sees only part of the corona: the longest exposure saturates over the inner
    corona, the shortest drowns the outer corona in noise. Walking from the longest exposure
    to the shortest, every stack is fitted onto the brightness scale of the one before it and
    added to a weighted average, so that each pixel is drawn from the exposures that recorded
    it within their usable range.

    The fit (:func:`umbra.hdr.equalization.equalize_brightness`) absorbs the exposure ratio
    itself, along with the illumination variations over the course of totality that a nominal
    ratio cannot describe. The four weighting parameters are fractions of the range between the
    black and white points (:func:`umbra.hdr.io.measure_black_and_white_points`), not absolute
    pixel values.
    """
    filepath_to_header = {p: imageio.read_header(p) for p in imageio.list_files(stacks_dir, extensions=imageio.extensions.FITS)}
    grouped_filepaths = fits.group_filepaths(filepath_to_header, group_keywords)

    # Groups come in ascending keyword order; compositing runs the other way, from the longest
    # exposure down. That the keyword order really is exposure order is checked as each stack is read.
    group_values_list = list(grouped_filepaths.keys())[::-1]
    filepaths = io.single_filepaths(grouped_filepaths, group_keywords, stacks_dir)[::-1]
    if len(filepaths) < 2:
        raise ValueError(
            f"HDR composition needs at least two stacks, found {len(filepaths)} in {stacks_dir}. "
            "Group the images by an exposure keyword during integration to produce one stack per exposure.")
    headers = [filepath_to_header[filepath] for filepath in filepaths]

    shape = fits.extract_shape(headers[0])
    if len(shape) != 3:
        raise ValueError(f"HDR composition expects colour stacks, got an image of shape {shape} in {stacks_dir}.")

    geometry = io.read_moon_geometry(headers, stacks_dir)
    exposure_times = io.read_exposure_times(headers, filepaths)

    center = np.array([geometry["MOON-X"], geometry["MOON-Y"]])
    moon_radius = cast(float, geometry["MOON-R"])
    # Padded past the limb: everything the moon swept over during totality
    padded_radius = equalization.MOON_RADIUS_FACTOR * moon_radius
    moon_mask = binary_disk(center, padded_radius, coords.Region.from_shape(shape))
    img_theta = angle_map(center[0], center[1], shape=shape[:2])
    img_radius = radius_map(center[0], center[1], shape=shape[:2])

    cprint("Measuring the black and white points:", style="bold", color="cyan")
    black_point, white_point = io.measure_black_and_white_points(filepaths)
    dynamic_range = white_point - black_point
    low_threshold = black_point + low_threshold * dynamic_range
    high_threshold = black_point + high_threshold * dynamic_range
    low_smoothness, high_smoothness = low_smoothness * dynamic_range, high_smoothness * dynamic_range
    cprint(f"Detected a black point at {black_point:.5f} and a white point at {white_point:.5f}.")
    cprint(f"Only pixels between {low_threshold:.5f} and {high_threshold:.5f} will be kept.")

    num_stacks = len(filepaths)
    hdr_img = np.zeros(shape, dtype=np.float32)
    sum_weights = np.zeros(shape[:2], dtype=np.float32)
    previous_mean = np.inf
    img_y, valid_y = None, None
    for index, filepath in enumerate(filepaths):
        cprint(f"Compositing {filepath.name} ({index + 1}/{num_stacks}):", style="bold", color="cyan")
        img_x, _ = imageio.read(filepath)

        # The groups were sorted by keyword value, never by measured brightness. Check the two agree
        # before fitting this stack onto a scale that could not represent it.
        current_mean = float(img_x.mean())
        if current_mean >= previous_mean:
            raise ValueError(
                f"{filepath.name} (mean {current_mean:.5f}) is not fainter than {filepaths[index-1].name} "
                f"(mean {previous_mean:.5f}): the stacks are not sorted by decreasing exposure. Group by a "
                "keyword whose sort order matches irradiance, such as the exposure time.")
        previous_mean = current_mean

        valid_x = weighting.in_range(img_x, low_threshold, high_threshold)
        # Each end of the ladder keeps the pixels no other stack measures: the longest exposure its
        # dark ones, the shortest its bright ones. The fit mask above still rejects both.
        low = (0.0, 0.0) if index == 0 else (low_threshold, low_smoothness)
        high = (1.0, 0.0) if index == num_stacks - 1 else (high_threshold, high_smoothness)
        weights = weighting.saturation_weighting(img_x, *low, *high)
        if save_weights:
            group_name = fits.format_group_name(group_values_list[index], group_keywords)
            imageio.write(os.path.join(hdr_dir, f"weights_{group_name}.fits"), weights, None, format=output_format)
        weights = weights * exposure_times[index]
        context.checkstate()

        # The longest exposure anchors the brightness scale, so there is nothing to fit it onto.
        if index > 0:
            assert img_y is not None and valid_y is not None
            cprint("Equalizing the brightness against the previous stack...", flush=True)
            img_x = equalization.equalize_brightness(img_x, img_theta, img_radius, img_y,
                                                     valid_x & valid_y & ~moon_mask, center, moon_radius)
            cprint("Brightness equalized.")
            context.checkstate()

        hdr_img += weights[:, :, None] * img_x
        sum_weights += weights
        context.emit_image(weighting.running_composite(hdr_img, sum_weights))
        cprint(f"Composited {filepath.name} ({index + 1}/{num_stacks}).", color="green")

        # The equalized stack carries the reference scale forward: it is what the next one is fitted onto.
        img_y, valid_y = img_x, valid_x

    uncovered = int(np.count_nonzero(sum_weights == 0))
    if uncovered:
        raise ValueError(
            f"{uncovered} pixels fall outside the usable range of every stack: saturated in the longer "
            "exposures and below the noise floor in the shorter ones. Widen the gap between the weighting "
            "thresholds, or shoot an intermediate exposure.")
    hdr_img /= sum_weights[:, :, None]

    # Fitting onto the longest exposure's scale pushes the inner corona well past 1, so the
    # composite is rescaled to [0,1]
    divisor = float(hdr_img.max())
    hdr_img /= divisor

    output_header = fits.combine([fits.intersect(headers), geometry])
    imageio.write(os.path.join(hdr_dir, "hdr.fits"), hdr_img, output_header, format=output_format)
    context.emit_image(hdr_img)
    cprint("HDR composition completed successfully.", style="bold", color="green")


if __name__ == "__main__":
    import sys
    import yaml
    from umbra.common.terminal import ColorTerminalStream
    sys.stdout = ColorTerminalStream()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    main(**config["hdr"])
