import os
from collections.abc import Sequence
from typing import cast

import astropy.io.fits
import numpy as np

from umbra.common import context, coords, fits, imageio
from umbra.common.disk import binary_disk
from umbra.common.polar import angle_map
from umbra.common.terminal import cprint
from umbra.hdr import equalization, io, weighting


def main(
    # IO
    stacks_dir: str,
    hdr_dir: str,
    group_keywords: Sequence[str],
    # Clipping
    low_threshold: float,
    low_smoothness: float,
    high_threshold: float,
    high_smoothness: float,
    # Diagnostics
    save_weights: bool,
) -> None:
    """
    Combine the per-exposure stacks produced by integration.py into a single HDR image.

    Each stack sees only part of the corona: the longest exposure saturates over the inner
    corona, the shortest drowns the outer corona in noise. Walking from the longest exposure
    to the shortest, every stack is fitted onto the brightness scale of the one before it and
    added to a weighted average, so that each pixel is drawn from the exposures that recorded
    it within their usable range.

    The fit (:func:`umbra.hdr.equalization.equalize_brightness`) absorbs the exposure ratio
    itself rather than dividing it out beforehand: an affine map whose offset and slope both
    vary with the angle around the moon, which also soaks up the transparency and sky-gradient
    differences that a nominal exposure ratio cannot describe.

    The four clipping parameters are fractions of the level at which the imaging system clips,
    measured as the maximum of the longest exposure, rather than absolute pixel values.
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
    # Padded past the limb: everything the moon swept over during totality, plus the registration
    # residuals, lunar relief and leaked corona that make the pixels just outside it
    # unrepresentative of the corona the fit is meant to describe.
    moon_radius = equalization.MOON_RADIUS_FACTOR * cast(float, geometry["MOON-R"])
    moon_mask = binary_disk(center, moon_radius, coords.Region.from_shape(shape))
    img_theta = angle_map(center[0], center[1], shape=shape[:2])

    # The longest exposure anchors the brightness scale every other stack is fitted onto. Its own
    # dark pixels are the only measurement of the outer corona, so it keeps them: no low cut.
    cprint(f"Reading the reference stack {filepaths[0].name}:", style="bold", color="cyan")
    img_y, _ = imageio.read(filepaths[0])
    previous_mean = float(img_y.mean())

    # The longest exposure saturates over the inner corona, so its maximum is the level at which
    # this imaging system clips. The four clipping parameters are fractions of it and become pixel
    # values here: that level depends on the sensor's full well and on what calibration did to it,
    # and is not knowable from the settings alone.
    saturation = float(img_y.max())
    low_threshold, high_threshold = low_threshold * saturation, high_threshold * saturation
    low_smoothness, high_smoothness = low_smoothness * saturation, high_smoothness * saturation
    cprint(f"Detected saturation value at {saturation:.5f}.")
    cprint(f"Only pixels between {low_threshold:.5f} and {high_threshold:.5f} will be kept.")

    valid_y = weighting.in_range(img_y, low_threshold, high_threshold)
    weights = weighting.saturation_weighting(img_y, 0, high_threshold, 0, high_smoothness)
    if save_weights:
        group_name = fits.format_group_name(group_values_list[0], group_keywords)
        imageio.write(os.path.join(hdr_dir, f"weights_{group_name}.fits"), weights, None)
    weights = weights * exposure_times[0]

    hdr_img = weights[:, :, None] * img_y
    sum_weights = weights.copy()
    context.emit_image(weighting.running_composite(hdr_img, sum_weights))

    num_remaining = len(filepaths) - 1
    for index, filepath in enumerate(filepaths[1:], start=1):
        cprint(f"Compositing {filepath.name} ({index}/{num_remaining}):", style="bold", color="cyan")
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
        # The shortest exposure is the only measurement of the inner corona, so it keeps its bright
        # pixels; every other stack is superseded there by a shorter one. Dropping the bound outright
        # is safe on the last pass, but only below the fit mask, which still rejects saturated pixels.
        if index == num_remaining:
            high_threshold, high_smoothness = 1.0, 0.0
        weights = weighting.saturation_weighting(img_x, low_threshold, high_threshold, low_smoothness, high_smoothness)
        if save_weights:
            group_name = fits.format_group_name(group_values_list[index], group_keywords)
            imageio.write(os.path.join(hdr_dir, f"weights_{group_name}.fits"), weights, None)
        weights = weights * exposure_times[index]
        context.checkstate()

        cprint("Equalizing the brightness against the previous stack...", color="cyan", flush=True)
        img_x = equalization.equalize_brightness(img_x, img_theta, img_y, valid_x & valid_y & ~moon_mask)
        cprint("Brightness equalized.", color="green")
        context.checkstate()

        hdr_img += weights[:, :, None] * img_x
        sum_weights += weights
        context.emit_image(weighting.running_composite(hdr_img, sum_weights))
        cprint(f"Composited {filepath.name} ({index}/{num_remaining}).", color="green")

        # The equalized stack carries the reference scale forward: it is what the next one is fitted onto.
        img_y, valid_y = img_x, valid_x

    uncovered = int(np.count_nonzero(sum_weights == 0))
    if uncovered:
        raise ValueError(
            f"{uncovered} pixels fall outside the usable range of every stack: saturated in the longer "
            "exposures and below the noise floor in the shorter ones. Widen the gap between the clipping "
            "thresholds, or shoot an intermediate exposure.")
    hdr_img /= sum_weights[:, :, None]

    # Fitting onto the longest exposure's scale pushes the inner corona well past 1, so the
    # composite is rescaled to [0,1] and the divisor recorded: multiply by it to undo.
    divisor = float(hdr_img.max())
    hdr_img /= divisor

    output_header = fits.update(
        fits.combine([fits.intersect(headers), geometry]),
        [astropy.io.fits.Card("HDRSCALE", divisor, "Multiply by this to undo normalization.")])
    imageio.write(os.path.join(hdr_dir, "hdr.fits"), hdr_img, output_header)
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
