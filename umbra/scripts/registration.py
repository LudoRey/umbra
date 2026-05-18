from collections.abc import Sequence
from typing import cast
from pathlib import Path

from umbra.common import fits
from umbra.common.typing import CheckStateCallback, ImageCallback
from umbra.registration import pipeline


def main(
    # IO
    input_dir: str | Path,
    ref_filename: str | None,
    anchor_filenames: Sequence[str] | None,
    group_keywords: Sequence[str] | None,
    moon_registered_dir: str | Path,
    sun_registered_dir: str | Path,
    # Moon detection
    image_scale: float,
    clipped_factor: float,
    edge_factor: float,
    # Sun registration
    sigma_high_pass_tangential: float,
    max_iter: int,
    # GUI interactions
    error_overlay_opacity: float,
    *,
    img_callback: ImageCallback = lambda _img: None,
    checkstate: CheckStateCallback = lambda: None,
) -> None:
    num_clipped_pixels, num_edge_pixels = pipeline.compute_moon_detection_params(image_scale, clipped_factor, edge_factor)
    input_dir, moon_registered_dir, sun_registered_dir = map(Path, (input_dir, moon_registered_dir, sun_registered_dir))

    ### Reference image
    ref_filename = pipeline.resolve_ref_filename(ref_filename, input_dir, group_keywords)
    ref_img, ref_header = fits.read_fits_as_float(input_dir / ref_filename, checkstate=checkstate)
    _, ref_moon_center, ref_moon_radius = pipeline.preprocess_and_detect_moon(ref_img, num_clipped_pixels, num_edge_pixels, img_callback=img_callback, checkstate=checkstate)
    ref_timestamp = fits.extract_timestamp(ref_header)
    ref_moon_header, ref_sun_header = pipeline.update_headers(ref_header, ref_moon_center, ref_moon_radius)
    fits.save_as_fits(ref_img, ref_moon_header, moon_registered_dir / ref_filename, checkstate=checkstate)
    fits.save_as_fits(ref_img, ref_sun_header, sun_registered_dir / ref_filename, checkstate=checkstate)

    ### Anchor images
    anchor_filenames = pipeline.resolve_anchor_filenames(anchor_filenames, input_dir, group_keywords, num_clipped_pixels)
    timestamps, moon_centers, moon_radii, sun_tforms_pairwise = pipeline.process_anchors(
        anchor_filenames, input_dir, num_clipped_pixels, num_edge_pixels,
        sigma_high_pass_tangential, max_iter, error_overlay_opacity,
        img_callback=img_callback, checkstate=checkstate,
    )
    sun_tforms = pipeline.compute_sun_transforms(timestamps, sun_tforms_pairwise, ref_timestamp)
    moon_tforms = pipeline.compute_moon_transforms(ref_moon_center, moon_centers, sun_tforms)
    theta_interp, sun_moon_translation_interp = pipeline.build_interpolants(timestamps, sun_tforms, moon_tforms)

    for i, filename in enumerate(anchor_filenames):
        img, header = fits.read_fits_as_float(input_dir / filename, checkstate=checkstate)
        moon_registered_img, moon_header, sun_registered_img, sun_header = pipeline.apply_transforms(
            img, header, moon_centers[i], moon_radii[i], moon_tforms[i], sun_tforms[i],
        )
        fits.save_as_fits(moon_registered_img, moon_header, moon_registered_dir / filename, checkstate=checkstate)
        fits.save_as_fits(sun_registered_img, sun_header, sun_registered_dir / filename, checkstate=checkstate)

    ### Remaining images
    remaining_filenames = pipeline.resolve_remaining_filenames(input_dir, ref_filename, anchor_filenames)
    for filename in remaining_filenames:
        img, header = fits.read_fits_as_float(input_dir / filename, checkstate=checkstate)
        _, moon_center, moon_radius = pipeline.preprocess_and_detect_moon(img, num_clipped_pixels, num_edge_pixels, img_callback=img_callback, checkstate=checkstate)
        timestamp = fits.extract_timestamp(header)
        moon_tform, sun_tform = pipeline.interp_transforms(timestamp, theta_interp, sun_moon_translation_interp, ref_moon_center, moon_center)
        moon_registered_img, moon_header, sun_registered_img, sun_header = pipeline.apply_transforms(
            img, header, moon_center, moon_radius, moon_tform, sun_tform,
        )
        fits.save_as_fits(moon_registered_img, moon_header, moon_registered_dir / filename, checkstate=checkstate)
        fits.save_as_fits(sun_registered_img, sun_header, sun_registered_dir / filename, checkstate=checkstate)


if __name__ == "__main__":
    import sys
    import yaml
    from umbra.common.terminal import ColorTerminalStream
    sys.stdout = ColorTerminalStream()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    main(**config["registration"])
