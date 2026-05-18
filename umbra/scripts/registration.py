import os
from collections.abc import Sequence
from typing import cast
from pathlib import Path

import numpy as np
import dateutil.parser
import scipy.interpolate
import skimage as sk

from umbra import registration
from umbra.common import transform, fits
from umbra.common.terminal import cprint
from umbra.common.typing import CheckStateCallback, ImageCallback


def main(
    # IO
    input_dir: str,
    ref_filename: str | None,
    anchor_filenames: Sequence[str] | None,
    moon_registered_dir: str,
    sun_registered_dir: str,
    group_keywords: Sequence[str],
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
    # Upper bound on the moon radius (coarse estimate used for threshold)
    moon_radius_pixels = 0.279 * 3600 / image_scale
    # Because of brightness variations, the multiplier should be large enough so that a complete annulus is clipped
    num_clipped_pixels = np.pi*(clipped_factor**2 - 1)*moon_radius_pixels**2
    # The number of pixels that correspond to the edge of the moon (given here by its circonference times a multiplier)
    num_edge_pixels = edge_factor*2*np.pi*moon_radius_pixels

    # Load and group filepaths by their keywords
    filepath_headers = fits.read_fits_headers(input_dir)
    grouped_filepaths = fits.get_grouped_filepaths(filepath_headers, group_keywords)

    ### Process reference image
    if not ref_filename:
        ref_filename = registration.auto.select_reference(grouped_filepaths, filepath_headers, group_keywords)
    # Load ref image
    ref_img, ref_header = fits.read_fits_as_float(os.path.join(input_dir, ref_filename), checkstate=checkstate)
    # Moon preprocessing and detection
    ref_processed_img = registration.moon.preprocess(ref_img, num_clipped_pixels, img_callback=img_callback, checkstate=checkstate)
    ref_moon_center, ref_moon_radius = registration.moon.detect_moon(ref_processed_img, num_edge_pixels, img_callback=img_callback, checkstate=checkstate)
    # Extract reference time
    ref_timestamp = dateutil.parser.parse(cast(str, ref_header["DATE-OBS"])).timestamp()
    # Save image
    ref_header = fits.update_header(ref_header, registration.moon.keyword_cards(ref_moon_center, ref_moon_radius))
    fits.save_as_fits(ref_img, ref_header, os.path.join(moon_registered_dir, ref_filename), checkstate=checkstate)
    fits.save_as_fits(ref_img, ref_header, os.path.join(sun_registered_dir, ref_filename), checkstate=checkstate)

    ### Process anchor images
    if not anchor_filenames:
        anchor_filenames = registration.auto.select_anchors(grouped_filepaths, filepath_headers, group_keywords, num_clipped_pixels)
    else:
        if len(anchor_filenames) < 2:
            raise ValueError("At least 2 anchors are required.")
        anchor_filenames = sorted(
            anchor_filenames,
            key=lambda f: cast(str, filepath_headers[Path(os.path.join(input_dir, f))]["DATE-OBS"])
        )
    # Track per-anchor moon detection results and pairwise sun transforms
    timestamps = []
    moon_centers = []
    moon_radii = []
    sun_tforms_pairwise = []  # length N-1; index i: anchor[i] -> anchor[i+1]

    prev_preprocessed_img: np.ndarray | None = None
    prev_mass_center: tuple[float, float] | None = None
    for i, filename in enumerate(anchor_filenames):
        img, header = fits.read_fits_as_float(os.path.join(input_dir, filename), checkstate=checkstate)
        # Moon preprocessing and detection
        processed_img = registration.moon.preprocess(img, num_clipped_pixels, img_callback=img_callback, checkstate=checkstate)
        moon_center, moon_radius = registration.moon.detect_moon(processed_img, num_edge_pixels, img_callback=img_callback, checkstate=checkstate)

        # Sun preprocessing
        preprocessed_img, mass_center = registration.sun.preprocess(processed_img, moon_center, moon_radius, sigma_high_pass_tangential, img_callback=img_callback, checkstate=checkstate)
        # Compute transform
        if prev_preprocessed_img is not None and prev_mass_center is not None:
            tform = registration.sun.compute_transform(prev_preprocessed_img, preprocessed_img, prev_mass_center, max_iter, error_overlay_opacity, img_callback=img_callback, checkstate=checkstate)
            sun_tforms_pairwise.append(tform)

        # Update trackers
        timestamps.append(dateutil.parser.parse(cast(str, header["DATE-OBS"])).timestamp())
        moon_centers.append(moon_center)
        moon_radii.append(moon_radius)

        prev_preprocessed_img = preprocessed_img
        prev_mass_center = mass_center

    # Extract ref-relative transforms from pairwise transforms
    identity = sk.transform.EuclideanTransform()
    sun_tforms_from_anchor_zero = [identity]
    for sun_tform_pairwise in sun_tforms_pairwise:
        sun_tforms_from_anchor_zero.append(cast(sk.transform.EuclideanTransform, sun_tforms_from_anchor_zero[-1] + sun_tform_pairwise))
    sun_tform_from_anchor_zero_to_ref = transform.interp_transforms(timestamps, sun_tforms_from_anchor_zero)(ref_timestamp)
    sun_tforms = [
        cast(sk.transform.EuclideanTransform, sun_tform_from_anchor_zero_to_ref.inverse + t)
        for t in sun_tforms_from_anchor_zero
    ]

    # Compute moon transforms
    moon_tforms = [
        transform.centered_rigid_transform(ref_moon_center, float(t.rotation), moon_center-ref_moon_center)
        for moon_center, t in zip(moon_centers, sun_tforms)
    ]

    for i, filename in enumerate(anchor_filenames):
        # Load image
        img, header = fits.read_fits_as_float(os.path.join(input_dir, filename), checkstate=checkstate)
        # Retrieve computed values
        moon_center = moon_centers[i]
        moon_radius = moon_radii[i]
        moon_tform = moon_tforms[i]
        sun_tform = sun_tforms[i]

        # Apply transforms and save registered images
        moon_registered_img = transform.warp(img, moon_tform.inverse.params) # inverse required for anchor to ref
        sun_registered_img = transform.warp(img, sun_tform.inverse.params) # inverse required for anchor to ref
        moon_registered_header = fits.update_header(header, registration.moon.keyword_cards(ref_moon_center, moon_radius))
        sun_registered_header = fits.update_header(header, registration.moon.keyword_cards(sun_tform.inverse(moon_center)[0], moon_radius))
        fits.save_as_fits(moon_registered_img, moon_registered_header, os.path.join(moon_registered_dir, filename), checkstate=checkstate)
        fits.save_as_fits(sun_registered_img, sun_registered_header, os.path.join(sun_registered_dir, filename), checkstate=checkstate)

    # Compute interpolants
    thetas = np.array([t.rotation for t in sun_tforms])
    sun_moon_translations = np.array([
        registration.sun.compute_sun_moon_translation(sun_tform, moon_tform)
        for sun_tform, moon_tform in zip(sun_tforms, moon_tforms)
    ])
    theta_interp = scipy.interpolate.interp1d(timestamps, thetas, kind='linear', fill_value='extrapolate') # type: ignore (fill_value is mistyped as float)
    sun_moon_translation_interp = scipy.interpolate.interp1d(timestamps, sun_moon_translations, kind='linear', axis=0, fill_value='extrapolate') # type: ignore

    # times_new = np.linspace(times.min(), times.max(), 100)
    # theta_new = theta_interp(times_new)
    # sun_moon_translations_new = sun_moon_translation_interp(times_new)

    # from matplotlib import pyplot as plt
    # fig, axes = plt.subplots(2)
    # axes[0].plot(times_new, theta_new)
    # axes[1].plot(times_new, sun_moon_translations_new)
    # axes[1].plot(times, sun_moon_translations, 'o')
    # plt.show()

    ### Register remaining images through interpolation
    filenames = os.listdir(input_dir)
    filenames = sorted(
        f for f in filenames
        if f.endswith(('.fits', '.fit'))
        and f != ref_filename and f not in anchor_filenames
    )
    for filename in filenames:
        # Load image
        img, header = fits.read_fits_as_float(os.path.join(input_dir, filename), checkstate=checkstate)
        # Moon preprocessing and detection
        processed_img = registration.moon.preprocess(img, num_clipped_pixels, img_callback=img_callback, checkstate=checkstate)
        moon_center, moon_radius = registration.moon.detect_moon(processed_img, num_edge_pixels, img_callback=img_callback, checkstate=checkstate)
        # Interpolate transform parameters
        timestamp = dateutil.parser.parse(cast(str, header["DATE-OBS"])).timestamp()
        theta = theta_interp(timestamp).item() # interp1d's call method always returns an array, even if the input is not one
        sun_moon_translation = sun_moon_translation_interp(timestamp)

        # Compute moon and sun transforms
        moon_tform = transform.centered_rigid_transform(ref_moon_center, theta, moon_center-ref_moon_center)
        sun_tform = transform.translation_transform(sun_moon_translation) + moon_tform

        # Apply transforms and save registered images
        moon_registered_img = transform.warp(img, moon_tform.inverse.params) # inverse required for anchor to ref
        sun_registered_img = transform.warp(img, sun_tform.inverse.params) # inverse required for anchor to ref
        moon_registered_header = fits.update_header(header, registration.moon.keyword_cards(ref_moon_center, moon_radius))
        sun_registered_header = fits.update_header(header, registration.moon.keyword_cards(sun_tform.inverse(moon_center)[0], moon_radius))
        fits.save_as_fits(moon_registered_img, moon_registered_header, os.path.join(moon_registered_dir, filename), checkstate=checkstate)
        fits.save_as_fits(sun_registered_img, sun_registered_header, os.path.join(sun_registered_dir, filename), checkstate=checkstate)


if __name__ == "__main__":
    import sys
    import yaml
    from umbra.common.terminal import ColorTerminalStream
    sys.stdout = ColorTerminalStream()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    main(**config["registration"])
