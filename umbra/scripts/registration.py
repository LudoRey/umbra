import os
from collections.abc import Sequence
from typing import cast

import numpy as np
import dateutil.parser
import scipy.interpolate

from umbra import registration
from umbra.common import transform, interpolation, fits
from umbra.common.terminal import cprint
from umbra.common.typing import CheckStateCallback, ImageCallback


def main(
    # IO
    input_dir: str,
    ref_filename: str,
    anchor_filenames: Sequence[str],
    moon_registered_dir: str,
    sun_registered_dir: str,
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

    # Initialize trackers to store the alignment results of the anchors. They will be used to align all other images.
    # All quantities are given from the reference to the anchor; first element corresponds to the reference and is thus left at 0.
    times = np.zeros(len(anchor_filenames) + 1) # Elapsed time
    thetas = np.zeros(len(anchor_filenames) + 1) # Rotation angle
    sun_moon_translations = np.zeros((len(anchor_filenames) + 1, 2)) # Relative translation of the sun with respect to the moon

    # Load ref image
    ref_img, ref_header = fits.read_fits_as_float(os.path.join(input_dir, ref_filename), checkstate=checkstate)
    # Moon preprocessing and detection
    ref_processed_img = registration.moon.preprocess(ref_img, num_clipped_pixels, img_callback=img_callback, checkstate=checkstate)
    ref_moon_center, ref_moon_radius = registration.moon.detect_moon(ref_processed_img, num_edge_pixels, img_callback=img_callback, checkstate=checkstate)
    # Sun preprocessing
    ref_processed_img, ref_mass_center = registration.sun.preprocess(ref_processed_img, ref_moon_center, ref_moon_radius, sigma_high_pass_tangential, img_callback=img_callback, checkstate=checkstate)
    # Extract reference time for later use in trackers
    ref_time_str = cast(str, ref_header["DATE-OBS"])
    # Save image
    ref_header = fits.update_header(ref_header, registration.moon.keyword_cards(ref_moon_center, ref_moon_radius))
    fits.save_as_fits(ref_img, ref_header, os.path.join(moon_registered_dir, ref_filename), checkstate=checkstate)
    fits.save_as_fits(ref_img, ref_header, os.path.join(sun_registered_dir, ref_filename), checkstate=checkstate)

    for i, filename in enumerate(anchor_filenames, start=1):
        # Load anchor image
        img, header = fits.read_fits_as_float(os.path.join(input_dir, filename), checkstate=checkstate)
        # Moon preprocessing and detection
        processed_img = registration.moon.preprocess(img, num_clipped_pixels, img_callback=img_callback, checkstate=checkstate)
        moon_center, moon_radius = registration.moon.detect_moon(processed_img, num_edge_pixels, img_callback=img_callback, checkstate=checkstate)

        # Sun preprocessing
        processed_img, _ = registration.sun.preprocess(processed_img, moon_center, moon_radius, sigma_high_pass_tangential, img_callback=img_callback, checkstate=checkstate)
        # Compute transform parameters
        theta, tx, ty = registration.sun.compute_transform(ref_processed_img, processed_img, ref_mass_center, max_iter, error_overlay_opacity, img_callback=img_callback, checkstate=checkstate)
        
        # Compute moon and sun transforms
        moon_tform = transform.centered_rigid_transform(ref_moon_center, theta, moon_center-ref_moon_center) # ref to anchor
        sun_tform = transform.centered_rigid_transform(ref_mass_center, theta, (tx,ty)) # ref to anchor

        # Update trackers
        time_str = cast(str, header["DATE-OBS"])
        times[i] = (dateutil.parser.parse(time_str) - dateutil.parser.parse(ref_time_str)).total_seconds()
        thetas[i] = theta
        sun_moon_translations[i] = registration.sun.compute_sun_moon_translation(sun_tform, moon_tform)
        cprint(f"Reference -> Anchor {i}:", style='bold')
        print(f"- Elapsed time      : {times[i]:>8.0f} sec")
        print(f"- Rotation          : {np.rad2deg(thetas[i]):>8.3f} deg")
        print(f"- Sun/moon delta (x): {sun_moon_translations[i][0]:>8.2f} px")
        print(f"- Sun/moon delta (y): {sun_moon_translations[i][1]:>8.2f} px")

        # Apply transforms and save registered images
        moon_registered_img = transform.warp(img, moon_tform.inverse.params) # inverse required for anchor to ref
        sun_registered_img = transform.warp(img, sun_tform.inverse.params) # inverse required for anchor to ref
        moon_registered_header = fits.update_header(header, registration.moon.keyword_cards(ref_moon_center, moon_radius))
        sun_registered_header = fits.update_header(header, registration.moon.keyword_cards(sun_tform.inverse(moon_center)[0], moon_radius))
        fits.save_as_fits(moon_registered_img, moon_registered_header, os.path.join(moon_registered_dir, filename), checkstate=checkstate)
        fits.save_as_fits(sun_registered_img, sun_registered_header, os.path.join(sun_registered_dir, filename), checkstate=checkstate)

    theta_interp = scipy.interpolate.interp1d(times, thetas, kind='linear', fill_value='extrapolate') # type: ignore (fill_value is mistyped as float)
    sun_moon_translation_interp = interpolation.LinearFitInterp(times, sun_moon_translations)

    # times_new = np.linspace(times.min(), times.max(), 100)
    # theta_new = theta_interp(times_new)
    # sun_moon_translations_new = sun_moon_translation_interp(times_new)

    # from matplotlib import pyplot as plt
    # fig, axes = plt.subplots(2)
    # axes[0].plot(times_new, theta_new)
    # axes[1].plot(times_new, sun_moon_translations_new)
    # axes[1].plot(times, sun_moon_translations, 'o')
    # plt.show()

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
        time_str = cast(str, header["DATE-OBS"])
        time = (dateutil.parser.parse(time_str) - dateutil.parser.parse(ref_time_str)).total_seconds()
        theta = theta_interp(time).item() # interp1d's call method always returns an array, even if the input is not one
        sun_moon_translation = sun_moon_translation_interp(time)

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
