from collections.abc import Sequence
import numpy as np
from pathlib import Path

from umbra.common import fits, imageio
from umbra.common.terminal import cprint
from umbra.registration import pipeline


def main(
    # IO
    fits_dir: str | Path,
    ref_filename: str | None,
    anchor_filenames: Sequence[str] | None,
    group_keywords: Sequence[str],
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
) -> None:
    num_clipped_pixels, num_edge_pixels = pipeline.compute_moon_detection_params(image_scale, clipped_factor, edge_factor)
    fits_dir, moon_registered_dir, sun_registered_dir = map(Path, (fits_dir, moon_registered_dir, sun_registered_dir))

    ref_filename = pipeline.resolve_ref_filename(ref_filename, fits_dir, group_keywords)
    anchor_filenames = pipeline.resolve_anchor_filenames(anchor_filenames, fits_dir, group_keywords, num_clipped_pixels)
    remaining_filenames = pipeline.resolve_remaining_filenames(fits_dir, ref_filename, anchor_filenames)

    ### Process anchor images: compute moon/sun transforms (in first anchor frame) and other values
    anchor_headers, moon_centers, moon_radii, sun_tforms_from_first_anchor, moon_tforms_from_first_anchor = pipeline.process_anchors(
        anchor_filenames, fits_dir, num_clipped_pixels, num_edge_pixels,
        sigma_high_pass_tangential, max_iter, error_overlay_opacity,
    )

    ### Process reference image and compute moon/sun transforms from first anchor to reference
    cprint(f"Processing reference image: {ref_filename}", style='bold', color='cyan')
    ref_img, ref_header = imageio.read(fits_dir / ref_filename)
    if ref_filename not in anchor_filenames:
        _, ref_moon_center, ref_moon_radius = pipeline.preprocess_and_detect_moon(ref_img, num_clipped_pixels, num_edge_pixels)
        ref_timestamp = fits.extract_timestamp(ref_header)
        anchor_timestamps = [fits.extract_timestamp(header) for header in anchor_headers]
        moon_tform_from_first_anchor_to_ref, sun_tform_from_first_anchor_to_ref = pipeline.interpolate_transforms_to_ref(
            sun_tforms_from_first_anchor, moon_tforms_from_first_anchor, moon_centers[0], anchor_timestamps, ref_moon_center, ref_timestamp,
        )
    else:
        cprint(f"Reference image is also an anchor. Extracting precomputed values.")
        anchor_index = anchor_filenames.index(ref_filename)
        ref_moon_center, ref_moon_radius = moon_centers[anchor_index], moon_radii[anchor_index]
        moon_tform_from_first_anchor_to_ref = moon_tforms_from_first_anchor[anchor_index]
        sun_tform_from_first_anchor_to_ref = sun_tforms_from_first_anchor[anchor_index]
    ref_moon_header, ref_sun_header = pipeline.update_headers(ref_header, ref_moon_center, ref_moon_radius)
    imageio.write(moon_registered_dir / ref_filename, ref_img, ref_moon_header)
    imageio.write(sun_registered_dir / ref_filename, ref_img, ref_sun_header)
    cprint(f"Reference image processed successfully.", color='green')

    ### Re-base transforms from reference to anchors, and extract anchor values (rotations and sun-moon translations)
    sun_tforms = pipeline.rebase_transforms_to_ref(sun_tform_from_first_anchor_to_ref, sun_tforms_from_first_anchor)
    moon_tforms = pipeline.rebase_transforms_to_ref(moon_tform_from_first_anchor_to_ref, moon_tforms_from_first_anchor)
    rotations, sun_moon_translations = pipeline.extract_anchor_values(sun_tforms, moon_tforms)
    
    ### Register anchor images using precomputed values
    for i, filename in enumerate(anchor_filenames):
        cprint(f"Registering anchor image {filename} ({i+1}/{len(anchor_filenames)}):", style='bold', color='cyan')
        img, header = imageio.read(fits_dir / filename)
        moon_center, moon_radius = moon_centers[i], moon_radii[i]
        cprint(f"Extracting anchor values:", style='bold')
        rotation, sun_moon_translation  = rotations[i], sun_moon_translations[i]
        print(f"- Rotation          : {np.rad2deg(rotation):>8.3f} deg")
        print(f"- Sun-moon delta (x): {sun_moon_translation[0]:>8.2f} px")
        print(f"- Sun-moon delta (y): {sun_moon_translation[1]:>8.2f} px")
        moon_tform, sun_tform = pipeline.recreate_transforms(rotation, sun_moon_translation, ref_moon_center, moon_center)
        moon_registered_img, moon_header, sun_registered_img, sun_header = pipeline.apply_transforms(
            img, header, moon_center, moon_radius, moon_tform, sun_tform,
        )
        imageio.write(moon_registered_dir / filename, moon_registered_img, moon_header)
        imageio.write(sun_registered_dir / filename, sun_registered_img, sun_header)
        cprint(f"Anchor image {filename} registered successfully ({i+1}/{len(anchor_filenames)}).", color='green')

    ### Register remaining images through moon detection + timestamp-based interpolation of anchor values
    if len(remaining_filenames) > 0:
        ref_timestamp = fits.extract_timestamp(ref_header)
        anchor_timestamps = [fits.extract_timestamp(header) for header in anchor_headers]
        rotation_interp, sun_moon_translation_interp = pipeline.build_anchor_values_interpolants(anchor_timestamps, sun_moon_translations, rotations)
        for i, filename in enumerate(remaining_filenames):
            cprint(f"Registering non-anchor image {filename} ({i+1}/{len(remaining_filenames)}):", style='bold', color='cyan')
            img, header = imageio.read(fits_dir / filename)
            _, moon_center, moon_radius = pipeline.preprocess_and_detect_moon(img, num_clipped_pixels, num_edge_pixels)
            cprint(f"Interpolating:", style='bold')
            timestamp = fits.extract_timestamp(header)
            print(f"Time delta: {timestamp - ref_timestamp} sec")
            rotation, sun_moon_translation = pipeline.interpolate_anchor_values(timestamp, rotation_interp, sun_moon_translation_interp)
            print(f"Interpolated anchor values:")
            print(f"- Rotation          : {np.rad2deg(rotation):>8.3f} deg")
            print(f"- Sun-moon delta (x): {sun_moon_translation[0]:>8.2f} px")
            print(f"- Sun-moon delta (y): {sun_moon_translation[1]:>8.2f} px")
            moon_tform, sun_tform = pipeline.recreate_transforms(rotation, sun_moon_translation, ref_moon_center, moon_center)
            moon_registered_img, moon_header, sun_registered_img, sun_header = pipeline.apply_transforms(
                img, header, moon_center, moon_radius, moon_tform, sun_tform,
            )
            imageio.write(moon_registered_dir / filename, moon_registered_img, moon_header)
            imageio.write(sun_registered_dir / filename, sun_registered_img, sun_header)
            cprint(f"Image {filename} registered successfully ({i+1}/{len(remaining_filenames)}).", color='green')
    cprint(f"Registration completed successfully.", style='bold', color='green')


if __name__ == "__main__":
    import sys
    import yaml
    from umbra.common.terminal import ColorTerminalStream
    sys.stdout = ColorTerminalStream()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    main(**config["registration"])
