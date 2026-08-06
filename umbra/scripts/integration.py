import os
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np

from umbra.common import context, fits, imageio
from umbra.common.terminal import cprint
from umbra import integration


# @trackers.track_info
def main(
    # IO
    moon_registered_dir: str,
    sun_registered_dir: str,
    stacks_dir: str,
    group_keywords: Sequence[str],
    # Outlier rejection
    outlier_threshold: float | None,
    # Moon rejection
    rejection_extra_radius: float,
    rejection_smoothness: float,
    # Fusion
    blend_smoothness: float,
) -> None:
    """
    Stack each group of registered images and merge the two alignments into one image.

    Every group is stacked twice from the same exposures: once from the sun-registered
    images, rejecting the moon, and once from the moon-registered ones. Merging the two
    restores the moon that the sun stack had to reject.
    """
    weight_fn = lambda stack, headers, region: integration.rejection.moon_rejection(
        stack, headers, rejection_extra_radius, rejection_smoothness, region)

    # Group the sun-registered images; registration writes both directories in pairs, under
    # the same filename, so each group's moon-registered counterparts follow from the names.
    filepath_to_header = {p: imageio.read_header(p) for p in imageio.list_files(sun_registered_dir, extensions=imageio.extensions.FITS)}
    grouped_filepaths = fits.group_filepaths(filepath_to_header, group_keywords)
    num_groups = len(grouped_filepaths)
    for group_idx, group_values in enumerate(grouped_filepaths.keys(), start=1):
        sun_filepaths = grouped_filepaths[group_values]
        moon_filepaths = [Path(moon_registered_dir) / p.name for p in sun_filepaths]
        for filepath in moon_filepaths:
            if not filepath.exists():
                raise FileNotFoundError(f"No moon-registered counterpart for {filepath.name}: expected {filepath}.")

        group_name = fits.format_group_name(group_values, group_keywords)
        group_suffix = f" from group {group_name}" if group_name else ""
        cprint(f"Stacking {len(sun_filepaths)} images{group_suffix} ({group_idx}/{num_groups}):", style="bold", color="cyan")

        cprint("Stacking the sun-registered images:", color="cyan")
        sun_img, sun_header, _ = integration.integrate(sun_filepaths, outlier_threshold, weight_fn)
        cprint("Sun-registered images stacked.", color="green")
        context.checkstate()

        cprint("Stacking the moon-registered images:", color="cyan")
        moon_img, moon_header, _ = integration.integrate(moon_filepaths, outlier_threshold)
        cprint("Moon-registered images stacked.", color="green")
        context.checkstate()

        # The moon lies at a fixed position in the moon-registered images, which is where it
        # ends up in the merged image. The sun-registered ones see it drift, so averaging
        # their positions would describe nothing.
        moon_geometry_header = fits.aggregate(
            [imageio.read_header(p) for p in moon_filepaths],
            {"MOON-X": np.mean, "MOON-Y": np.mean, "MOON-R": np.mean})

        cprint("Merging the moon and sun stacks...", color="cyan", flush=True)
        center = np.array([moon_geometry_header["MOON-X"], moon_geometry_header["MOON-Y"]])
        radius = cast(float, moon_geometry_header["MOON-R"])
        merged_img = integration.fusion.merge(sun_img, moon_img, center, radius, blend_smoothness)
        cprint("Moon and sun stacks merged.", color="green")
        context.emit_image(merged_img)

        output_header = fits.combine([sun_header, moon_header, moon_geometry_header])
        filename = f"stack_{group_name}.fits" if group_name else "stack.fits"
        imageio.write(os.path.join(stacks_dir, filename), merged_img, output_header)
        cprint(f"Stacked {len(sun_filepaths)} images{group_suffix} successfully ({group_idx}/{num_groups}).", color="green")
    cprint("Stacking completed successfully.", style='bold', color='green')


if __name__ == "__main__":
    import sys
    import yaml
    from umbra.common.terminal import ColorTerminalStream
    sys.stdout = ColorTerminalStream()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    main(**config["integration"])
