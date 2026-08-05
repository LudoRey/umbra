import os
from typing import cast
import numpy as np

from umbra.common import coords, disk, fits, imageio
from umbra.common.terminal import cprint


def main(
    sun_stacks_dir,
    moon_stacks_dir,
    merged_stacks_dir,
    smoothness,
):
    os.makedirs(merged_stacks_dir, exist_ok=True)

    # The stacks directories also hold "rejection_map_<group-idx>.fits"; skip those.
    sun_paths = [
        p for p in imageio.list_files(sun_stacks_dir, extensions=imageio.extensions.FITS)
        if not p.stem.startswith("rejection")
    ]

    for sun_path in sun_paths:
        moon_path = os.path.join(moon_stacks_dir, sun_path.name)
        if not os.path.exists(moon_path):
            cprint(f"No matching moon stack for {sun_path.name}, skipping.", color="yellow")
            continue

        cprint(f"Fusing {sun_path.name}...", style="bold", color="cyan")
        sun_img, sun_header = imageio.read(sun_path)
        moon_img, moon_header = imageio.read(moon_path)

        center = np.array([moon_header["MOON-X"], moon_header["MOON-Y"]])
        radius = cast(float, moon_header["MOON-R"])
        region = coords.Region.from_shape(sun_img.shape[:2])

        # Three regimes across the limb: the moon image inside the disk, the sun image outside it,
        # and the darker of the two over a transition annulus straddling the limb, where the moon
        # image may hold corona leaked over the limb and the sun image the pixels the moon
        # rejection dropped. The annulus spans radius +/- smoothness, so the minimum takes over
        # completely at the limb itself. The weights sum to 1 everywhere, so each handover is
        # continuous even where the two stacks disagree; a hard switch would ring wherever min
        # picks the other side.
        moon_weights = disk.smooth_disk(center, radius - smoothness, region, smoothness)
        moon_or_min_weights = disk.smooth_disk(center, radius, region, smoothness)
        min_weights = moon_or_min_weights - moon_weights
        merged_img = (moon_weights[:, :, None] * moon_img
                      + min_weights[:, :, None] * np.minimum(moon_img, sun_img)
                      + (1 - moon_or_min_weights)[:, :, None] * sun_img)

        merged_header = fits.combine([sun_header, moon_header])
        imageio.write(os.path.join(merged_stacks_dir, sun_path.name), merged_img, merged_header)
        # Persist the moon weights as a diagnostic layer (view it alongside sun/moon/merged).
        imageio.write(os.path.join(merged_stacks_dir, f"{sun_path.stem}_mask.fits"), min_weights, None)


if __name__ == "__main__":
    import sys
    import yaml
    from umbra.common.terminal import ColorTerminalStream
    sys.stdout = ColorTerminalStream()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    main(**config["fusion"])
