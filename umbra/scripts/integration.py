import os
from collections.abc import Sequence

from umbra.common import fits, imageio
from umbra.common.terminal import cprint
from umbra import integration
from umbra.common.typing import CheckStateCallback, ImageCallback


# @trackers.track_info
def main(
    # IO
    registered_dir: str,
    stacks_dir: str,
    group_keywords: Sequence[str],
    # Outlier rejection
    outlier_threshold: float,
    # Moon rejection (optional, for sun-registered images)
    moon_rejection: bool = False,
    extra_radius_pixels: float = 0,
    smoothness: float = 0,
    # GUI interactions
    *,
    img_callback: ImageCallback = lambda _img: None,
    checkstate: CheckStateCallback = lambda: None,
) -> None:
    # Moon rejection produces per-pixel weights; otherwise a uniform mean is used.
    weight_fn = None
    if moon_rejection:
        weight_fn = lambda stack, headers, region: integration.rejection.moon_rejection(
            stack, headers, extra_radius_pixels, smoothness, region)

    # Process each group
    filepath_to_header = {p: imageio.read_header(p) for p in imageio.list_files(registered_dir, extensions=imageio.extensions.FITS)}
    grouped_filepaths = fits.group_filepaths(filepath_to_header, group_keywords)
    num_groups = len(grouped_filepaths)
    for group_idx, group_values in enumerate(grouped_filepaths.keys(), start=1):
        filepaths = grouped_filepaths[group_values]
        group_identifier = ', '.join([f'{k}={v}' for k, v in zip(group_keywords, group_values)])
        cprint(f"Stacking {len(filepaths)} images from group {group_identifier} ({group_idx}/{num_groups}):", style="bold", color="cyan")

        img, output_header, total_weights = integration.integrate(
            filepaths, outlier_threshold, weight_fn, img_callback=img_callback, checkstate=checkstate)

        group_name = " - ".join([f"{group_keywords[i]}_{group_values[i]}" for i in range(len(group_keywords))])
        imageio.write(os.path.join(stacks_dir, f"{group_name}.fits"), img, output_header, checkstate=checkstate)
        imageio.write(os.path.join(stacks_dir, f"{group_name}_rejection.fits"), total_weights, None, checkstate=checkstate)
        cprint(f"Group {group_identifier} stacked successfully ({group_idx}/{num_groups}).", color="green")
    cprint(f"Stacking completed successfully.", style='bold', color='green')

if __name__ == "__main__":
    import sys
    import yaml
    from umbra.common.terminal import ColorTerminalStream
    sys.stdout = ColorTerminalStream()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    main(**config["sun_integration"])
    main(**config["moon_integration"])
