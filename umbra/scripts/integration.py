import gc
import os
import numpy as np

from umbra.common import fits, trackers, coords
from umbra.common.terminal import cprint
from umbra import integration

@trackers.track_info
def main(
    moon_registered_dir,
    sun_registered_dir,
    moon_stacks_dir,
    sun_stacks_dir,
    group_keywords,
    image_scale,
    outlier_threshold,
    extra_radius_pixels,
    smoothness
):
    # Process each group
    grouped_filepaths = fits.get_grouped_filepaths(sun_registered_dir, group_keywords)
    for group_idx, group_values in enumerate(grouped_filepaths.keys(), start=1):
        # Extract info about the group
        filepaths = grouped_filepaths[group_values]
        # filepaths = filepaths[::3]
        num_images = len(filepaths)
        header = fits.read_fits_header(filepaths[0])
        shape = (header["NAXIS2"], header["NAXIS1"], header["NAXIS3"]) # (H, W, C)
        cprint(f"Stacking {num_images} images from group {group_idx}:", style="bold")
        for keyword, value in zip(group_keywords, group_values):
            cprint(f"- {fits.format_keyword(keyword)}: {value}")
            
        # Prepare output arrays
        img = np.zeros(shape, dtype=np.float32)
        rejection_map = np.zeros(shape, dtype=np.float32)
        
        # Chunking based on available memory
        available_mem = integration.memory.get_available_memory()
        safe_memory_fraction = 0.8
        safe_available_mem = int(available_mem * safe_memory_fraction)
        cprint(f"Using {safe_memory_fraction*100:.0f}% of available memory for the stack: {safe_available_mem / 1000000:.2f} MB.")
        rows_ranges = integration.memory.compute_rows_ranges_for_stack(num_images, shape, safe_available_mem)
        num_chunks = len(rows_ranges)
        
        # Process each chunk
        for chunk_idx, rows_range in enumerate(rows_ranges, start=1):
            cprint(f"Processing chunk {chunk_idx}/{num_chunks}", style="bold")
            region = coords.Region(width=shape[1], height=rows_range[1]-rows_range[0], left=0, top=rows_range[0])
            stack, headers = integration.io.read_stack(filepaths, rows_range)
            # Pixel rejection
            stack = integration.rejection.moon_rejection(stack, headers, extra_radius_pixels, region)
            # stack = integration.rejection.outlier_rejection(stack, outlier_threshold)
            # Update output arrays
            img[rows_range[0]:rows_range[1]] = integration.reduce.average_ignore_nan(stack)
            rejection_map[rows_range[0]:rows_range[1]] = integration.rejection.compute_rejection_map(stack)
            # Free memory
            del stack
            gc.collect()
            
        output_header = fits.extract_subheader(header, group_keywords+["MOON-X", "MOON-Y"])
        group_name = " - ".join([f"{group_keywords[i]}_{group_values[i]}" for i in range(len(group_keywords))])
        fits.save_as_fits(img, output_header, os.path.join(sun_stacks_dir, f"{group_name}.fits"))
        fits.save_as_fits(rejection_map, None, os.path.join(sun_stacks_dir, f"{group_name}_rejection.fits"))

    return img, rejection_map


if __name__ == "__main__":
    import sys
    import yaml
    from umbra.common.terminal import ColorTerminalStream
    from matplotlib import pyplot as plt
    sys.stdout = ColorTerminalStream()

    with open("config_custom.yaml") as f:
        config = yaml.safe_load(f)

    img, rejection_map = main(**config["integration"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title("Stacked Image")
    axes[0].imshow(display.ht_lut(img.data, m=0.003518, vmin=0.001835, vmax=1))
    axes[1].set_title("Rejection Map")
    axes[1].imshow(rejection_map)
    plt.show()