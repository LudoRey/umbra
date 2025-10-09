import numpy as np

from umbra.common import fits
from umbra.common.utils import cprint
from umbra import integration

def main(
    moon_registered_dir,
    sun_registered_dir,
    moon_stacks_dir,
    sun_stacks_dir,
    group_keywords,
    image_scale,
    rejection_threshold,
    extra_radius_pixels,
    smoothness
):
    # Determine available memory
    available_mem = integration.memory.get_available_memory()
    safe_memory_fraction = 0.8
    safe_available_mem = int(available_mem * safe_memory_fraction)
    cprint(f"Using {safe_memory_fraction*100:.0f}% of available memory for stacking: {safe_available_mem / 1024**2:.2f} MiB.")
    
    # Process moon-registered images
    grouped_filepaths = fits.get_grouped_filepaths(sun_registered_dir, group_keywords)
    for group_idx, group_values in enumerate(grouped_filepaths.keys(), start=1):
        # Extract info about the group
        filepaths = grouped_filepaths[group_values]
        num_images = len(filepaths)
        header = fits.read_fits_header(filepaths[0])
        shape = (header["NAXIS2"], header["NAXIS1"], header["NAXIS3"]) # (H, W, C)
        cprint(f"Stacking {num_images} images from group {group_idx}:", style="bold")
        for keyword, value in zip(group_keywords, group_values):
            cprint(f"- {fits.format_keyword(keyword)}: {value}")

        # Chunking based on available memory
        rows_ranges = integration.memory.compute_rows_ranges_for_stack(num_images, shape, available_mem)
        num_chunks = len(rows_ranges)
        
        # Prepare output arrays
        img = np.zeros(shape, dtype=np.float64)
        rejection_map = np.zeros(shape, dtype=np.float32)
        
        # Process each chunk
        for chunk_idx, rows_range in enumerate(rows_ranges, start=1):
            cprint(f"Processing chunk {chunk_idx}/{num_chunks}", style="bold")
            stack = integration.io.read_stack(filepaths, rows_range)
            masked_stack = integration.rejection.outlier_mad_rejection(stack, rejection_threshold)

            rejection_map[rows_range[0]:rows_range[1]] = masked_stack.mask.astype(np.float32).max(axis=0)
            img[rows_range[0]:rows_range[1]] = masked_stack.mean(axis=0) # Since stack is a masked array, the mean ignores masked values

    return img, rejection_map


if __name__ == "__main__":
    import sys
    import yaml
    from umbra.common.utils import ColorTerminalStream
    sys.stdout = ColorTerminalStream()

    with open("config_custom.yaml") as f:
        config = yaml.safe_load(f)

    main(**config["integration"])
