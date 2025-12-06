import gc
import os
import numpy as np
import tracemalloc

from umbra.common import fits
from umbra.common.terminal import cprint
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
    # Process each group
    grouped_filepaths = fits.get_grouped_filepaths(sun_registered_dir, group_keywords)
    for group_idx, group_values in enumerate(grouped_filepaths.keys(), start=1):
        # Extract info about the group
        filepaths = grouped_filepaths[group_values]
        filepaths = filepaths[::5]
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
        safe_memory_fraction = 0.5
        safe_available_mem = int(available_mem * safe_memory_fraction)
        cprint(f"Using {safe_memory_fraction*100:.0f}% of available memory for the stack: {safe_available_mem / 1024**2:.2f} MiB.")
        rows_ranges = integration.memory.compute_rows_ranges_for_stack(num_images, shape, safe_available_mem)
        num_chunks = len(rows_ranges)
        
        tracemalloc.start()
        # Process each chunk
        for chunk_idx, rows_range in enumerate(rows_ranges, start=1):
            cprint(f"Processing chunk {chunk_idx}/{num_chunks}", style="bold")
            stack, headers = integration.io.read_stack(filepaths, rows_range)
            integration.memory.print_memory_usage()
            stack = integration.rejection.moon_rejection(stack, headers)
            integration.memory.print_memory_usage()
            stack = integration.rejection.outlier_rejection(stack, rejection_threshold)
            integration.memory.print_memory_usage()
            cprint("Stacking...", end=" ", flush=True)
            stack.mean(axis=0, out=img[rows_range[0]:rows_range[1]])
            integration.memory.print_memory_usage()
            cprint("Done.")
            cprint("Computing rejection map...", end=" ", flush=True)
            stack.mask.max(axis=0, out=rejection_map[rows_range[0]:rows_range[1]])
            integration.memory.print_memory_usage()
            cprint("Done.")

            del stack
            gc.collect()
            integration.memory.print_memory_usage()
            return
            
        output_header = fits.extract_subheader(header, group_keywords+["MOON-X", "MOON-Y"])
        group_name = " - ".join([f"{group_keywords[i]}_{group_values[i]}" for i in range(len(group_keywords))])
        fits.save_as_fits(img, output_header, os.path.join(moon_stacks_dir, f"{group_name}.fits"))
        fits.save_as_fits(rejection_map, None, os.path.join(moon_stacks_dir, f"{group_name}_rejection.fits"))

        current, peak = tracemalloc.get_traced_memory()
        cprint(f"Current memory usage: {current / 1024**2:.2f} MiB")
        cprint(f"Peak memory usage: {peak / 1024**2:.2f} MiB")
        tracemalloc.stop()

    return img, rejection_map


if __name__ == "__main__":
    import sys
    import yaml
    from umbra.common.terminal import ColorTerminalStream
    sys.stdout = ColorTerminalStream()

    with open("config_custom.yaml") as f:
        config = yaml.safe_load(f)

    main(**config["integration"])
