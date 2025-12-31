import numpy as np
import psutil

from umbra.common.terminal import cprint

def compute_stacking_memory_requirements(num_images, height, width, num_channels, byte_per_pixel=4):
    '''Peak memory usage during stacking. It is attained during outlier rejection.'''
    stack_memory = height * width * num_channels * num_images * byte_per_pixel
    weights_memory = height * width * num_images * byte_per_pixel
    peak_rejection_memory = height * width * num_channels * (num_images * 2 + byte_per_pixel * 2) # see outlier_rejection notes
    return stack_memory + weights_memory + peak_rejection_memory

def compute_rows_ranges_for_stack(num_images, shape, max_mem, dtype=np.float32):
    height, width, num_channels = shape
    required_stack_mem = compute_stacking_memory_requirements(num_images, height, width, num_channels, np.dtype(dtype).itemsize)
    n_chunks = int(np.ceil(required_stack_mem / max_mem))
    chunk_mem = required_stack_mem / n_chunks

    if n_chunks > 1:
        cprint(f"The stack is divided into {n_chunks} chunks. Memory usage: {chunk_mem / 1000000:.2f} MB.")
    else:
        cprint(f"The entire stack is processed in a single chunk. Memory usage: {chunk_mem / 1000000:.2f} MB.")

    # Divide into chunks as evenly as possible, with the remainder distributed among the first chunks
    chunk_sizes = [(height // n_chunks) + (1 if i < height % n_chunks else 0) for i in range(n_chunks)]

    rows_ranges = []
    row_start = 0
    for size in chunk_sizes:
        row_end = row_start + size
        rows_ranges.append((row_start, row_end))
        row_start = row_end
    return rows_ranges

def get_available_memory():
    mem = psutil.virtual_memory()
    return mem.available