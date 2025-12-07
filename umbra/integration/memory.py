import numpy as np
import psutil

from umbra.common.terminal import cprint

def compute_stack_memory_requirements(num_images, shape, dtype=np.float32):
    '''See docstring notes of umbra.integration.rejection.outlier_rejection for details.'''
    stack_memory = np.prod(shape) * num_images * (np.dtype(dtype).itemsize)
    rejection_memory = np.prod(shape) * (num_images * 2 + (np.dtype(dtype).itemsize) * 2)
    return stack_memory + rejection_memory

def compute_rows_ranges_for_stack(num_images, shape, max_mem, dtype=np.float32):
    H = shape[0]
    required_stack_mem = compute_stack_memory_requirements(num_images, shape, dtype)
    n_chunks = int(np.ceil(required_stack_mem / max_mem))
    chunk_mem = required_stack_mem / n_chunks

    if n_chunks > 1:
        cprint(f"The stack is divided into {n_chunks} chunks to fit into memory. Each chunk uses approximately {chunk_mem / 1000000:.2f} MB of memory.")
    else:
        cprint(f"The entire stack fits into memory. Approximate memory usage: {chunk_mem / 1000000:.2f} MB.")

    # Divide into chunks as evenly as possible, with the remainder distributed among the first chunks
    chunk_sizes = [(H // n_chunks) + (1 if i < H % n_chunks else 0) for i in range(n_chunks)]

    rows_ranges = []
    start = 0
    for size in chunk_sizes:
        end = start + size
        rows_ranges.append((start, end))
        start = end
    return rows_ranges

def get_available_memory():
    mem = psutil.virtual_memory()
    return mem.available