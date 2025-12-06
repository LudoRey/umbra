import numpy as np
from umbra.common import fits
from umbra.common.terminal import cprint


def read_stack(filepaths, rows_range=None):
    N = len(filepaths)
    header = fits.read_fits_header(filepaths[0])
    if rows_range is None: 
        rows_range = (0, header["NAXIS2"])
    cprint(f"Reading rows {rows_range[0]} -> {rows_range[1]}:")
    W, C = header["NAXIS1"], header["NAXIS3"]
    H = rows_range[1] - rows_range[0]
    stack = np.zeros((N, H, W, C), dtype=np.float32)
    headers = []
    for i in range(N):
        cprint(f"Loading image {i+1}/{N}...", end='\r', flush=True)
        stack[i], header = fits.read_fits_as_float(filepaths[i], rows_range, verbose=False)
        headers.append(header)
    print()
    return stack, headers
