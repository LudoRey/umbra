import numpy as np
from umbra.common import fits


def read_stack(filepaths, rows_range=None):
    N = len(filepaths)
    header = fits.read_fits_header(filepaths[0])
    if rows_range is None: 
        H = header["NAXIS2"]
    else:
        H = min(rows_range[1], header["NAXIS2"]) - max(rows_range[0], 0)
    W, C = header["NAXIS1"], header["NAXIS3"]
    stack = np.zeros((N, H, W, C))
    for i in range(N):
        stack[i], _ = fits.read_fits_as_float(filepaths[i], rows_range)
    return stack
