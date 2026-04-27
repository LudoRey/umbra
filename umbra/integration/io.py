import numpy as np
from umbra.common import fits
from umbra.common.terminal import cprint

def read_stack(filepaths, rows_range=None, *, checkstate):
    N = len(filepaths)
    header = fits.read_fits_header(filepaths[0])
    if rows_range is None: 
        rows_range = (0, header["NAXIS2"])
    W, C = header["NAXIS1"], header["NAXIS3"]
    H = rows_range[1] - rows_range[0]
    stack = np.zeros((N, H, W, C), dtype=np.float32)
    headers = []
    cprint(f"Loading images...", end=' ', flush=True)
    for i in range(N):
        stack[i], header = fits.read_fits_as_float(filepaths[i], rows_range, verbose=False, checkstate=checkstate)
        headers.append(header)
    print("Done.")
    return stack, headers
