from pathlib import Path
from typing import cast

import numpy as np
from umbra.common import fits, coords
from umbra.common.terminal import cprint

def read_stack(filepaths: list[Path], region: coords.Region | None = None, *, checkstate):
    N = len(filepaths)
    header = fits.read_fits_header(filepaths[0])
    W, H, C = cast(int, header["NAXIS1"]), cast(int, header["NAXIS2"]), cast(int, header["NAXIS3"])
    if region is None:
        region = coords.Region(width=W, height=H)
    else:
        W = region.width
        H = region.height
    stack = np.zeros((N, H, W, C), dtype=np.float32)
    headers = []
    cprint(f"Loading images...", end=' ', flush=True)
    for i in range(N):
        stack[i], header = fits.read_fits_as_float(filepaths[i], region, verbose=False, checkstate=checkstate)
        headers.append(header)
    print("Done.")
    return stack, headers
