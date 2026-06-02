from collections.abc import Sequence
from pathlib import Path
from typing import cast

import astropy.io.fits
import numpy as np
from umbra.common import fits, coords
from umbra.common.terminal import cprint
from umbra.common.typing import CheckStateCallback


def read_shape(filepath: Path | str) -> tuple[int, int, int]:
    """Return the (H, W, C) shape of an image frame without loading its full data."""
    header = fits.read_fits_header(filepath)
    return cast(int, header["NAXIS2"]), cast(int, header["NAXIS1"]), cast(int, header["NAXIS3"])


def read_stack(
    filepaths: Sequence[Path | str],
    region: coords.Region | None = None,
    *,
    checkstate: CheckStateCallback,
) -> tuple[np.ndarray, list[astropy.io.fits.Header]]:
    N = len(filepaths)
    H, W, C = read_shape(filepaths[0])
    if region is None:
        region = coords.Region(width=W, height=H)
    else:
        W = region.width
        H = region.height
    stack = np.zeros((N, H, W, C), dtype=np.float32)
    headers = []
    cprint(f"Loading images...", end=' ', flush=True)
    for i in range(N):
        stack[i], header = fits.read_fits(filepaths[i], region, verbose=False, checkstate=checkstate)
        headers.append(header)
    print("Done.")
    return stack, headers
