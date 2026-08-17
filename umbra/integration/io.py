from collections.abc import Sequence
from pathlib import Path

import astropy.io.fits
import numpy as np
from umbra.common import convert, coords, imageio
from umbra.common.terminal import cprint


def read_stack(
    filepaths: Sequence[Path | str],
    region: coords.Region | None = None,
) -> tuple[np.ndarray, list[astropy.io.fits.Header]]:
    N = len(filepaths)
    shape = imageio.read_shape(filepaths[0])
    if region is not None:
        shape = (region.height, region.width, *shape[2:])
    stack = np.zeros((N, *shape), dtype=np.float32)
    headers = []
    cprint(f"Loading images...", end=' ', flush=True)
    for i in range(N):
        # Converting into the stack slice keeps the frame's float32 form from being allocated
        # and then copied here, which matters once several frames are read at once.
        data, header = imageio.read(filepaths[i], region, to_float32=False, verbose=False)
        convert.to_float32(data, out=stack[i])
        headers.append(header)
    print("Done.")
    return stack, headers
