from collections.abc import Sequence
from pathlib import Path
from typing import cast

import astropy.io.fits
import numpy as np
from umbra.common import coords, imageio
from umbra.common.terminal import cprint
from umbra.common.typing import CheckStateCallback


def read_stack(
    filepaths: Sequence[Path | str],
    region: coords.Region | None = None,
    *,
    checkstate: CheckStateCallback,
) -> tuple[np.ndarray, list[astropy.io.fits.Header]]:
    N = len(filepaths)
    shape = imageio.read_shape(filepaths[0])
    if region is not None:
        shape = (region.height, region.width, *shape[2:])
    stack = np.zeros((N, *shape), dtype=np.float32)
    headers = []
    cprint(f"Loading images...", end=' ', flush=True)
    for i in range(N):
        stack[i], header = imageio.read(filepaths[i], region, verbose=False, checkstate=checkstate)
        headers.append(header)
    print("Done.")
    return stack, headers
