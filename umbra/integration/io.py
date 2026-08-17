from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import astropy.io.fits
import numpy as np
from umbra.common import context, convert, coords, imageio
from umbra.common.terminal import cprint

# Reading a frame is part disk, part decompression, and the two want different widths: a few
# requests in flight already saturate the disk, while decompression keeps scaling because
# astropy releases the GIL for it. Four is the width that costs the least across formats --
# beyond it uncompressed frames lose more to queue contention than compressed ones gain.
MAX_READ_WORKERS = 4


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

    def read_into_stack(index: int) -> astropy.io.fits.Header:
        # Converting into the stack slice keeps the frame's float32 form from being allocated
        # and then copied here, which would otherwise be live in every worker at once.
        data, header = imageio.read(filepaths[index], region, to_float32=False, verbose=False)
        convert.to_float32(data, out=stack[index])
        return header

    with ThreadPoolExecutor(max_workers=min(N, MAX_READ_WORKERS)) as pool:
        futures = [pool.submit(read_into_stack, i) for i in range(N)]
        try:
            for future in futures:
                headers.append(future.result())
                # The runner's handlers are thread-local, so a pause or an abort can only be
                # honoured here, on the thread that called us, and not inside a worker.
                context.checkstate()
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise
    print("Done.")
    return stack, headers
