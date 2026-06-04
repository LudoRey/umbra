from pathlib import Path

import numpy as np
import rawpy

from umbra.common import coords, bayer
from umbra.common.fits import Header
from umbra.common.imageio._backends import exif


def read(filepath: Path | str, region: coords.Region | None = None) -> tuple[np.ndarray, Header]:
    """Read a camera RAW file as ``(mosaic, header)``, never debayered.

    The mosaic is a native 2D Bayer pattern; the pattern is recorded in the header
    under ``BAYERPAT``.
    """
    filepath = Path(filepath)
    with rawpy.imread(str(filepath)) as raw:
        img = raw.raw_image_visible.copy()
        bayer_pattern = _extract_bayer_pattern(raw)
    if region is not None:
        img = img[region.top:region.top+region.height, region.left:region.left+region.width]
    header = read_header(filepath)
    header["BAYERPAT"] = (bayer_pattern, "Bayer color filter array pattern")
    return img, header


def read_header(filepath: Path | str) -> Header:
    return exif.build_header_from_exif(exif.read_exif(filepath))


def read_shape(filepath: Path | str) -> tuple[int, ...]:
    """Return the (H, W) shape of the raw mosaic (always 2D)."""
    with rawpy.imread(str(filepath)) as raw:
        return raw.sizes.height, raw.sizes.width


def _extract_bayer_pattern(raw: rawpy.RawPy) -> str:
    colors = raw.raw_pattern
    if colors is None or colors.shape != (2, 2):
        raise ValueError("Unsupported RAW Bayer pattern")
    names = raw.color_desc.decode("ascii", errors="ignore")
    try:
        pattern = "".join(names[int(colors[row, col])] for row in range(2) for col in range(2))
    except (IndexError, TypeError):
        raise ValueError("Unsupported RAW Bayer pattern")
    if pattern not in bayer.PATTERNS:
        raise ValueError(f"Unsupported RAW Bayer pattern {pattern}")
    return pattern
