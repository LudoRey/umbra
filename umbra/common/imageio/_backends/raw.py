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
        img = _scale_to_uint16(raw.raw_image_visible, raw.white_level)
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


def _scale_to_uint16(img: np.ndarray, white_level: int) -> np.ndarray:
    """Stretch a raw mosaic to fill the full uint16 range.

    Camera RAWs store low-depth samples (e.g. 14-bit) in a uint16 container, so the data
    would otherwise read too dark. The black level is preserved.

    ``white_level`` is the value at which a pixel is considered saturated, as reported by
    the RAW decoder. The sample depth is taken as the number of bits needed to represent it
    (e.g. a value near 16383 implies 14-bit data). Frames whose depth cannot be inferred
    from ``white_level`` are returned unchanged.
    """
    bits = int(white_level).bit_length()
    shift = 16 - bits
    if not 0 < shift < 16:
        # shift == 0: already 16-bit; shift < 0: white_level > 16-bit; shift == 16: white_level
        # is 0 (LibRaw could not determine saturation). In all cases, leave the data unscaled.
        return img.copy()
    return (img << shift) | (img >> (bits - shift))


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
