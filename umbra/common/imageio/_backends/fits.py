import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import astropy.io.fits

from umbra.common import convert, coords
from umbra.common.fits import Header, extract_shape

Format = Literal["float32", "uint16", "uint16-compressed"]


@dataclass(frozen=True)
class _Encoding:
    """How the pixels of a given format reach the file."""
    convert: Callable[[np.ndarray], np.ndarray]
    compression: str | None


# Compressed float32 has no entry: astropy quantizes floating point data to integers before
# compressing it, so the precision would be lost either way, silently.
ENCODINGS: dict[Format, _Encoding] = {
    "float32": _Encoding(convert.to_float32, None),
    "uint16": _Encoding(convert.to_uint16, None),
    "uint16-compressed": _Encoding(convert.to_uint16, "RICE_1"),
}

TILE_ROWS = 64


def _image_hdu(
    hdul: astropy.io.fits.HDUList, filepath: Path | str,
) -> astropy.io.fits.PrimaryHDU | astropy.io.fits.ImageHDU:
    """The HDU holding the image.

    A compressed image cannot live in the primary HDU, so it sits in the first extension
    behind an empty one.
    """
    hdu = hdul[1] if hdul[0].header["NAXIS"] == 0 and len(hdul) > 1 else hdul[0]
    hdu.verify('silentfix')
    if hdu.header["NAXIS"] == 0:
        raise ValueError(f"Found no image data in {filepath}.")
    return hdu


def read(filepath: Path | str, region: coords.Region | None = None) -> tuple[np.ndarray, Header]:
    """Read a FITS file into ``(data, header)`` (native dtype, HxW or HxWxC).

    A CFA mosaic exposes its pattern via the ``BAYERPAT`` header keyword.
    """
    with astropy.io.fits.open(filepath) as hdul:
        hdu = _image_hdu(hdul, filepath)
        header = hdu.header
        # Crop while still in FITS-native CxHxW (color) / HxW (mono) layout. Indexing
        # `section` reads only the requested rows, where indexing `data` would first
        # materialize the whole image whenever astropy has to transform it on the way in
        # -- decompressing every tile, or applying the uint16 BZERO offset.
        if region is None:
            img = hdu.section[:]
        elif len(extract_shape(header)) == 2:
            img = hdu.section[region.top:region.bottom, region.left:region.right]
        else:
            img = hdu.section[:, region.top:region.bottom, region.left:region.right]
    # FITS stores color as CxHxW; expose it as HxWxC.
    if img.ndim == 3:
        img = np.moveaxis(img, 0, 2)
    return img, header


def read_header(filepath: Path | str) -> Header:
    with astropy.io.fits.open(filepath) as hdul:
        return _image_hdu(hdul, filepath).header


def read_shape(filepath: Path | str) -> tuple[int, ...]:
    return extract_shape(read_header(filepath))


def write(
    filepath: Path | str,
    data: np.ndarray,
    header: Header | None,
    *,
    format: Format = "float32",
) -> None:
    """Write image data and header to a FITS file, creating parent dirs as needed.

    ``format`` selects the on-disk representation: ``"float32"`` stores the values as they
    are, ``"uint16"`` halves the file and quantizes to steps of 1/65535, and
    ``"uint16-compressed"`` adds lossless RICE compression on top of that.
    """
    if format not in ENCODINGS:
        raise ValueError(f"Unsupported format {format!r}, expected one of {', '.join(ENCODINGS)}.")
    encoding = ENCODINGS[format]
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = encoding.convert(data)
    if data.ndim == 3:
        # astropy byteswaps into the FITS big-endian order element by element when handed a
        # strided view, which costs far more than materializing the transpose up front.
        data = np.ascontiguousarray(np.moveaxis(data, 2, 0))
    if encoding.compression is None:
        hdu = astropy.io.fits.PrimaryHDU(data=data, header=header)
    else:
        # A tile is decompressed whole, so tall tiles make a reader working in row bands pay
        # for rows it did not ask for. The height does not affect the compression ratio.
        tile_shape = (1, TILE_ROWS, data.shape[-1]) if data.ndim == 3 else (TILE_ROWS, data.shape[-1])
        hdu = astropy.io.fits.CompImageHDU(data=data, header=header,
                                           compression_type=encoding.compression,
                                           tile_shape=tile_shape)
    hdu.writeto(filepath, overwrite=True)
