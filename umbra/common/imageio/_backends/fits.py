import os
from pathlib import Path

import numpy as np
import astropy.io.fits

from umbra.common import coords
from umbra.common.fits import Header, extract_shape


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


def write(filepath: Path | str, data: np.ndarray, header: Header | None) -> None:
    """Write image data and header to a FITS file, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if data.ndim == 3:
        # astropy byteswaps into the FITS big-endian order element by element when handed a
        # strided view, which costs far more than materializing the transpose up front.
        data = np.ascontiguousarray(np.moveaxis(data, 2, 0))
    hdu = astropy.io.fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(filepath, overwrite=True)
