import os
from pathlib import Path
from typing import cast

import numpy as np
import astropy.io.fits

from umbra.common import coords
from umbra.common.fits import Header, extract_shape


def read(filepath: Path | str, region: coords.Region | None = None) -> tuple[np.ndarray, Header]:
    """Read a FITS file into ``(data, header)`` (native dtype, HxW or HxWxC).

    A CFA mosaic exposes its pattern via the ``BAYERPAT`` header keyword.
    """
    with astropy.io.fits.open(filepath) as hdul:
        hdu = cast(astropy.io.fits.PrimaryHDU, hdul[0])
        hdu.verify('silentfix')
        header = hdu.header
        data = hdu.data
        if not isinstance(data, np.ndarray):
            raise ValueError(f"Found no image data in {filepath}.")
        # Crop while still in FITS-native CxHxW (color) / HxW (mono) layout.
        if region is None:
            img = data
        elif data.ndim == 2:
            img = data[region.top:region.bottom, region.left:region.right]
        else:
            img = data[:, region.top:region.bottom, region.left:region.right]
    # FITS stores color as CxHxW; expose it as HxWxC.
    if img.ndim == 3:
        img = np.moveaxis(img, 0, 2)
    return img, header


def read_header(filepath: Path | str) -> Header:
    with astropy.io.fits.open(filepath) as hdul:
        hdu = cast(astropy.io.fits.PrimaryHDU, hdul[0])
        hdu.verify('silentfix')
        return hdu.header


def read_shape(filepath: Path | str) -> tuple[int, ...]:
    return extract_shape(read_header(filepath))


def write(filepath: Path | str, data: np.ndarray, header: Header | None) -> None:
    """Write image data and header to a FITS file, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if data.ndim == 3:
        data = np.moveaxis(data, 2, 0)
    hdu = astropy.io.fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(filepath, overwrite=True)
