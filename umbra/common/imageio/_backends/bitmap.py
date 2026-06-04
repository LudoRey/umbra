from pathlib import Path

import numpy as np
from PIL import Image

from umbra.common import coords
from umbra.common.fits import Header
from umbra.common.imageio._backends import exif


def read(filepath: Path | str, region: coords.Region | None = None) -> tuple[np.ndarray, Header]:
    """Read a developed image (JPEG/PNG/TIFF/...) as ``(data, header)``, HxW or HxWxC."""
    filepath = Path(filepath)
    with Image.open(filepath) as image:
        img = np.asarray(image)
    if region is not None:
        img = img[region.top:region.top+region.height, region.left:region.left+region.width]
    return img, read_header(filepath)


def read_header(filepath: Path | str) -> Header:
    return exif.build_header_from_exif(exif.read_exif(filepath))


def read_shape(filepath: Path | str) -> tuple[int, ...]:
    with Image.open(filepath) as image:
        width, height = image.size
        num_channels = len(image.getbands())
        if num_channels > 1:
            return height, width, num_channels
        return height, width
