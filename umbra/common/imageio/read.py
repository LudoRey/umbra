from pathlib import Path

import cv2
import numpy as np
import rawpy
from PIL import Image


RAW_EXTENSIONS = frozenset({
    ".3fr",
    ".ari",
    ".arw",
    ".bay",
    ".braw",
    ".cr2",
    ".cr3",
    ".crw",
    ".dcr",
    ".dng",
    ".erf",
    ".fff",
    ".iiq",
    ".k25",
    ".kdc",
    ".mef",
    ".mos",
    ".mrw",
    ".nef",
    ".nrw",
    ".orf",
    ".pef",
    ".raf",
    ".raw",
    ".rw2",
    ".rwl",
    ".sr2",
    ".srf",
    ".srw",
    ".x3f",
})
PIL_EXTENSIONS = frozenset({
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp"
})
SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | PIL_EXTENSIONS


def list_image_filepaths(dirpath: Path | str) -> list[Path]:
    """Return sorted supported image file paths in a directory."""
    dirpath = Path(dirpath)
    return sorted(
        p
        for p in dirpath.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def read_image(filepath: Path | str) -> tuple[np.ndarray, str | None]:
    """Read an image file, dispatching by extension.

    Returns (image_array, bayer_pattern). bayer_pattern is None for non-RAW formats.
    """
    filepath = Path(filepath)
    if filepath.suffix.lower() in RAW_EXTENSIONS:
        return read_raw(filepath)
    return read_pillow(filepath), None


def read_raw(filepath: Path | str) -> tuple[np.ndarray, str | None]:
    filepath = Path(filepath)
    with rawpy.imread(str(filepath)) as raw:
        img = raw.raw_image_visible.copy()
        bayer_pattern = _raw_bayer_pattern(raw)
        if bayer_pattern is None:
            raise ValueError(f"Unsupported RAW Bayer pattern in {filepath.name}.")
        return _debayer_raw(img, bayer_pattern), bayer_pattern


def read_pillow(filepath: Path | str) -> np.ndarray:
    with Image.open(filepath) as image:
        image = image.convert("RGB")
        img = np.asarray(image)
    return img


def _raw_bayer_pattern(raw: rawpy.RawPy) -> str | None:
    colors = raw.raw_pattern
    if colors is None or colors.shape != (2, 2):
        return None
    names = raw.color_desc.decode("ascii", errors="ignore")
    try:
        pattern = "".join(names[int(colors[row, col])] for row in range(2) for col in range(2))
    except (IndexError, TypeError):
        return None
    return pattern if pattern in {"RGGB", "BGGR", "GRBG", "GBRG"} else None


def _debayer_raw(img: np.ndarray, bayer_pattern: str) -> np.ndarray:
    if img.ndim != 2:
        raise ValueError("RAW sensor data must be single-channel before debayering.")
    code_by_pattern = {
        "RGGB": cv2.COLOR_BayerRGGB2RGB,
        "BGGR": cv2.COLOR_BayerBGGR2RGB,
        "GRBG": cv2.COLOR_BayerGRBG2RGB,
        "GBRG": cv2.COLOR_BayerGBRG2RGB,
    }
    try:
        code = code_by_pattern[bayer_pattern]
    except KeyError as exc:
        raise ValueError(f"Unsupported Bayer pattern {bayer_pattern}.") from exc
    return cv2.cvtColor(img, code)


def validate_or_convert_image(img: np.ndarray) -> np.ndarray:
    if np.issubdtype(img.dtype, np.floating):
        if img.min() < 0 or img.max() > 1:
            raise ValueError("Floating point image values must be in the range [0, 1].")
        return img
    elif np.issubdtype(img.dtype, np.unsignedinteger):
        itemsize = img.dtype.itemsize
        if itemsize == 2: # already uint16
            return img
        elif itemsize == 1: # uint8
            return img.astype(np.uint16) * (2**8 + 1)
        else:
            return img >> (itemsize * 8 - 16) # downscale to 16 bits by bit-shifting
    else:
        raise ValueError(f"Unsupported image dtype {img.dtype}.")
