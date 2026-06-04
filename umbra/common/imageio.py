from pathlib import Path
from typing import Any

import exifread
import numpy as np
import rawpy
from PIL import Image
import cv2

from umbra.common import convert


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


def read_image_and_bayer_pattern(
    filepath: Path | str,
    *,
    to_float: bool = True,
    debayer: bool = True,
) -> tuple[np.ndarray, str | None]:
    """Read an image file, dispatching by extension.

    Parameters
    ----------
    filepath : Path or str
        Path to the image file.
    to_float : bool, optional
        If True, convert the image to float32 in [0, 1].
    debayer : bool, optional
        If True, debayer RAW sensor data into an RGB image. Ignored for
        non-RAW formats, which are already in RGB.

    Returns
    -------
    np.ndarray
        The image array.
    str | None
        The Bayer pattern, or None if not applicable.
    """
    filepath = Path(filepath)
    bayer_pattern = None
    if filepath.suffix.lower() in RAW_EXTENSIONS:
        img, bayer_pattern = read_raw(filepath)
        if debayer:
            img = debayer_image(img, bayer_pattern)
    else:
        img = read_pillow(filepath)
    if to_float:
        img = convert.to_float(img)
    return img, bayer_pattern


def read_image(
    filepath: Path | str,
    *,
    to_float: bool = True,
    debayer: bool = True,
    ) -> np.ndarray:
    """Convenience wrapper around read_image_and_bayer_pattern that discards the Bayer pattern."""
    img, _ = read_image_and_bayer_pattern(filepath, to_float=to_float, debayer=debayer)
    return img


def read_raw(filepath: Path | str) -> tuple[np.ndarray, str]:
    filepath = Path(filepath)
    with rawpy.imread(str(filepath)) as raw:
        img = raw.raw_image_visible.copy()
        bayer_pattern = _extract_bayer_pattern(raw)
        return img, bayer_pattern


def read_pillow(filepath: Path | str) -> np.ndarray:
    with Image.open(filepath) as image:
        img = np.asarray(image)
    return img


def read_metadata(filepath: Path | str) -> dict[str, Any]:
    """Read EXIF metadata from an image file."""
    filepath = Path(filepath)
    with filepath.open("rb") as file:
        return exifread.process_file(file, details=False, builtin_types=True)


def _extract_bayer_pattern(raw: rawpy.RawPy) -> str:
    colors = raw.raw_pattern
    if colors is None or colors.shape != (2, 2):
        raise ValueError("Unsupported RAW Bayer pattern")
    names = raw.color_desc.decode("ascii", errors="ignore")
    try:
        pattern = "".join(names[int(colors[row, col])] for row in range(2) for col in range(2))
    except (IndexError, TypeError):
        raise ValueError(f"Unsupported RAW Bayer pattern")
    if pattern not in {"RGGB", "BGGR", "GRBG", "GBRG"}:
        raise ValueError(f"Unsupported RAW Bayer pattern {pattern}")
    return pattern


def debayer_image(img: np.ndarray, bayer_pattern: str) -> np.ndarray:
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
