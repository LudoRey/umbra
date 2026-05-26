from pathlib import Path
from typing import Any

import astropy.io.fits
import cv2
import exifread
import rawpy
import numpy as np
from PIL import Image

from umbra.common import fits


RAW_EXTENSIONS = {
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
}
PIL_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp"
}
SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | PIL_EXTENSIONS


def list_image_filepaths(dirpath: Path | str) -> list[Path]:
    """Return sorted supported image file paths in a directory."""
    dirpath = Path(dirpath)
    return sorted(
        p
        for p in dirpath.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def convert_file(
    input_filepath: Path | str,
    output_filepath: Path | str,
) -> tuple[np.ndarray, astropy.io.fits.Header]:
    """Convert one image file to FITS."""
    input_filepath = Path(input_filepath)
    output_filepath = Path(output_filepath)

    metadata = _extract_metadata(input_filepath)
    if input_filepath.suffix.lower() in RAW_EXTENSIONS:
        img, bayer_pattern = _read_raw(input_filepath)
        if bayer_pattern is not None:
            metadata["BAYERPAT"] = bayer_pattern
    else:
        img = _read_pillow(input_filepath)

    img = _validate_or_convert_image(img)

    header = _build_header(input_filepath, metadata)
    fits.save_as_fits(img, header, output_filepath)
    return img, header


def _read_raw(filepath: Path) -> tuple[np.ndarray, str | None]:
    with rawpy.imread(str(filepath)) as raw:
        img = raw.raw_image_visible.copy()
        bayer_pattern = _raw_bayer_pattern(raw)
        if bayer_pattern is None:
            raise ValueError(f"Unsupported RAW Bayer pattern in {filepath.name}.")
        return _debayer_raw(img, bayer_pattern), bayer_pattern


def _read_pillow(filepath: Path) -> np.ndarray:
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


def _validate_or_convert_image(img: np.ndarray) -> np.ndarray:
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
    

def _extract_metadata(filepath: Path) -> dict[str, Any]:
    with filepath.open("rb") as file:
        return exifread.process_file(file, details=False, builtin_types=True)


def _build_header(filepath: Path, metadata: dict[str, Any]) -> astropy.io.fits.Header:
    header = astropy.io.fits.Header()
    _set_if_present(header, "DATE-OBS", _date_obs(metadata), "Observation date and time")
    _set_if_present(header, "EXPTIME", metadata.get("EXIF ExposureTime"), "Exposure time in seconds")
    _set_if_present(header, "ISOSPEED", metadata.get("EXIF ISOSpeedRatings") or metadata.get("EXIF PhotographicSensitivity"), "ISO speed")
    _set_if_present(header, "FOCALLEN", metadata.get("EXIF FocalLength"), "Focal length in mm")
    _set_if_present(header, "FNUMBER", metadata.get("EXIF FNumber"), "Lens F-number")
    _set_if_present(header, "INSTRUME", _camera(metadata), "Camera model")
    _set_if_present(header, "BAYERPAT", metadata.get("BAYERPAT"), "Source Bayer pattern")
    header["ORIGIN"] = ("Umbra", "Software that created this file")
    header["SRCFMT"] = (filepath.suffix.lower().lstrip(".")[:68], "Source image format")
    return header


def _set_if_present(header: astropy.io.fits.Header, key: str, value: Any, comment: str) -> None:
    if value is not None:
        header[key] = (value, comment)


def _date_obs(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("EXIF DateTimeOriginal") or metadata.get("EXIF DateTimeDigitized") or metadata.get("Image DateTime")
    if value is None:
        return None
    text = str(value).strip()
    if len(text) >= 19 and text[10] == " ":
        return f"{text[0:10]}T{text[11:19]}"
    return text


def _camera(metadata: dict[str, Any]) -> str | None:
    make = metadata.get("Image Make")
    model = metadata.get("Image Model")
    if make and model:
        return f"{make} {model}"[:68]
    if model:
        return str(model)[:68]
    return None
