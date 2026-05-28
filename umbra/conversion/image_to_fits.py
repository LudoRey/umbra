from pathlib import Path
from typing import Any

import astropy.io.fits
import numpy as np

from umbra.common import fits, imageio


def convert_file(
    input_filepath: Path | str,
    output_filepath: Path | str,
) -> tuple[np.ndarray, astropy.io.fits.Header]:
    """Convert one image file to FITS."""
    input_filepath = Path(input_filepath)
    output_filepath = Path(output_filepath)

    metadata = imageio.extract_metadata(input_filepath)
    img, bayer_pattern = imageio.read_image(input_filepath)
    if bayer_pattern is not None:
        metadata["BAYERPAT"] = bayer_pattern
        img = imageio.debayer(img, bayer_pattern)

    img = imageio.validate_or_convert_dtype(img)

    header = _build_header(input_filepath, metadata)
    fits.save_as_fits(img, header, output_filepath)
    return img, header


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
