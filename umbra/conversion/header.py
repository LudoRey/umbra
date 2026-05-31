from typing import Any

import astropy.io.fits


def build_header_from_exif(metadata: dict[str, Any]) -> astropy.io.fits.Header:
    header = astropy.io.fits.Header()
    _set_if_present(header, "DATE-OBS", _date_obs(metadata), "Observation date and time")
    _set_if_present(header, "EXPTIME", metadata.get("EXIF ExposureTime"), "Exposure time in seconds")
    _set_if_present(header, "ISOSPEED", metadata.get("EXIF ISOSpeedRatings") or metadata.get("EXIF PhotographicSensitivity"), "ISO speed")
    _set_if_present(header, "FOCALLEN", metadata.get("EXIF FocalLength"), "Focal length in mm")
    _set_if_present(header, "FNUMBER", metadata.get("EXIF FNumber"), "Lens F-number")
    _set_if_present(header, "INSTRUME", _camera(metadata), "Camera model")
    header["ORIGIN"] = ("Umbra", "Software that created this file")
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
