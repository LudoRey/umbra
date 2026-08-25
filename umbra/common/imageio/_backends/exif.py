import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

import exifread

from umbra.common.fits import Header


# Identifies the CR3 uuid box that holds the EXIF IFDs, as written by Canon.
_CR3_METADATA_UUID = bytes.fromhex("85c0b687820f11e08111f4ce462b6a48")


def read_exif(filepath: Path | str) -> dict[str, Any]:
    """Read EXIF metadata from an image file."""
    filepath = Path(filepath)
    with filepath.open("rb") as file:
        if file.read(12)[4:12] == b"ftypcrx ":
            return _read_cr3_exif(file)
        file.seek(0)
        return exifread.process_file(file, details=False, builtin_types=True)


def _read_cr3_exif(file: BinaryIO) -> dict[str, Any]:
    """Read EXIF metadata from an open CR3 file.

    CR3 is an ISO base media container rather than a TIFF, so exifread cannot locate
    the metadata on its own. Canon stores the IFDs verbatim as TIFF blocks in boxes of
    a uuid box under ``moov``: ``CMT1`` holds IFD0 and ``CMT2`` the Exif IFD.
    """
    metadata: dict[str, Any] = {}
    for box_type, payload in _iter_boxes(_read_cr3_metadata_box(file)):
        if box_type == b"CMT1":
            metadata.update(_read_tiff_block(payload))
        elif box_type == b"CMT2":
            # Read on its own, the Exif IFD yields tags named as if they were IFD0 ones.
            metadata.update({key.replace("Image ", "EXIF ", 1): value
                             for key, value in _read_tiff_block(payload).items()})
    return metadata


def _read_tiff_block(data: bytes) -> dict[str, Any]:
    return exifread.process_file(io.BytesIO(data), details=False, builtin_types=True)


def _read_cr3_metadata_box(file: BinaryIO) -> bytes:
    """Return the payload of the CR3 uuid box holding the EXIF IFDs."""
    for box_type, payload in _iter_boxes(_read_moov_box(file)):
        if box_type == b"uuid" and payload[:16] == _CR3_METADATA_UUID:
            return payload[16:]
    raise ValueError("CR3 file has no metadata uuid box")


def _read_moov_box(file: BinaryIO) -> bytes:
    """Return the payload of the top-level ``moov`` box."""
    file.seek(0)
    while True:
        header = file.read(8)
        if len(header) < 8:
            raise ValueError("CR3 file has no moov box")
        size = int.from_bytes(header[:4], "big")
        header_size = 8
        if size == 1:  # the true size is a 64-bit integer following the box type
            size = int.from_bytes(file.read(8), "big")
            header_size = 16
        if size < header_size:
            raise ValueError("CR3 file has a malformed box")
        if header[4:8] == b"moov":
            return file.read(size - header_size)
        file.seek(size - header_size, 1)


def _iter_boxes(data: bytes) -> Iterator[tuple[bytes, bytes]]:
    """Yield ``(type, payload)`` for each ISO base media box in ``data``."""
    offset = 0
    while offset + 8 <= len(data):
        size = int.from_bytes(data[offset:offset+4], "big")
        if size < 8 or offset + size > len(data):
            raise ValueError("CR3 file has a malformed box")
        yield data[offset+4:offset+8], data[offset+8:offset+size]
        offset += size


def build_header_from_exif(metadata: dict[str, Any]) -> Header:
    """Map an EXIF metadata dict onto a FITS header."""
    header = Header()
    _set_if_present(header, "DATE-OBS", _date_obs(metadata), "Observation date and time")
    _set_if_present(header, "EXPTIME", metadata.get("EXIF ExposureTime"), "Exposure time in seconds")
    _set_if_present(header, "ISOSPEED", metadata.get("EXIF ISOSpeedRatings") or metadata.get("EXIF PhotographicSensitivity"), "ISO speed")
    _set_if_present(header, "FOCALLEN", metadata.get("EXIF FocalLength"), "Focal length in mm")
    _set_if_present(header, "FNUMBER", metadata.get("EXIF FNumber"), "Lens F-number")
    _set_if_present(header, "INSTRUME", metadata.get("Image Model"), "Camera model")
    header["ORIGIN"] = ("Umbra", "Software that created this file")
    return header


def _set_if_present(header: Header, key: str, value: Any, comment: str) -> None:
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
