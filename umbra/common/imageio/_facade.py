from pathlib import Path
from types import ModuleType

import numpy as np

from umbra.common import bayer, context, convert, coords
from umbra.common.terminal import cprint
from umbra.common.fits import Header, extract_bayer_pattern
from umbra.common.imageio.extensions import BITMAP, FITS, RAW, SUPPORTED
from umbra.common.imageio._backends import bitmap, fits, raw


_BACKEND_BY_EXTENSIONS: tuple[tuple[frozenset[str], ModuleType], ...] = (
    (RAW, raw),
    (BITMAP, bitmap),
    (FITS, fits),
)


def backend_for(filepath: Path | str) -> ModuleType:
    """Return the I/O backend module responsible for a file, by extension."""
    ext = Path(filepath).suffix.lower()
    for exts, backend in _BACKEND_BY_EXTENSIONS:
        if ext in exts:
            return backend
    raise ValueError(f"Unsupported file extension: {ext}")


def read(
    filepath: Path | str,
    region: coords.Region | None = None,
    *,
    to_float: bool = True,
    debayer: bool = False,
    verbose: bool = True,
) -> tuple[np.ndarray, Header]:
    """Read any supported image file into ``(data, header)``.

    With ``to_float=True`` the pixel data is converted to float in [0, 1]; pass
    ``to_float=False`` to keep the native dtype.

    By default the data is not debayered: CFA sources expose their mosaic and
    record the pattern in ``header["BAYERPAT"]`` (read it via
    :func:`umbra.common.fits.extract_bayer_pattern`). Pass ``debayer=True`` to
    demosaic CFA sources into an RGB image via :func:`umbra.common.bayer.debayer`;
    sources whose header records no Bayer pattern are left untouched.
    """
    if verbose:
        cprint(f"Opening {filepath}...")
    data, header = backend_for(filepath).read(filepath, region)
    if to_float:
        data = convert.to_float(data)
    if debayer:
        pattern = extract_bayer_pattern(header)
        if pattern is not None:
            data = bayer.debayer(data, pattern)
    context.checkstate()
    return data, header


def read_header(filepath: Path | str) -> Header:
    return backend_for(filepath).read_header(filepath)


def read_shape(filepath: Path | str) -> tuple[int, ...]:
    return backend_for(filepath).read_shape(filepath)


def list_files(dirpath: Path | str, *, extensions: frozenset[str] | None = None) -> list[Path]:
    """Return sorted image file paths in a directory.

    By default lists every supported extension; pass ``extensions`` (e.g.
    ``imageio.extensions.FITS``) to restrict to a single format.
    """
    if extensions is None:
        extensions = SUPPORTED
    dirpath = Path(dirpath)
    return sorted(
        p
        for p in dirpath.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )


def write(
    filepath: Path | str,
    data: np.ndarray,
    header: Header | None,
    *,
    verbose: bool = True,
) -> None:
    """Write image data and header to a file, dispatching by extension.

    Raises
    ------
    NotImplementedError
        If the backend for this file type does not support writing.
    """
    backend = backend_for(filepath)
    write = getattr(backend, "write", None)
    if write is None:
        raise NotImplementedError(f"Writing {Path(filepath).suffix} files is not supported.")
    if verbose:
        cprint(f"Writing {filepath}...")
    write(filepath, data, header)
    context.checkstate()
