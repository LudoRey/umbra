from pathlib import Path

import numpy as np

from umbra.common import imageio
from umbra.common.typing import CheckStateCallback
from umbra import integration


def calibrate(
    light: np.ndarray,
    dark: np.ndarray | None = None,
    flat: np.ndarray | None = None,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calibrate a light frame: ``(light - subtrahend) / normalized_flat``, clipped to [0, 1].

    The subtrahend is the dark frame if provided, otherwise the bias frame, otherwise zero. The flat
    field ``(flat - bias)`` is normalized by its mean so it preserves the overall brightness. When no
    flat is provided, only the dark/bias subtraction is applied. ``bias`` must be provided when ``flat``
    is. All arrays must share the light's shape (grayscale mosaic or color).
    """
    subtrahend = dark if dark is not None else bias
    out = light if subtrahend is None else light - subtrahend
    if flat is not None:
        if bias is None:
            raise ValueError("Bias frame is required when flat frame is provided.")
        flat_field = flat - bias
        flat_field = flat_field / flat_field.mean()
        out = out / flat_field
    return np.clip(out, 0, 1)


def load_or_create_master(
    path: Path | str,
    outlier_threshold: float = 3.0,
    save_master: bool = True,
    *,
    checkstate: CheckStateCallback = lambda: None,
) -> np.ndarray:
    """
    Resolve a calibration master frame.

    If ``path`` is a directory, its frames are integrated into a master via a sigma-clipped
    mean (``outlier_threshold`` in units of standard deviation). When ``save_master`` is True,
    the master is written as a .fits file named after the directory, in its parent directory.
    If ``path`` is a file, it is treated as a master frame and read directly.
    """
    path = Path(path)
    if path.is_dir():
        filepaths = imageio.list_files(path)
        if not filepaths:
            raise ValueError(f"No supported image files found in {path}.")
        img, header, _ = integration.integrate(filepaths, outlier_threshold, checkstate=checkstate)
        if save_master:
            imageio.write(path.parent / f"master_{path.name}.fits", img, header, checkstate=checkstate)
        return img
    return imageio.read(path, checkstate=checkstate)[0]
