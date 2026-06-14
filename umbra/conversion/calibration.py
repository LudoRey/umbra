from pathlib import Path

import numpy as np

from umbra.common import imageio
from umbra.common.terminal import cprint
from umbra.common.typing import CheckStateCallback
from umbra import integration


def calibrate(
    light: np.ndarray,
    dark: np.ndarray | None = None,
    flat: np.ndarray | None = None,
    bias: np.ndarray | None = None,
    pattern: str | None = None,
) -> np.ndarray:
    """
    Calibrate a light frame: ``(light - subtrahend) / normalized_flat``, clipped to [0, 1].

    The subtrahend is the dark frame if provided, otherwise the bias frame, otherwise zero. The flat
    field ``(flat - bias)`` is normalized so it preserves overall brightness without altering color
    balance: each color is divided by its own mean rather than a single global mean. For a CFA mosaic
    (``pattern`` given) the Bayer positions are grouped by color (e.g. ``RGGB`` -> R, G, B) and each
    group normalized against the combined mean of its positions; for a mono frame the global mean is
    used. When no flat is provided, only the dark/bias subtraction is applied. ``bias`` must be
    provided when ``flat`` is. All arrays must share the light's shape.
    """
    subtrahend = dark if dark is not None else bias
    out = light if subtrahend is None else light - subtrahend
    if flat is not None:
        if bias is None:
            raise ValueError("Bias frame is required when flat frame is provided.")
        out = out / _normalize_flat(flat - bias, pattern)
    return np.clip(out, 0, 1)


def _normalize_flat(flat_field: np.ndarray, pattern: str | None) -> np.ndarray:
    """Scale ``flat_field`` so each color has unit mean.

    For a mono frame (``pattern`` None) the global mean is used. For a CFA mosaic the four positions
    of the 2x2 Bayer cell (``pattern`` read row-major, so ``pattern[2 * row + col]``) are grouped by
    color and each group divided by the combined mean of its positions.
    """
    if pattern is None:
        return flat_field / flat_field.mean()
    positions_by_color: dict[str, list[tuple[int, int]]] = {}
    for index, color in enumerate(pattern):
        positions_by_color.setdefault(color, []).append((index // 2, index % 2))
    normalized = np.empty_like(flat_field)
    for positions in positions_by_color.values():
        components = [flat_field[row::2, col::2] for row, col in positions]
        mean = np.concatenate([c.ravel() for c in components]).mean()
        for (row, col), component in zip(positions, components):
            normalized[row::2, col::2] = component / mean
    return normalized


def load_or_create_master(
    path: Path | str,
    type: str,
    outlier_threshold: float | None = None,
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
        cprint(f"Stacking {len(filepaths)} {type} images:", style="bold", color="cyan")
        img, header, _ = integration.integrate(filepaths, outlier_threshold, checkstate=checkstate)
        if save_master:
            imageio.write(path.parent / f"master_{type}.fits", img, header, checkstate=checkstate)
        cprint(f"{type.capitalize()} images stacked successfully.", color="green")
        return img
    return imageio.read(path, checkstate=checkstate)[0]
