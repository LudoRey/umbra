import numpy as np
import cv2


PATTERNS = frozenset({"RGGB", "BGGR", "GRBG", "GBRG"})


def debayer(img: np.ndarray, pattern: str) -> np.ndarray:
    """Demosaic a single-channel CFA mosaic into an RGB image.

    Parameters
    ----------
    img : np.ndarray
        2D single-channel Bayer mosaic.
    pattern : str
        One of :data:`PATTERNS`.

    Returns
    -------
    np.ndarray
        HxWx3 RGB image.

    Raises
    ------
    ValueError
        If ``img`` is not single-channel or ``pattern`` is unsupported.
    """
    if img.ndim != 2:
        raise ValueError("RAW sensor data must be single-channel before debayering.")
    code_by_pattern = {
        "RGGB": cv2.COLOR_BayerRGGB2RGB,
        "BGGR": cv2.COLOR_BayerBGGR2RGB,
        "GRBG": cv2.COLOR_BayerGRBG2RGB,
        "GBRG": cv2.COLOR_BayerGBRG2RGB,
    }
    try:
        code = code_by_pattern[pattern]
    except KeyError as exc:
        raise ValueError(f"Unsupported Bayer pattern {pattern}.") from exc
    return cv2.cvtColor(img, code)
