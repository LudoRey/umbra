import numpy as np


PATTERNS = frozenset({"RGGB", "BGGR", "GRBG", "GBRG"})
ALGORITHMS = frozenset({"bilinear", "malvar", "menon"})


def debayer(img: np.ndarray, pattern: str, algorithm: str = "bilinear") -> np.ndarray:
    """Demosaic a single-channel CFA mosaic into an RGB image.

    Operates on floating-point data directly, so it can run on calibrated
    frames whose values may be negative or exceed the original ADU range.

    Parameters
    ----------
    img : np.ndarray
        2D single-channel Bayer mosaic.
    pattern : str
        One of :data:`PATTERNS`.
    algorithm : str, optional
        Demosaicing algorithm, one of :data:`ALGORITHMS`:
        ``"bilinear"`` (fastest, default), ``"malvar"`` (Malvar et al., 2004), or
        ``"menon"`` (Menon et al., 2007, DDFAPD; highest quality).

    Returns
    -------
    np.ndarray
        HxWx3 RGB image with the same dtype as ``img``.

    Raises
    ------
    ValueError
        If ``img`` is not single-channel, ``pattern`` is unsupported, or
        ``algorithm`` is unknown.
    """
    if img.ndim != 2:
        raise ValueError("RAW sensor data must be single-channel before debayering.")
    if pattern not in PATTERNS:
        raise ValueError(f"Unsupported Bayer pattern {pattern}.")
    # Imported lazily: colour_demosaicing pulls in the "colour" (and with it scipy/matplotlib)
    # which takes a long time to import, especially in frozen environments.
    from colour_demosaicing import (
        demosaicing_CFA_Bayer_bilinear,
        demosaicing_CFA_Bayer_Malvar2004,
        demosaicing_CFA_Bayer_Menon2007,
    )

    demosaic_by_algorithm = {
        "bilinear": demosaicing_CFA_Bayer_bilinear,
        "malvar": demosaicing_CFA_Bayer_Malvar2004,
        "menon": demosaicing_CFA_Bayer_Menon2007,
    }
    try:
        demosaic = demosaic_by_algorithm[algorithm]
    except KeyError as exc:
        raise ValueError(f"Unsupported demosaicing algorithm {algorithm}.") from exc
    print("Debayering...", end="", flush=True)
    rgb = demosaic(img, pattern)
    print("Done.")
    return np.clip(rgb, 0, 1)
