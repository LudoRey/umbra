import warnings

import numpy as np


def to_float32(img: np.ndarray) -> np.ndarray:
    """Convert an image array to float32 in [0, 1].

    Floating point values already outside [0, 1] are passed through: they are representable,
    and whoever has to bound them -- a display, a conversion to a fixed-point dtype -- knows
    better than this function what to do about them.
    """
    if np.issubdtype(img.dtype, np.floating):
        if _is_outside_zero_one(img):
            warnings.warn("Floating point image values fall outside [0, 1].", UserWarning)
        return np.asarray(img, dtype=np.float32)
    elif np.issubdtype(img.dtype, np.unsignedinteger):
        result = img.astype(np.float32)
        result /= np.iinfo(img.dtype).max
        return result
    elif np.issubdtype(img.dtype, np.signedinteger):
        result = img.astype(np.float32) / np.iinfo(img.dtype).max
        if result.min() < 0:
            raise ValueError("Signed integer image contains negative values.")
        warnings.warn(
            f"Image has signed integer dtype ({img.dtype}), which is not officially supported. "
            "Consider using unsigned integer or floating point.",
            UserWarning,
        )
        return result
    else:
        raise ValueError(f"Unsupported image dtype {img.dtype}, could not convert to float32.")


def to_uint16(img: np.ndarray) -> np.ndarray:
    """Rescale an image so that its full scale spans [0, 65535], as uint16.

    Full scale is 1.0 for floating point images and the dtype maximum for integer ones. What
    an individual image happens to reach is never consulted, so a scene keeps its brightness
    from frame to frame. Values are truncated rather than rounded, biasing them down by up to
    one step; the endpoints stay exact either way.

    Raises
    ------
    ValueError
        If ``img`` is neither floating point nor unsigned integer.
    """
    if np.issubdtype(img.dtype, np.floating):
        if _is_outside_zero_one(img):
            warnings.warn("Floating point image values outside [0, 1] were clipped.", UserWarning)
        # Bounding the values is what makes the cast meaningful: casting 1.5 or -0.5 wraps
        # around to an unrelated brightness instead of saturating.
        values = img * 65535
        np.clip(values, 0, 65535, out=values)
    elif np.issubdtype(img.dtype, np.unsignedinteger):
        bits = np.iinfo(img.dtype).bits
        if bits > 16:  # drop the low bits: 0xABCD1234 -> 0xABCD
            values = img >> (bits - 16)
        elif bits < 16:  # 65535 = 255 * 257, so replicating the byte lands 0xFF on 0xFFFF
            widened = img.astype(np.uint16)
            values = (widened << (16 - bits)) | widened
        else:
            values = img
    else:
        raise ValueError(f"Unsupported image dtype {img.dtype}, could not convert to uint16.")
    return values.astype(np.uint16)


def _is_outside_zero_one(img: np.ndarray) -> bool:
    return bool(img.min() < 0 or img.max() > 1)

