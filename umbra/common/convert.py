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
    """Convert an image array to uint16 in [0, 65535]."""
    if np.issubdtype(img.dtype, np.unsignedinteger):
        shift = np.iinfo(img.dtype).bits - 16
        if shift == 0: # already uint16
            return img
        if shift > 0:  # wider than uint16: keep the high bits
            return (img >> shift).astype(np.uint16)
        else:  # narrower than uint16: widen by replicating the bit pattern (uint8 0xAB -> 0xABAB)
            img = img.astype(np.uint16)
            return (img << -shift) | img
    elif np.issubdtype(img.dtype, np.floating):
        if _is_outside_zero_one(img):
            warnings.warn("Floating point image values outside [0, 1] were clipped.", UserWarning)
        # Bounding the values is what makes the cast meaningful: casting 1.5 or -0.5 wraps
        # around to an unrelated brightness instead of saturating.
        scaled = img * 65535
        np.clip(scaled, 0, 65535, out=scaled)
        return scaled.astype(np.uint16)
    else:
        raise ValueError(f"Unsupported image dtype {img.dtype}, could not convert to uint16.")


def _is_outside_zero_one(img: np.ndarray) -> bool:
    return bool(img.min() < 0 or img.max() > 1)

