import warnings

import numpy as np


def to_float(img: np.ndarray) -> np.ndarray:
    """Convert an image array to float32 in [0, 1]."""
    if np.issubdtype(img.dtype, np.floating):
        _ensure_float_in_zero_one(img)
        return img.astype(np.float32)
    elif np.issubdtype(img.dtype, np.unsignedinteger):
        return img.astype(np.float32) / np.iinfo(img.dtype).max
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
        _ensure_float_in_zero_one(img)
        return (img * 65535).astype(np.uint16)
    else:
        raise ValueError(f"Unsupported image dtype {img.dtype}, could not convert to uint16.")
    
def _ensure_float_in_zero_one(img: np.ndarray):
    if img.min() < 0 or img.max() > 1:
        warnings.warn("Floating point image values outside [0, 1] were clipped.", UserWarning)
        img = np.clip(img, 0, 1)
    return img

