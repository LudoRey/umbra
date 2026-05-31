import warnings

import numpy as np


def to_float(img: np.ndarray) -> np.ndarray:
    """Convert an image array to float32 in [0, 1]."""
    if np.issubdtype(img.dtype, np.floating):
        if img.min() < 0 or img.max() > 1:
            raise ValueError("Floating point image values must be in the range [0, 1].")
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
        raise ValueError(f"Unsupported image dtype {img.dtype}.")
