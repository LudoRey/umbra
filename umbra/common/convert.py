import warnings

import numpy as np


def to_float32(img: np.ndarray, *, out: np.ndarray | None = None) -> np.ndarray:
    """Convert an image array to float32 in [0, 1].

    Floating point values already outside [0, 1] are passed through: they are representable,
    and whoever has to bound them -- a display, a conversion to a fixed-point dtype -- knows
    better than this function what to do about them.

    ``out`` receives the result, sparing the caller that already holds a destination (a slice
    of a stack, a transposed buffer) both the allocation and the copy into it.
    """
    _check_out_dtype(out, np.float32)
    if np.issubdtype(img.dtype, np.floating):
        if _is_outside_zero_one(img):
            warnings.warn("Floating point image values fall outside [0, 1].", UserWarning)
        if out is None:
            return np.asarray(img, dtype=np.float32)
        np.copyto(out, img)
        return out
    elif np.issubdtype(img.dtype, np.unsignedinteger):
        return np.divide(img, np.iinfo(img.dtype).max, out=out, dtype=np.float32)
    elif np.issubdtype(img.dtype, np.signedinteger):
        result = np.divide(img, np.iinfo(img.dtype).max, out=out, dtype=np.float32)
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


def to_uint16(img: np.ndarray, *, out: np.ndarray | None = None) -> np.ndarray:
    """Rescale an image so that its full scale spans [0, 65535], as uint16.

    Full scale is 1.0 for floating point images and the dtype maximum for integer ones. What
    an individual image happens to reach is never consulted, so a scene keeps its brightness
    from frame to frame. Values are truncated rather than rounded, biasing them down by up to
    one step; the endpoints stay exact either way.

    ``out`` receives the result, sparing the caller that already holds a destination (a slice
    of a stack, a transposed buffer) both the allocation and the copy into it.

    Raises
    ------
    ValueError
        If ``img`` is neither floating point nor unsigned integer, or if ``out`` is not uint16.
    """
    _check_out_dtype(out, np.uint16)
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
    if out is None:
        return values.astype(np.uint16)
    # Narrowing is never a safe cast, and discarding those bits is the point here.
    np.copyto(out, values, casting="unsafe")
    return out


def _check_out_dtype(out: np.ndarray | None, dtype: type[np.generic]) -> None:
    """Reject a destination the conversion would have to silently reinterpret.

    Numpy would take a float64 destination for a float32 result, and the unsafe cast into a
    uint16 destination would take anything at all.
    """
    if out is not None and out.dtype != dtype:
        raise ValueError(f"out has dtype {out.dtype}, expected {np.dtype(dtype)}.")


def _is_outside_zero_one(img: np.ndarray) -> bool:
    return bool(img.min() < 0 or img.max() > 1)

