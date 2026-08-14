import numpy as np
from astropy.io import fits
from umbra.common.pyx.lut import apply_lut_rgb, apply_lut_grayscale

def combine_red_green(img1, img2):
    img = np.zeros((img1.shape[0], img1.shape[1], 3))
    img[:,:,0] = img1
    img[:,:,1] = img2 
    return img

# Quantiles converge long before every pixel is needed, and this is the whole cost of
# compute_statistics, so the image is sampled rather than read in full.
ROW_SUBSAMPLING = 4
# The black point is read off these precomputed levels instead of the image. Uniform spacing
# puts the finest value resolution where the pixels are densest, which is where the black
# point sits; 2**14 of them resolve it to within one ADU of a 16-bit sensor.
QUANTILE_LEVELS = np.linspace(0, 1, 2**14)

def compute_statistics(x, has_nans: bool = False):
    '''Returns the quantiles of the above-zero pixels of x, and its maximum.'''
    # Sorting outright costs about what a single np.quantile does, and makes every further
    # quantile a lookup, so the sampled rows are sorted once and indexed.
    values = np.sort(x[::ROW_SUBSAMPLING], axis=None)
    if has_nans:
        # np.sort orders NaNs last, leaving the finite values as a prefix.
        values = values[:np.searchsorted(values, np.nan, side="left")]
    # Pixels at or below zero already display as black, so the quantiles are taken among the
    # rest; otherwise a padded border would absorb them and pin the black point at zero.
    zeros = np.searchsorted(values, 0, side="right")
    return {"quantiles": quantile(values, zeros, QUANTILE_LEVELS),
            "max": np.nanmax(x) if has_nans else x.max()}

def quantile(values, start, level):
    '''Linearly interpolated quantile(s) of the sorted values[start:], as np.quantile takes it.
    level may be an array, which costs no more than the sort it is read from.'''
    index = start + np.asarray(level)*(values.size - start - 1)
    low, high = np.floor(index).astype(np.intp), np.ceil(index).astype(np.intp)
    return values[low] + (values[high] - values[low])*(index - low)

def auto_ht_params(statistics, shadow_gain: float = 10, black_point: float = 0.2):
    '''shadow_gain is the slope of the MTF at 0, and the midpoint is exactly its reciprocal.
    black_point is the percentage of the above-zero pixels left below it.'''
    return (1/(1 + shadow_gain),
            np.interp(black_point/100, QUANTILE_LEVELS, statistics["quantiles"]),
            statistics["max"])

def ht(x, m, vmin=None, vmax=None):
    if vmin is None:
        vmin = x.min()
    if vmax is None:
        vmax = x.max()
    x = np.clip(x, vmin, vmax)
    x = (x - vmin)/(vmax - vmin)
    x = mtf(x, m)
    return x

LUT_BITS = 24

def generate_ht_lut(m, vmin, vmax, bits=LUT_BITS):
    x = np.linspace(0,1,2**bits)
    lut = ht(x, m, vmin, vmax)
    lut = (lut * 255).astype(np.uint8)
    return lut

# def apply_lut(x, lut): # super slow, now implemented in Cython
#     x = lut[x]
#     return x

def ht_lut(x, m, vmin=None, vmax=None, bits=LUT_BITS, has_nans: bool = False):
    '''Returns an 8-bit image. NaN pixels are mapped to 0 (black).'''
    nan_mask = None
    if has_nans:
        nan_mask = np.isnan(x)
        x = x.copy()
        x[nan_mask] = 0.0
    lut = generate_ht_lut(m, vmin, vmax, bits)
    if x.ndim == 3:
        result = apply_lut_rgb(x, lut)
    elif x.ndim == 2:
        result = apply_lut_grayscale(x, lut)
    else:
        raise ValueError("Unsupported number of dimensions")
    if nan_mask is not None:
        result[nan_mask] = 0
    return result

def mtf(x, m):
    if m == 0:
        return np.zeros_like(x)
    if m == 1:
        return np.ones_like(x)
    return (m-1)*x/((2*m-1)*x-m)

def add_crop_inset(img, crop_center, crop_radii, scale=4, border_value=np.nan, border_thickness=2):
    # Crop
    i_left, i_right = crop_center[0]-crop_radii[0], crop_center[0]+crop_radii[0]
    j_top, j_bottom = crop_center[1]-crop_radii[1], crop_center[1]+crop_radii[1]
    crop = img[i_left:i_right+1, j_top:j_bottom+1]
    # Crop border
    img[i_left-border_thickness:i_right+1+border_thickness, j_top-border_thickness:j_top] = border_value
    img[i_left-border_thickness:i_right+1+border_thickness, j_bottom+1:j_bottom+1+border_thickness] = border_value
    img[i_left-border_thickness:i_left, j_top-border_thickness:j_bottom+1+border_thickness] = border_value
    img[i_right+1:i_right+1+border_thickness, j_top-border_thickness:j_bottom+1+border_thickness] = border_value
    # Add inset
    inset = crop.repeat(scale,axis=0).repeat(scale,axis=1)
    img[-inset.shape[0]:, -inset.shape[1]:] = inset
    # Inset border 
    img[-inset.shape[0]:, -inset.shape[1]-border_thickness:-inset.shape[1]] = border_value
    img[-inset.shape[0]-border_thickness:-inset.shape[0], -inset.shape[1]:] = border_value

def crop(img, left, right, top, bottom, header=None):
    # Crop image
    new_img = img[top:bottom+1, left:right+1]
    if header is not None:
        # Create and update new header
        new_header = fits.Header(header, copy=True)
        new_header["NAXIS1"], new_header["NAXIS2"] = img.shape[1], img.shape[0] 
        for k, v in new_header.items():
            if k in ["MOON-X", "SUN-X", "TRANS-X"]:
                new_header[k] = v - left 
            if k in ["MOON-Y", "SUN-Y", "TRANS-Y"]:
                new_header[k] = v - top
        return new_img, new_header 
    else:
        return new_img

def center_crop(img, x_c, y_c, w=512, h=512, header=None):
    return crop(img, x_c-w//2, x_c+w//2-1, y_c-h//2, y_c+h//2-1, header)

def normalize(img: np.ndarray) -> np.ndarray:
    return (img - img.min()) / (img.max() - img.min())
