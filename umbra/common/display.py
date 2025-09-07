import numpy as np
from astropy.io import fits
from umbra.common.pyx.lut import apply_lut_rgb, apply_lut_grayscale

def combine_red_green(img1, img2):
    img = np.zeros((img1.shape[0], img1.shape[1], 3))
    img[:,:,0] = img1
    img[:,:,1] = img2 
    return img

def compute_statistics(x):
    '''Returns a dictionary containing median, MAD, and max values of x.'''
    statistics = {}
    statistics["median"] = np.median(x)
    statistics["MAD"] = np.median(np.abs(x-statistics["median"]))*1.4826 # constant to make consistent with gaussian distribution https://pixinsight.com/forum/index.php?threads/pixinsight-1-8-0-ripley-redesigned-statistics-tool.6119/
    statistics["max"] = x.max()
    return statistics

def auto_ht_params(statistics, clip_from_median=-2.8, target_median=0.25):       
    vmax = statistics["max"]
    vmin = statistics["median"] + clip_from_median*statistics["MAD"]
    # Update median
    median = (statistics["median"] - vmin)/(vmax - vmin)
    # Compute the midpoint that yields a specified target median value
    m = mtf(median, target_median) # this might look weird but you can prove that it works
    return m, vmin, vmax

def ht(x, m, vmin=None, vmax=None):
    if vmin is None:
        vmin = x.min()
    if vmax is None:
        vmax = x.max()
    x = np.clip(x, vmin, vmax)
    x = (x - vmin)/(vmax - vmin)
    x = mtf(x, m)
    return x

def generate_ht_lut(m, vmin, vmax, bits=16):
    x = np.linspace(0,1,2**bits)
    lut = ht(x, m, vmin, vmax)
    lut = (lut * 255).astype(np.uint8)
    return lut

# def apply_lut(x, lut): # super slow, now implemented in Cython
#     x = lut[x]
#     return x

def ht_lut(x, m, vmin=None, vmax=None, bits=16):
    '''Returns an 8-bit image.'''
    lut = generate_ht_lut(m, vmin, vmax, bits)
    if x.ndim == 3:
        x = apply_lut_rgb(x, lut) 
    if x.ndim == 2:
        x = apply_lut_grayscale(x, lut) 
    return x

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

def normalize(img):
    return (img - img.min()) / (img.max() - img.min())