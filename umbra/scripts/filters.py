import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.transform import warp_polar
from scipy.ndimage import gaussian_filter

from umbra.common.display import crop, center_crop, ht
from umbra.common.fits import read_fits_as_float, save_as_fits
from umbra.common.polar import angle_map, radius_map, coords_cart_to_polar, coords_polar_to_cart, warp_cart_to_polar, warp_polar_to_cart
from umbra.common.filters import achf, radial_tangential, partial_filter

# def uniform_grid_points(shape, d):
#     x, y = np.arange(shape[1]), np.arange(shape[0])
#     X, Y = np.meshgrid(x, y)
#     img = np.logical_and(X % d == 0, Y % d == 0).astype('float')
#     return img
# shape = [201, 201]
# img = uniform_grid_points(shape, d = 20)
# x_c, y_c = (shape[1]-1) / 2, (shape[0]-1) / 2

IMAGE_FILEPATH = "D:\\_ECLIPSE2024\\data\\totality\\merged_hdr\\hdr.fits"
MASK_FILEPATH = "D:\\_ECLIPSE2024\\data\\totality\\merged_hdr\\moon_mask.fits"

FILTERED_DIR = "D:\\_ECLIPSE2024\\data\\totality\\filtered"

os.makedirs(FILTERED_DIR, exist_ok=True)

SIGMA = 0.5
mode = 'tangential_radial'

# Load image
img, header = read_fits_as_float(IMAGE_FILEPATH)
img, header = crop(img, 300, -20, 300, -20, header=header)
moon_mask, _ = read_fits_as_float(MASK_FILEPATH)
moon_mask = crop(moon_mask, 300, -20, 300, -20)
mask = 1 - moon_mask
#save_as_fits(img, header, os.path.join(FILTERED_DIR, f"image.fits"), convert_to_uint16=False)
x_c, y_c = header["SUN-X"], header["SUN-Y"]

img = img.mean(axis=2)
mask = mask[:,:,0]
#mask = (mask > 0).astype('float')

if mode == 'tangential_radial' or mode == 'ACHF':
    output_shape = [2000, 4000] # (theta, rho)
    img_polar, theta_factor, rho_factor = warp_cart_to_polar(img, x_c, y_c, output_shape, return_factors=True)
    mask_polar = warp_cart_to_polar(mask, x_c, y_c, output_shape)
    if mode == 'tangential_radial':
        rho_0 = np.nonzero(mask_polar)[1].min() / rho_factor # Extract the (smallest) moon radius
        filter_args = {'sigma': SIGMA, 'rho_0': rho_0, 'rho_factor': rho_factor, 'theta_factor': theta_factor}
        blurred_img_polar = partial_filter(img_polar, mask_polar, radial_tangential, filter_args)
    if mode == 'ACHF':
        j_0 = np.nonzero(mask_polar)[1].min() # Extract the smallest column index that contains a non-zero element in the mask 
        filter_args = {'sigma': SIGMA, 'j_0': j_0, 'rho_factor': rho_factor, 'theta_factor': theta_factor}
        blurred_img_polar = partial_filter(img_polar, mask_polar, achf, filter_args)
    blurred_img = warp_polar_to_cart(blurred_img_polar, x_c, y_c, img.shape)
    
if mode == 'gaussian':
    blurred_img = partial_filter(img, mask, gaussian_filter, {'sigma' : SIGMA})

amount = 10
highpass_img = img - blurred_img
sharpened_img = (1+amount)*img - amount*blurred_img

save_as_fits(highpass_img, header, os.path.join(FILTERED_DIR, f"highpass_{mode}_{SIGMA}.fits"), convert_to_uint16=False)
save_as_fits(sharpened_img, header, os.path.join(FILTERED_DIR, f"{mode}_{SIGMA}.fits"), convert_to_uint16=False)

fig, axes = plt.subplots(2,2)
axes = axes.flatten()

m = 0.01
low = img.min()
high = img.max()
img = ht(img, m, low, high)
blurred_img = ht(blurred_img, m, low, high)
sharpened_img = ht(sharpened_img, m, low, high)

axes[0].imshow(center_crop(img, int(x_c), int(y_c)))
axes[1].imshow(center_crop(mask, int(x_c), int(y_c)))
axes[2].imshow(center_crop(blurred_img, int(x_c), int(y_c)))
axes[3].imshow(center_crop(sharpened_img, int(x_c), int(y_c)))


# m = 0.000001
# sharpened_img = ht(sharpened_img, m, shadow_clipping=True)
plt.show()