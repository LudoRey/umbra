import os
import numpy as np

from umbra.common.fits import remove_pedestal, read_fits_as_float, save_as_fits, extract_subheader, get_grouped_filepaths, read_fits_header
from parameters import MOON_RADIUS_DEGREE
from parameters import IMAGE_SCALE
from parameters import SUN_HDR_DIR, SUN_STACKS_DIR, MOON_REGISTERED_DIR
from parameters import GROUP_KEYWORDS

from umbra.common.disk import binary_disk
from umbra.hdr.hdr import saturation_weighting, equalize_brightness, compute_scaling_factor
from umbra.common.polar import angle_map

LOW_CLIPPING_THRESHOLD = 0.005
LOW_SMOOTHNESS = 0.001
HIGH_CLIPPING_THRESHOLD = 0.1 # if exptimes[i+1]/exptimes[i] = k, then we should have HIGH/LOW << k
HIGH_SMOOTHNESS = 0.01

EXTRA_RADIUS_PIXELS = 10

os.makedirs(SUN_HDR_DIR, exist_ok=True)

moon_radius_pixels = MOON_RADIUS_DEGREE * 3600 / IMAGE_SCALE
moon_radius_pixels += EXTRA_RADIUS_PIXELS

grouped_filepaths = get_grouped_filepaths(SUN_STACKS_DIR, GROUP_KEYWORDS) # we need sorted files based on irradiance

# Initialize stuff
ref_header = read_fits_header(grouped_filepaths[list(grouped_filepaths.keys())[0]][0])
ref_scaling_factor = compute_scaling_factor(ref_header, GROUP_KEYWORDS)
shape = (ref_header["NAXIS2"], ref_header["NAXIS1"], ref_header["NAXIS3"])
# Make moon mask (but it is ambiguous; here we arbitrarily take the moon position from the reference image, which can be found in the header of any moon-aligned image)
filename = [fname for fname in os.listdir(MOON_REGISTERED_DIR) if fname.endswith('fits')][0]
moon_header = read_fits_header(os.path.join(MOON_REGISTERED_DIR, filename))
moon_mask = binary_disk(moon_header["MOON-X"], moon_header["MOON-Y"], radius=moon_radius_pixels, shape=shape[0:2])
# Make theta image once and for all
img_theta = angle_map(ref_header["SUN-X"], ref_header["SUN-Y"], shape=shape[0:2])

# Read first reference image (the longest exposure)
group_name = list(grouped_filepaths.keys())[-1]
img_y, header_y = read_fits_as_float(grouped_filepaths[group_name][0])
# Compute mask and weights
mask_y = (img_y.max(axis=2) > LOW_CLIPPING_THRESHOLD) * (img_y.max(axis=2) < HIGH_CLIPPING_THRESHOLD)
weights = saturation_weighting(img_y.max(axis=2), 0, HIGH_CLIPPING_THRESHOLD, LOW_SMOOTHNESS, HIGH_SMOOTHNESS)
save_as_fits(weights, None, os.path.join(SUN_HDR_DIR, f"weights_{group_name}.fits"))
weights = weights * header_y["EXPTIME"]**2
# Compute scaled irradiance (normalize to shortest exposure)
scaling_factor = compute_scaling_factor(header_y, GROUP_KEYWORDS)
img_y = remove_pedestal(img_y, header_y) * ref_scaling_factor / scaling_factor

# Add weighted image to the HDR image
hdr_img = weights[:,:,None] * img_y
sum_weights = weights

for group_name in reversed(list(grouped_filepaths.keys())[:-1]):
    # Read image to fit
    img_x, header_x = read_fits_as_float(grouped_filepaths[group_name][0])
    # Compute mask and weights
    mask_x = (img_x.max(axis=2) > LOW_CLIPPING_THRESHOLD) * (img_x.max(axis=2) < HIGH_CLIPPING_THRESHOLD)
    if group_name == list(grouped_filepaths.keys())[0]: # shortest exposure : no high range clipping
        weights = saturation_weighting(img_x.max(axis=2), LOW_CLIPPING_THRESHOLD, 1, LOW_SMOOTHNESS, HIGH_SMOOTHNESS)
    else:
        weights = saturation_weighting(img_x.max(axis=2), LOW_CLIPPING_THRESHOLD, HIGH_CLIPPING_THRESHOLD, LOW_SMOOTHNESS, HIGH_SMOOTHNESS)
    save_as_fits(weights, None, os.path.join(SUN_HDR_DIR, f"weights_{group_name}.fits"))
    weights = weights * header_x["EXPTIME"]**2
    # Compute scaled irradiance (normalize to shortest exposure)
    scaling_factor = compute_scaling_factor(header_x, GROUP_KEYWORDS)
    img_x = remove_pedestal(img_x, header_x) * ref_scaling_factor / scaling_factor

    # Create combined mask 
    mask = mask_x * mask_y * ~moon_mask
    
    # Fit image
    print("Equalizing the brightness...")
    img_x = equalize_brightness(img_x, img_theta, img_y, mask, return_coeffs=False)

    # Add weighted scaled irradiance to the HDR image
    hdr_img += weights[:,:,None] * img_x
    sum_weights += weights

    # Fitted image becomes the new reference
    img_y = img_x
    mask_y = mask_x

hdr_img /= sum_weights[:,:,None]
hdr_img = np.clip(hdr_img, 0, 1)

output_header = extract_subheader(header_x, ["SUN-X", "SUN-Y"])

save_as_fits(hdr_img, output_header, os.path.join(SUN_HDR_DIR, "hdr.fits"), convert_to_uint16=False)