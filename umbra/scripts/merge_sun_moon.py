import os
import numpy as np
from scipy import ndimage

from umbra.common.fits import read_fits_as_float, combine_headers, save_as_fits
from parameters import MERGED_HDR_DIR

SIGMA = 2
MOON_THRESHOLD = 0.002

os.makedirs(MERGED_HDR_DIR, exist_ok=True)

moon_filepath, sun_filepath = "data\\totality\\moon_hdr\\hdr.fits", "data\\totality\\sun_hdr\\hdr.fits"
img_moon, header_moon = read_fits_as_float(moon_filepath)
img_sun, header_sun = read_fits_as_float(sun_filepath)

print(f"Merging moon and sun images...")

x_c = header_moon["MOON-X"]
y_c = header_moon["MOON-Y"]

# Extract binary moon mask
threshold_mask = img_moon.mean(axis=2) < MOON_THRESHOLD
label_map, _ = ndimage.label(threshold_mask)
moon_label = label_map[int(y_c), int(x_c)]
moon_mask = (label_map == moon_label).astype('float')
# Outward-only smoothing of the mask
moon_mask = ndimage.gaussian_filter(moon_mask, sigma=SIGMA)
moon_mask = np.clip(2*moon_mask, 0, 1)
# We only want to add the moon, which correspond to dark pixels.
# The border of the moon mask may be associated with bright pixels (esp. when sigma is high), which should be disregarded (or weighted less)
# We correct the border of the mask by multiplying it with the inverse of the moon intensity (scaled to [0,1] in that region)
border_pixels = (moon_mask > 0)*(moon_mask < 1)
corrections = (img_moon.mean(axis=2) < img_sun.mean(axis=2))[border_pixels == 1]
moon_mask[border_pixels == 1] *= corrections

img_merged = moon_mask[:,:,None]*img_moon + (1-moon_mask)[:,:,None]*img_sun

header_merged = combine_headers(header_moon, header_sun)
save_as_fits(img_merged, header_merged, os.path.join(MERGED_HDR_DIR, f"hdr.fits"), convert_to_uint16=False)
save_as_fits(moon_mask[:,:,None], None, os.path.join(MERGED_HDR_DIR, f"moon_mask.fits"))