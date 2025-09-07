import os
import numpy as np

from umbra.common.disk import linear_falloff_disk
from umbra.common.fits import read_fits_as_float, save_as_fits, extract_subheader, read_fits_header, get_grouped_filepaths
from parameters import MOON_RADIUS_DEGREE
from parameters import IMAGE_SCALE
from parameters import SUN_REGISTERED_DIR, SUN_STACKS_DIR
from parameters import GROUP_KEYWORDS

EXTRA_RADIUS_PIXELS = 2
SMOOTHNESS = 10

moon_radius_pixels = MOON_RADIUS_DEGREE * 3600 / IMAGE_SCALE
moon_radius_pixels += EXTRA_RADIUS_PIXELS

os.makedirs(SUN_STACKS_DIR, exist_ok=True)

# Make a dictionary that contains for each group (key) a list of associated filepaths (value)
grouped_filepaths = get_grouped_filepaths(SUN_REGISTERED_DIR, GROUP_KEYWORDS)

for group_key in grouped_filepaths.keys():
    # Need header to get image shape, and info about group
    header = read_fits_header(grouped_filepaths[group_key][0])
    shape = (header["NAXIS2"], header["NAXIS1"], header["NAXIS3"])
    print(f"Stacking images from group {group_key} :")
    for keyword in GROUP_KEYWORDS:
        print(f"    - {keyword} : {header[keyword]}")
    # Initialize stuff
    stacked_img = np.zeros(shape)
    filler_img = np.zeros(shape)
    sum_weights = np.zeros(shape[0:2])
    max_dist_to_moon_center = np.zeros(shape[0:2])
    # Loop over subs
    for filepath in grouped_filepaths[group_key]:
        # Read image
        img, header = read_fits_as_float(filepath)
        # Get the moon center coordinates of the current frame
        moon_x_c, moon_y_c = header["MOON-X"], header["MOON-Y"] 

        ### Update main stack image 
        # 1. Add pixels outside the moon mask (dist_to_moon_center is useful later)
        print("Computing moon mask used to weight image...")
        moon_mask, dist_to_moon_center = linear_falloff_disk(moon_x_c, moon_y_c, moon_radius_pixels, shape[0:2], smoothness=SMOOTHNESS, return_dist=True)
        print("Adding weighted image to the main stack...")
        stacked_img += img * (1-moon_mask[:,:,None])
        # 2. Update sum_weights
        new_in_stack = np.logical_and(moon_mask < 1, sum_weights == 0) # used later to remove those pixels from the filler image
        sum_weights += 1-moon_mask

        ### Update filler image (which only contains pixels NOT in the main stack, i.e with sum_weights == 0)
        print("Updating filler image...")
        # 1. Remove pixels that are now in the main stack
        filler_img[new_in_stack] = 0
        # 2. For pixels that are not in the main stack : use those that are the farthest so far from the moon center
        # Effectively, only the subs nearest to C2 and C3 will be used.
        farthest_from_moon_center_so_far = dist_to_moon_center >= max_dist_to_moon_center
        mask = (sum_weights == 0) * farthest_from_moon_center_so_far
        filler_img[mask] = img[mask]
        #print(f"Updated {np.count_nonzero(mask)/np.count_nonzero(sum_weights == 0)*100:.0f}% pixels of the filler image.")
        # 3. Update max distance tracker
        max_dist_to_moon_center[farthest_from_moon_center_so_far] = dist_to_moon_center[farthest_from_moon_center_so_far]

    stacked_img[sum_weights != 0] /= sum_weights[sum_weights != 0, None]
    merged_img = np.copy(stacked_img)
    merged_img[sum_weights == 0] = filler_img[sum_weights == 0]

    output_header = extract_subheader(header, GROUP_KEYWORDS+["PEDESTAL", "SUN-X", "SUN-Y"]) # common keywords
    group_name = " - ".join([f"{GROUP_KEYWORDS[i]} {group_key[i]}" for i in range(len(group_key))])
    save_as_fits(merged_img, output_header, os.path.join(SUN_STACKS_DIR, f"{group_name}.fits"))