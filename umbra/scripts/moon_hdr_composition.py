import os
import numpy as np

from umbra.common.fits import remove_pedestal, read_fits_as_float, save_as_fits, extract_subheader, read_fits_header, get_grouped_filepaths
from parameters import MOON_HDR_DIR, MOON_STACKS_DIR
from parameters import GROUP_KEYWORDS

from umbra.hdr.hdr import saturation_weighting, compute_scaling_factor

LOW_CLIPPING_THRESHOLD = 0.005
LOW_SMOOTHNESS = 0.001
HIGH_CLIPPING_THRESHOLD = 0.1 # if exptimes[i+1]/exptimes[i] = k, then we should have HIGH/LOW << k
HIGH_SMOOTHNESS = 0.01

os.makedirs(MOON_HDR_DIR, exist_ok=True)

grouped_filepaths = get_grouped_filepaths(MOON_STACKS_DIR, GROUP_KEYWORDS) # we need sorted files based on irradiance

# Initialize stuff
ref_header = read_fits_header(grouped_filepaths[list(grouped_filepaths.keys())[0]][0])
ref_scaling_factor = compute_scaling_factor(ref_header, GROUP_KEYWORDS)
shape = (ref_header["NAXIS2"], ref_header["NAXIS1"], ref_header["NAXIS3"])
hdr_img = np.zeros(shape)
sum_weights = np.zeros(shape[0:2])

for group_name in grouped_filepaths.keys():
    # Read image
    img, header = read_fits_as_float(grouped_filepaths[group_name][0])
    # Compute weight
    if group_name == list(grouped_filepaths.keys())[0]: # shortest exposure : no high range clipping
        weights = saturation_weighting(img.max(axis=2), LOW_CLIPPING_THRESHOLD, 1, LOW_SMOOTHNESS, HIGH_SMOOTHNESS)
    elif group_name == list(grouped_filepaths.keys())[-1]: # longest exposure : no low range clipping
        weights = saturation_weighting(img.max(axis=2), 0, HIGH_CLIPPING_THRESHOLD, LOW_SMOOTHNESS, HIGH_SMOOTHNESS)
    else:
        weights = saturation_weighting(img.max(axis=2), LOW_CLIPPING_THRESHOLD, HIGH_CLIPPING_THRESHOLD, LOW_SMOOTHNESS, HIGH_SMOOTHNESS)
    save_as_fits(weights, None, os.path.join(MOON_HDR_DIR, f"weights_{group_name}.fits"))
    weights = weights * header["EXPTIME"]**2
    # Compute scaled irradiance (normalize to shortest exposure)
    scaling_factor = compute_scaling_factor(header, GROUP_KEYWORDS)
    img = remove_pedestal(img, header) * ref_scaling_factor / scaling_factor

    # Add weighted scaled irradiance to the HDR image
    hdr_img += weights[:,:,None] * img
    sum_weights += weights

hdr_img /= sum_weights[:,:,None]
hdr_img = np.clip(hdr_img, 0, 1)

output_header = extract_subheader(header, ["MOON-X", "MOON-Y"])

save_as_fits(hdr_img, output_header, os.path.join(MOON_HDR_DIR, "hdr.fits"), convert_to_uint16=False)