import os
import numpy as np

from umbra.common.fits import remove_pedestal, read_fits_as_float, save_as_fits, extract_subheader, read_fits_header, get_grouped_filepaths
from umbra.hdr import saturation_weighting, compute_scaling_factor

def main(
    moon_stacks_dir,
    moon_hdr_dir,
    group_keywords,
    low_clipping_threshold,
    low_smoothness,
    high_clipping_threshold,
    high_smoothness
):
    os.makedirs(moon_hdr_dir, exist_ok=True)

    grouped_filepaths = get_grouped_filepaths(moon_stacks_dir, group_keywords) # we need sorted files based on irradiance

    # Initialize stuff
    ref_header = read_fits_header(grouped_filepaths[list(grouped_filepaths.keys())[0]][0])
    ref_scaling_factor = compute_scaling_factor(ref_header, group_keywords)
    shape = (ref_header["NAXIS2"], ref_header["NAXIS1"], ref_header["NAXIS3"])
    hdr_img = np.zeros(shape)
    sum_weights = np.zeros(shape[0:2])

    for group_name in grouped_filepaths.keys():
        # Read image
        img, header = read_fits_as_float(grouped_filepaths[group_name][0])
        # Compute weight
        if group_name == list(grouped_filepaths.keys())[0]: # shortest exposure : no high range clipping
            weights = saturation_weighting(img.max(axis=2), low_clipping_threshold, 1, low_smoothness, high_smoothness)
        elif group_name == list(grouped_filepaths.keys())[-1]: # longest exposure : no low range clipping
            weights = saturation_weighting(img.max(axis=2), 0, high_clipping_threshold, low_smoothness, high_smoothness)
        else:
            weights = saturation_weighting(img.max(axis=2), low_clipping_threshold, high_clipping_threshold, low_smoothness, high_smoothness)
        save_as_fits(weights, None, os.path.join(moon_hdr_dir, f"weights_{group_name}.fits"))
        weights = weights * header["EXPTIME"]**2
        # Compute scaled irradiance (normalize to shortest exposure)
        scaling_factor = compute_scaling_factor(header, group_keywords)
        img = remove_pedestal(img, header) * ref_scaling_factor / scaling_factor

        # Add weighted scaled irradiance to the HDR image
        hdr_img += weights[:,:,None] * img
        sum_weights += weights

    hdr_img /= sum_weights[:,:,None]
    hdr_img = np.clip(hdr_img, 0, 1)

    output_header = extract_subheader(header, ["MOON-X", "MOON-Y"])

    save_as_fits(hdr_img, output_header, os.path.join(moon_hdr_dir, "hdr.fits"), convert_to_uint16=False)
    
if __name__ == "__main__":
    import sys
    import yaml
    from umbra.common.utils import ColorTerminalStream
    sys.stdout = ColorTerminalStream()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    main(**config["moon_hdr_composition"])