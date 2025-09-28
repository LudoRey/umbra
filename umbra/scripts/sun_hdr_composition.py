import os
import numpy as np

from umbra.common.fits import remove_pedestal, read_fits_as_float, save_as_fits, extract_subheader, get_grouped_filepaths, read_fits_header
from umbra.common.disk import binary_disk
from umbra.hdr import saturation_weighting, equalize_brightness, compute_scaling_factor
from umbra.common.polar import angle_map

def main(
    sun_stacks_dir,
    sun_hdr_dir,
    moon_registered_dir,
    group_keywords,
    image_scale,
    extra_radius_pixels,
    low_clipping_threshold,
    low_smoothness,
    high_clipping_threshold,
    high_smoothness
):
    os.makedirs(sun_hdr_dir, exist_ok=True)

    moon_radius_pixels = 0.279 * 3600 / image_scale # TODO: use header's MOON-R 
    moon_radius_pixels += extra_radius_pixels

    grouped_filepaths = get_grouped_filepaths(sun_stacks_dir, group_keywords) # we need sorted files based on irradiance

    # Initialize stuff
    ref_header = read_fits_header(grouped_filepaths[list(grouped_filepaths.keys())[0]][0])
    ref_scaling_factor = compute_scaling_factor(ref_header, group_keywords)
    shape = (ref_header["NAXIS2"], ref_header["NAXIS1"], ref_header["NAXIS3"])
    # Make moon mask (but it is ambiguous; here we arbitrarily take the moon position from the reference image, which can be found in the header of any moon-aligned image)
    filename = [fname for fname in os.listdir(moon_registered_dir)][0]
    moon_header = read_fits_header(os.path.join(moon_registered_dir, filename))
    moon_mask = binary_disk(moon_header["MOON-X"], moon_header["MOON-Y"], radius=moon_radius_pixels, shape=shape[0:2])
    # Make theta image once and for all
    img_theta = angle_map(ref_header["MOON-X"], ref_header["MOON-Y"], shape=shape[0:2]) # TODO: SUN-X and SUN-Y

    # Read first reference image (the longest exposure)
    group_name = list(grouped_filepaths.keys())[-1]
    img_y, header_y = read_fits_as_float(grouped_filepaths[group_name][0])
    # Compute mask and weights
    mask_y = (img_y.max(axis=2) > low_clipping_threshold) * (img_y.max(axis=2) < high_clipping_threshold)
    weights = saturation_weighting(img_y.max(axis=2), 0, high_clipping_threshold, low_smoothness, high_smoothness)
    save_as_fits(weights, None, os.path.join(sun_hdr_dir, f"weights_{group_name}.fits"))
    weights = weights * header_y["EXPTIME"]**2
    # Compute scaled irradiance (normalize to shortest exposure)
    scaling_factor = compute_scaling_factor(header_y, group_keywords)
    img_y = remove_pedestal(img_y, header_y) * ref_scaling_factor / scaling_factor

    # Add weighted image to the HDR image
    hdr_img = weights[:,:,None] * img_y
    sum_weights = weights

    for group_name in reversed(list(grouped_filepaths.keys())[:-1]):
        # Read image to fit
        img_x, header_x = read_fits_as_float(grouped_filepaths[group_name][0])
        # Compute mask and weights
        mask_x = (img_x.max(axis=2) > low_clipping_threshold) * (img_x.max(axis=2) < high_clipping_threshold)
        if group_name == list(grouped_filepaths.keys())[0]: # shortest exposure : no high range clipping
            weights = saturation_weighting(img_x.max(axis=2), low_clipping_threshold, 1, low_smoothness, high_smoothness)
        else:
            weights = saturation_weighting(img_x.max(axis=2), low_clipping_threshold, high_clipping_threshold, low_smoothness, high_smoothness)
        save_as_fits(weights, None, os.path.join(sun_hdr_dir, f"weights_{group_name}.fits"))
        weights = weights * header_x["EXPTIME"]**2
        # Compute scaled irradiance (normalize to shortest exposure)
        scaling_factor = compute_scaling_factor(header_x, group_keywords)
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

    output_header = extract_subheader(header_x, []) # TODO: SUN-X and SUN-Y for filters

    save_as_fits(hdr_img, output_header, os.path.join(sun_hdr_dir, "hdr.fits"), convert_to_uint16=False)
    
if __name__ == "__main__":
    import sys
    import yaml
    from umbra.common.utils import ColorTerminalStream
    sys.stdout = ColorTerminalStream()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    main(**config["sun_hdr_composition"])