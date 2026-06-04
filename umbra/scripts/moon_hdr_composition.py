import os
import numpy as np

from umbra.common import fits, imageio
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

    filepath_to_header = {p: imageio.read_header(p) for p in imageio.list_files(moon_stacks_dir, extensions=imageio.extensions.FITS)}
    grouped_filepaths = fits.group_filepaths(filepath_to_header, group_keywords) # we need sorted files based on irradiance

    # Initialize stuff
    ref_header = imageio.read_header(grouped_filepaths[list(grouped_filepaths.keys())[0]][0])
    ref_scaling_factor = compute_scaling_factor(ref_header, group_keywords)
    shape = (ref_header["NAXIS2"], ref_header["NAXIS1"], ref_header["NAXIS3"])
    hdr_img = np.zeros(shape)
    sum_weights = np.zeros(shape[0:2])

    headers = []
    for group_name in grouped_filepaths.keys():
        # Read image
        img, header = imageio.read(grouped_filepaths[group_name][0])
        headers.append(header)
        # Compute weight
        if group_name == list(grouped_filepaths.keys())[0]: # shortest exposure : no high range clipping
            weights = saturation_weighting(img.max(axis=2), low_clipping_threshold, 1, low_smoothness, high_smoothness)
        elif group_name == list(grouped_filepaths.keys())[-1]: # longest exposure : no low range clipping
            weights = saturation_weighting(img.max(axis=2), 0, high_clipping_threshold, low_smoothness, high_smoothness)
        else:
            weights = saturation_weighting(img.max(axis=2), low_clipping_threshold, high_clipping_threshold, low_smoothness, high_smoothness)
        imageio.write(os.path.join(moon_hdr_dir, f"weights_{group_name}.fits"), weights, None)
        weights = weights * header["EXPTIME"]**2
        # Compute scaled irradiance (normalize to shortest exposure)
        scaling_factor = compute_scaling_factor(header, group_keywords)
        pedestal = header.get("PEDESTAL")
        if pedestal is not None and isinstance(pedestal, (int, float)):
            img = np.maximum(img - pedestal / 65535, 0)
            header.remove("PEDESTAL")
        img = img * ref_scaling_factor / scaling_factor

        # Add weighted scaled irradiance to the HDR image
        hdr_img += weights[:,:,None] * img
        sum_weights += weights

    hdr_img /= sum_weights[:,:,None]
    hdr_img = np.clip(hdr_img, 0, 1)

    output_header = fits.intersect(headers)

    imageio.write(os.path.join(moon_hdr_dir, "hdr.fits"), hdr_img, output_header)
    
if __name__ == "__main__":
    import sys
    import yaml
    from umbra.common.terminal import ColorTerminalStream
    sys.stdout = ColorTerminalStream()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    main(**config["moon_hdr_composition"])
