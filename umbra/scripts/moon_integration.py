import os
import numpy as np

from umbra.common.fits import read_fits_as_float, save_as_fits, get_grouped_filepaths, read_fits_header, extract_subheader

def main(
    moon_registered_dir,
    moon_stacks_dir,
    group_keywords
):
    os.makedirs(moon_stacks_dir, exist_ok=True)

    # Make a dictionary that contains for each group (key) a list of associated filepaths (value)
    grouped_filepaths = get_grouped_filepaths(moon_registered_dir, group_keywords)

    for group_key in grouped_filepaths.keys():
        # Need header to get image shape, and info about group
        header = read_fits_header(grouped_filepaths[group_key][0])
        shape = (header["NAXIS2"], header["NAXIS1"], header["NAXIS3"])
        print(f"Stacking images from group {group_key} :")
        for keyword in group_keywords:
            print(f"    - {keyword} : {header[keyword]}")
        # Initialize stuff            
        stacked_img = np.zeros(shape)
        counts = 0
        # Loop over subs
        for filepath in grouped_filepaths[group_key]:
            # Read image
            img, header = read_fits_as_float(filepath)
            print("Adding image to the main stack...")
            stacked_img += img
            counts +=1

        stacked_img /= counts

        output_header = extract_subheader(header, group_keywords+["PEDESTAL", "MOON-X", "MOON-Y"]) # common keywords
        group_name = " - ".join([f"{group_keywords[i]}_{group_key[i]}" for i in range(len(group_key))])
        save_as_fits(stacked_img, output_header, os.path.join(moon_stacks_dir, f"{group_name}.fits"))
        
if __name__ == "__main__":
    import sys
    import yaml
    from umbra.common.terminal import ColorTerminalStream
    sys.stdout = ColorTerminalStream()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    main(**config["moon_integration"])