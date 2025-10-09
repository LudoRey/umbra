from matplotlib import pyplot as plt
import numpy as np
from umbra.common.fits import get_grouped_filepaths
from umbra.common.display import ht_lut
from umbra.common.utils import Timer
from umbra.integration.io import read_stack
import astropy.stats

def main(
    moon_registered_dir,
    sun_registered_dir,
    moon_stacks_dir,
    sun_stacks_dir,
    group_keywords,
    image_scale,
    rejection_threshold,
    extra_radius_pixels,
    smoothness
):
    # Get filepaths
    grouped_filepaths = get_grouped_filepaths(sun_registered_dir, group_keywords)
    group = ("0.06667",)
    filepaths = grouped_filepaths[group]
    filepaths = filepaths[::3]

    # Read a small subregion of the stack. there should be a blue hot pixel going from top left to bottom right
    left = 4184
    top = 1424
    width = 280
    height = 280

    stack = read_stack(filepaths, [top, top + height])
    stack = stack[:, :, left:left+width]
        
    # Sigma clip
    # with copy = False, the resulting masked array's data shares memory with the input array, so be careful
    # sigma = 3 is the default in astropy / pixinsight and works ok for stdfunc = 'std', not for 'mad_std'
    with Timer():
        masked_stack: np.ma.MaskedArray = astropy.stats.sigma_clip(
            stack, 
            sigma=rejection_threshold, 
            stdfunc='mad_std', 
            axis=0, 
            copy=False, 
            maxiters=1
        )

    # Visualize results
    rejection_map = masked_stack.mask.astype(np.float32).max(axis=0)

    max_dev = np.max(np.abs(stack - np.median(stack, axis=0)), axis=0)
    std_dev = astropy.stats.mad_std(stack, axis=0) # np.std(stack, axis=0)
    normalized_max_dev = max_dev / std_dev # in units of std deviation
    
    img = masked_stack.mean(axis=0) # Since stack is a masked array, the mean ignores masked values

    fig, axes = plt.subplots(2,3)
    axes[0,0].imshow(ht_lut(img, m=0.003518, vmin=0.001835, vmax=1))
    axes[0,1].imshow(rejection_map)
    axes[1,0].imshow(normalized_max_dev[...,0])
    axes[1,1].imshow(normalized_max_dev[...,1])
    axes[1,2].imshow(normalized_max_dev[...,2])
    plt.show()
    
if __name__ == "__main__":
    import sys
    import yaml
    from umbra.common.utils import ColorTerminalStream
    sys.stdout = ColorTerminalStream()

    with open("config_custom.yaml") as f:
        config = yaml.safe_load(f)

    main(**config["integration"])
