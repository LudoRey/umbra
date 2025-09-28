from matplotlib import pyplot as plt
import numpy as np
from umbra.common.fits import read_fits_as_float, get_grouped_filepaths, ht_lut, Timer
from umbra.integration import read_stack
from astropy.stats import sigma_clip

def main(
    sun_registered_dir,
    group_keywords
):
    # Make a dictionary that contains for each group (key) a list of associated filepaths (value)
    grouped_filepaths = get_grouped_filepaths(sun_registered_dir, group_keywords)
    group_name = next(iter(grouped_filepaths))
    group_name = "EXP-0.06667"

    filepaths = grouped_filepaths[group_name]
    #filepaths = filepaths[:5]

    row_start = 3295
    row_end = 3315

    col_start = 970
    col_end = 980

    stack = read_stack(filepaths, [row_start, row_end])
    stack = stack[:,:,col_start:col_end]

    stack = np.ma.MaskedArray(stack, np.zeros_like(stack))
    with Timer():
        stack = sigma_clip(stack, sigma_lower=100, sigma_upper=8, stdfunc='std', axis=0, copy=False)

    rejection_map = stack.mask.astype('float')
    rejection_map = rejection_map.max(axis=0)
    #rejection_map = rejection_map.mean(axis=2)

    img = stack.mean(axis=0)

    fig, axes = plt.subplots(1,2)
    axes[0].imshow(ht_lut(img.data, m=0.003518, vmin=0.001835, vmax=1))
    axes[1].imshow(rejection_map)
    plt.show()
