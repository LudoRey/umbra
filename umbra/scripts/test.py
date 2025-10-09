from umbra.common.fits import get_grouped_filepaths, read_fits_as_float, read_fits
from umbra.integration import memory
from memory_profiler import profile
import numpy as np

def main(
    sun_registered_dir,
    group_keywords,
    **kwargs
):
    # Get filepaths
    grouped_filepaths = get_grouped_filepaths(sun_registered_dir, group_keywords)
    group = ("0.06667",)
    filepaths = grouped_filepaths[group]
    filepath = filepaths[0]
    
    top = 1424
    height = 1000

    with memory.memory_tracker():
        img, header = profile(read_fits_as_float)(filepath, rows_range=(top, top + height))

if __name__ == "__main__":
    import sys
    import yaml
    from umbra.common.utils import ColorTerminalStream
    sys.stdout = ColorTerminalStream()

    with open("config_custom.yaml") as f:
        config = yaml.safe_load(f)

    main(**config["integration"])