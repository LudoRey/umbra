import os
from pathlib import Path

from umbra.common import bayer, fits, imageio
from umbra.common.terminal import cprint
from umbra.common.typing import CheckStateCallback, ImageCallback


def main(
    images_dir: str | Path,
    fits_dir: str | Path,
    *,
    img_callback: ImageCallback = lambda _img: None,
    checkstate: CheckStateCallback = lambda: None,
) -> None:
    images_dir = Path(images_dir)
    fits_dir = Path(fits_dir)
    if fits_dir.resolve() == images_dir.resolve():
        raise ValueError("Output directory must differ from the input directory.")
    filepaths = imageio.list_files(images_dir)
    if not filepaths:
        raise ValueError(f"No supported image files found in {images_dir}.")

    os.makedirs(fits_dir, exist_ok=True)
    for idx, filepath in enumerate(filepaths, start=1):
        output_filepath = fits_dir / f"{filepath.stem}.fits"
        cprint(f"Converting {filepath.name} ({idx}/{len(filepaths)})...", style="bold", color="cyan")
        img, header = imageio.read(filepath)
        pattern = fits.extract_bayer_pattern(header)
        if pattern is not None:
            img = bayer.debayer(img, pattern, algorithm=debayer_algorithm)
            header.remove("BAYERPAT")  # output is debayered RGB, no longer a mosaic
        imageio.write(output_filepath, img, header)
        img_callback(img)
        checkstate()
    cprint("Conversion completed successfully.", style="bold", color="green")


if __name__ == "__main__":
    import sys
    import yaml
    from umbra.common.terminal import ColorTerminalStream
    sys.stdout = ColorTerminalStream()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    main(**config["conversion"])
