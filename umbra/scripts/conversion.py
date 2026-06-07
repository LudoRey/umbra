import os
from pathlib import Path

from umbra.common import bayer, fits, imageio
from umbra.common.terminal import cprint
from umbra.common.typing import CheckStateCallback, ImageCallback
from umbra import conversion


def main(
    images_dir: str | Path,
    fits_dir: str | Path,
    dark_path: str | Path | None = None,
    flat_path: str | Path | None = None,
    bias_path: str | Path | None = None,
    debayer_algorithm: str = "menon",
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
    if flat_path is not None and bias_path is None:
        raise ValueError("Bias is required when flat is provided.")

    dark_master = conversion.load_or_create_master(dark_path, checkstate=checkstate) if dark_path is not None else None
    flat_master = conversion.load_or_create_master(flat_path, checkstate=checkstate) if flat_path is not None else None
    bias_master = conversion.load_or_create_master(bias_path, checkstate=checkstate) if bias_path is not None else None
    calibrating = dark_master is not None or flat_master is not None or bias_master is not None

    os.makedirs(fits_dir, exist_ok=True)
    for idx, filepath in enumerate(filepaths, start=1):
        output_filepath = fits_dir / f"{filepath.stem}.fits"
        cprint(f"Converting {filepath.name} ({idx}/{len(filepaths)})...", style="bold", color="cyan")
        img, header = imageio.read(filepath, checkstate=checkstate)
        img_callback(img)
        pattern = fits.extract_bayer_pattern(header)
        if calibrating:
            cprint("Calibrating...", end=" ", flush=True)
            img = conversion.calibrate(img, dark=dark_master, flat=flat_master, bias=bias_master)
            print("Done.")
        checkstate()
        img_callback(img)
        if pattern is not None:
            img = bayer.debayer(img, pattern, algorithm=debayer_algorithm)
            header.remove("BAYERPAT")
        imageio.write(output_filepath, img, header, checkstate=checkstate)
        img_callback(img)
    cprint("Conversion completed successfully.", style="bold", color="green")


if __name__ == "__main__":
    import sys
    import yaml
    from umbra.common.terminal import ColorTerminalStream
    sys.stdout = ColorTerminalStream()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    main(**config["conversion"])
