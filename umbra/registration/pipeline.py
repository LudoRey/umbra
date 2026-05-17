from collections.abc import Sequence
from typing import cast
from pathlib import Path

import astropy.io.fits
import numpy as np
import scipy.interpolate
import skimage as sk

from umbra import registration
from umbra.common import transform, fits
from umbra.common.terminal import cprint
from umbra.common.typing import CheckStateCallback, ImageCallback


def compute_moon_detection_params(
    image_scale: float,
    clipped_factor: float,
    edge_factor: float,
) -> tuple[float, float]:
    """Compute pixel-space moon detection parameters from physical quantities."""
    moon_radius_pixels = 0.279 * 3600 / image_scale
    num_clipped_pixels = np.pi * (clipped_factor**2 - 1) * moon_radius_pixels**2
    num_edge_pixels = edge_factor * 2 * np.pi * moon_radius_pixels
    return num_clipped_pixels, num_edge_pixels


def resolve_ref_filename(
    ref_filename: str | None,
    input_dir: Path,
    group_keywords: Sequence[str] | None,
) -> str:
    """Return the reference filename, auto-selecting if not provided."""
    if not ref_filename:
        if not group_keywords:
            raise ValueError("Group keywords must be provided for automatic reference selection.")
        ref_filename = registration.auto.select_reference(input_dir, group_keywords)
    if not (input_dir / ref_filename).exists():
        raise ValueError(f"Reference file {ref_filename} does not exist in the input directory.")
    if not ref_filename.endswith((".fits", ".fit")):
        raise ValueError("Reference file must be a FITS file (.fits or .fit).")
    return ref_filename


def resolve_anchor_filenames(
    anchor_filenames: Sequence[str] | None,
    input_dir: Path,
    group_keywords: Sequence[str] | None,
    num_clipped_pixels: float,
) -> list[str]:
    """Return a validated, sorted list of anchor filenames."""
    filepath_headers = fits.read_fits_headers(input_dir)
    if not anchor_filenames:
        if not group_keywords:
            raise ValueError("Group keywords must be provided for automatic anchor selection.")
        return registration.auto.select_anchors(input_dir, group_keywords, num_clipped_pixels)
    if len(anchor_filenames) < 2:
        raise ValueError("At least 2 anchors are required.")
    for filename in anchor_filenames:
        if not (input_dir / filename).exists():
            raise ValueError(f"Anchor file {filename} does not exist in the input directory.")
        if not filename.endswith((".fits", ".fit")):
            raise ValueError(f"Anchor file {filename} must be a FITS file (.fits or .fit).")
    return sorted(
        anchor_filenames,
        key=lambda f: fits.extract_timestamp(filepath_headers[input_dir / f]),
    )


def resolve_remaining_filenames(
    input_dir: Path,
    ref_filename: str,
    anchor_filenames: Sequence[str],
) -> list[str]:
    """Return sorted filenames of non-ref, non-anchor FITS files in input_dir."""
    return sorted(
        p.name for p in input_dir.iterdir()
        if p.suffix in (".fits", ".fit")
        and p.name != ref_filename
        and p.name not in anchor_filenames
    )


def preprocess_and_detect_moon(
    img: np.ndarray,
    num_clipped_pixels: float,
    num_edge_pixels: float,
    *,
    checkstate: CheckStateCallback,
    img_callback: ImageCallback,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Preprocess an image and detect the moon, returning (preprocessed_img, moon_center, moon_radius)."""
    img = registration.moon.preprocess(img, num_clipped_pixels, checkstate=checkstate, img_callback=img_callback)
    moon_center, moon_radius = registration.moon.detect_moon(img, num_edge_pixels, checkstate=checkstate, img_callback=img_callback)
    return img, moon_center, moon_radius


def process_anchors(
    anchor_filenames: list[str],
    input_dir: Path,
    num_clipped_pixels: float,
    num_edge_pixels: float,
    sigma_high_pass_tangential: float,
    max_iter: int,
    error_overlay_opacity: float,
    *,
    img_callback: ImageCallback = lambda _img: None,
    checkstate: CheckStateCallback = lambda: None,
) -> tuple[list[float], list[np.ndarray], list[float], list[sk.transform.EuclideanTransform]]:
    """Load anchors, detect moon, compute sun preprocessing and pairwise sun transforms.

    Processes anchors one at a time to avoid storing all preprocessed images simultaneously.
    Returns (timestamps, moon_centers, moon_radii, sun_tforms_pairwise).
    sun_tforms_pairwise has length N-1; index i maps anchor[i] -> anchor[i+1].
    """
    timestamps: list[float] = []
    moon_centers: list[np.ndarray] = []
    moon_radii: list[float] = []
    sun_tforms_pairwise: list[sk.transform.EuclideanTransform] = []

    prev_preprocessed_img: np.ndarray | None = None
    prev_mass_center: tuple[float, float] | None = None

    for i, filename in enumerate(anchor_filenames):
        cprint(f"Processing anchor image {filename} ({i+1}/{len(anchor_filenames)}):", style='bold', color='cyan')
        img, header = fits.read_fits_as_float(input_dir / filename, checkstate=checkstate)
        img, moon_center, moon_radius = preprocess_and_detect_moon(img, num_clipped_pixels, num_edge_pixels, checkstate=checkstate, img_callback=img_callback)
        preprocessed_img, mass_center = registration.sun.preprocess(img, moon_center, moon_radius, sigma_high_pass_tangential, img_callback=img_callback, checkstate=checkstate)
        cprint(f"Anchor image {filename} processed successfully ({i+1}/{len(anchor_filenames)}).", color='green')

        if prev_preprocessed_img is not None and prev_mass_center is not None:
            cprint(f"Computing transform from anchors {i} -> {i+1}:", style='bold', color='cyan')
            tform = registration.sun.compute_transform(
                prev_preprocessed_img, preprocessed_img,
                prev_mass_center, max_iter, error_overlay_opacity,
                img_callback=img_callback, checkstate=checkstate,
            )
            sun_tforms_pairwise.append(tform)
            cprint(f"Transform from anchors {i} -> {i+1} computed successfully.", color='green')

        timestamps.append(fits.extract_timestamp(header))
        moon_centers.append(moon_center)
        moon_radii.append(moon_radius)

        prev_preprocessed_img = preprocessed_img
        prev_mass_center = mass_center

    return timestamps, moon_centers, moon_radii, sun_tforms_pairwise


def compute_sun_transforms(
    timestamps: list[float],
    sun_tforms_pairwise: list[sk.transform.EuclideanTransform],
    ref_timestamp: float,
) -> list[sk.transform.EuclideanTransform]:
    """Convert pairwise anchor transforms into ref-relative absolute transforms.

    Returns one transform per anchor mapping that anchor -> ref frame.
    """
    # Compute transforms to each anchor from the first anchor (anchor[0] is identity)
    identity = sk.transform.EuclideanTransform()
    sun_tforms_from_anchor_zero: list[sk.transform.EuclideanTransform] = [identity]
    for tform_pairwise in sun_tforms_pairwise:
        sun_tforms_from_anchor_zero.append(
            cast(sk.transform.EuclideanTransform, sun_tforms_from_anchor_zero[-1] + tform_pairwise)
        )
    # Compute transforms to ref from anchor zero, then to each anchor from ref
    sun_tform_from_anchor_zero_to_ref = transform.create_interp(timestamps, sun_tforms_from_anchor_zero)(ref_timestamp)
    return [
        cast(sk.transform.EuclideanTransform, sun_tform_from_anchor_zero_to_ref.inverse + t)
        for t in sun_tforms_from_anchor_zero
    ]


def compute_moon_transforms(
    ref_moon_center: np.ndarray,
    moon_centers: list[np.ndarray],
    sun_tforms: list[sk.transform.EuclideanTransform],
) -> list[sk.transform.EuclideanTransform]:
    """Compute per-anchor moon registration transforms."""
    return [
        transform.centered_rigid_transform(ref_moon_center, float(t.rotation), moon_center - ref_moon_center)
        for moon_center, t in zip(moon_centers, sun_tforms)
    ]

def extract_interpolation_inputs(
    sun_tforms: list[sk.transform.EuclideanTransform],
    moon_tforms: list[sk.transform.EuclideanTransform],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract rotation angles and sun-moon translation vectors for interpolation."""
    rotations = np.array([t.rotation for t in sun_tforms])
    sun_moon_translations = np.array([
        registration.sun.compute_sun_moon_translation(sun_tform, moon_tform)
        for sun_tform, moon_tform in zip(sun_tforms, moon_tforms)
    ])
    return rotations, sun_moon_translations

def build_interpolants(
    timestamps: list[float],
    sun_moon_translations: np.ndarray,
    rotations: np.ndarray,
) -> tuple[scipy.interpolate.interp1d, scipy.interpolate.interp1d]:
    """Build linear interpolants for rotation and sun-moon translation over anchor timestamps.

    Returns (rotation_interp, sun_moon_translation_interp).
    """
    rotation_interp = scipy.interpolate.interp1d(timestamps, rotations, kind="linear", fill_value="extrapolate")  # type: ignore
    sun_moon_translation_interp = scipy.interpolate.interp1d(timestamps, sun_moon_translations, kind="linear", axis=0, fill_value="extrapolate")  # type: ignore
    return rotation_interp, sun_moon_translation_interp

def interpolate_values(
    timestamp: float,
    rotation_interp: scipy.interpolate.interp1d,
    sun_moon_translation_interp: scipy.interpolate.interp1d,
) -> tuple[float, np.ndarray]:
    """Interpolate rotation and sun-moon translation for a given timestamp."""
    rotation = rotation_interp(timestamp).item()
    sun_moon_translation = sun_moon_translation_interp(timestamp)
    return rotation, sun_moon_translation


def recreate_transforms(
    rotation: float,
    sun_moon_translation: np.ndarray,
    ref_moon_center: np.ndarray,
    moon_center: np.ndarray,
) -> tuple[sk.transform.EuclideanTransform, sk.transform.EuclideanTransform]:
    """Recreate moon and sun transforms from given parameters."""
    moon_tform = transform.centered_rigid_transform(ref_moon_center, rotation, moon_center - ref_moon_center)
    sun_tform = cast(sk.transform.EuclideanTransform, transform.translation_transform(sun_moon_translation) + moon_tform)
    return moon_tform, sun_tform


def update_headers(
    header: astropy.io.fits.Header,
    moon_center: np.ndarray,
    moon_radius: float,
    moon_tform: sk.transform.EuclideanTransform | None = None,
    sun_tform: sk.transform.EuclideanTransform | None = None,
) -> tuple[astropy.io.fits.Header, astropy.io.fits.Header]:
    """Produce updated FITS headers for moon-registered and sun-registered images."""
    sun_registered_moon_center = sun_tform.inverse(moon_center)[0] if sun_tform is not None else moon_center
    moon_registered_moon_center = moon_tform.inverse(moon_center)[0] if moon_tform is not None else moon_center
    moon_header = fits.update_header(header, registration.moon.keyword_cards(moon_registered_moon_center, moon_radius))
    sun_header = fits.update_header(header, registration.moon.keyword_cards(sun_registered_moon_center, moon_radius))
    return moon_header, sun_header


def apply_transforms(
    img: np.ndarray,
    header: astropy.io.fits.Header,
    moon_center: np.ndarray,
    moon_radius: float,
    moon_tform: sk.transform.EuclideanTransform,
    sun_tform: sk.transform.EuclideanTransform,
) -> tuple[np.ndarray, astropy.io.fits.Header, np.ndarray, astropy.io.fits.Header]:
    """Warp an image with moon and sun transforms and produce updated headers.

    Returns (moon_registered_img, moon_header, sun_registered_img, sun_header).
    """
    moon_registered_img = transform.warp(img, moon_tform.inverse.params)
    sun_registered_img = transform.warp(img, sun_tform.inverse.params)
    moon_header, sun_header = update_headers(header, moon_center, moon_radius, moon_tform, sun_tform)
    return moon_registered_img, moon_header, sun_registered_img, sun_header
