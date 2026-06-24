from collections.abc import Sequence
from typing import cast
from pathlib import Path
import warnings

import astropy.io.fits
import numpy as np
import scipy.interpolate
import skimage as sk

from umbra import registration
from umbra.common import fits, imageio, transform
from umbra.common.terminal import cprint


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
    fits_dir: Path,
    group_keywords: Sequence[str],
) -> str:
    """Return the reference filename, auto-selecting if not provided."""
    if not ref_filename:
        ref_filename = registration.auto.select_reference(fits_dir, group_keywords)
    else:
        validate_filenames(fits_dir, [ref_filename], "Reference")
    return ref_filename


def resolve_anchor_filenames(
    anchor_filenames: Sequence[str] | None,
    fits_dir: Path,
    group_keywords: Sequence[str],
    num_clipped_pixels: float,
) -> list[str]:
    """Return a validated, sorted list of anchor filenames."""
    filepaths = imageio.list_files(fits_dir, extensions=imageio.extensions.FITS)
    filepath_to_header = {p: imageio.read_header(p) for p in filepaths}

    try:
        filepath_to_timestamp = {p: fits.extract_timestamp(header) for p, header in filepath_to_header.items()}
    except ValueError:
        warnings.warn("Could not extract timestamps from FITS headers. All images will be selected as anchors.")
        resolved_anchor_filenames = [p.name for p in filepaths]

    else:
        if not anchor_filenames:
            resolved_anchor_filenames = registration.auto.select_anchors(fits_dir, group_keywords, num_clipped_pixels)
        else:
            resolved_anchor_filenames = sorted(anchor_filenames, key=lambda f: filepath_to_timestamp[fits_dir / f])
            if len(resolved_anchor_filenames) < 2:
                raise ValueError("At least 2 anchors are required.")
            validate_filenames(fits_dir, resolved_anchor_filenames, "Anchor")

    return resolved_anchor_filenames


def validate_filenames(fits_dir: Path, filenames: Sequence[str], file_type: str) -> None:
    """Validate that all filenames exist in the input directory."""
    for filename in filenames:
        if not (fits_dir / filename).exists():
            raise ValueError(f"{file_type} file {filename} does not exist in the input directory.")
        if not filename.endswith(tuple(imageio.extensions.FITS)):
            raise ValueError(f"{file_type} file {filename} must be a FITS file (.fits or .fit).")


def resolve_remaining_filenames(
    fits_dir: Path,
    ref_filename: str,
    anchor_filenames: Sequence[str],
) -> list[str]:
    """Return sorted filenames of non-ref, non-anchor FITS files in fits_dir."""
    return sorted(
        p.name for p in fits_dir.iterdir()
        if p.suffix in imageio.extensions.FITS
        and p.name != ref_filename
        and p.name not in anchor_filenames
    )


def preprocess_and_detect_moon(
    img: np.ndarray,
    num_clipped_pixels: float,
    num_edge_pixels: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Preprocess an image and detect the moon, returning (preprocessed_img, moon_center, moon_radius)."""
    img = registration.moon.preprocess(img, num_clipped_pixels)
    moon_center, moon_radius = registration.moon.detect_moon(img, num_edge_pixels)
    return img, moon_center, moon_radius


def process_anchors(
    anchor_filenames: list[str],
    fits_dir: Path,
    num_clipped_pixels: float,
    num_edge_pixels: float,
    sigma_high_pass_tangential: float,
    max_iter: int,
    error_overlay_opacity: float,
) -> tuple[list[astropy.io.fits.Header], list[np.ndarray], list[float], list[sk.transform.EuclideanTransform], list[sk.transform.EuclideanTransform]]:
    """Load anchors, detect moon, and sun-align images.

    Processes anchors one at a time to avoid storing all preprocessed images simultaneously.
    Returns (anchor_headers, moon_centers, moon_radii, sun_tforms_from_first_anchor, moon_tforms_from_first_anchor).
    Both transform families have length N and are expressed in the first anchor frame; index i maps anchor[0] -> anchor[i].
    """
    anchor_headers: list[astropy.io.fits.Header] = []
    moon_centers: list[np.ndarray] = []
    moon_radii: list[float] = []
    sun_tforms_pairwise: list[sk.transform.EuclideanTransform] = []

    prev_preprocessed_img: np.ndarray | None = None
    prev_mass_center: tuple[float, float] | None = None

    for i, filename in enumerate(anchor_filenames):
        cprint(f"Processing anchor image {filename} ({i+1}/{len(anchor_filenames)}):", style='bold', color='cyan')
        img, header = imageio.read(fits_dir / filename)
        img, moon_center, moon_radius = preprocess_and_detect_moon(img, num_clipped_pixels, num_edge_pixels)
        preprocessed_img, mass_center = registration.sun.preprocess(img, moon_center, moon_radius, sigma_high_pass_tangential)
        cprint(f"Anchor image {filename} processed successfully ({i+1}/{len(anchor_filenames)}).", color='green')

        if prev_preprocessed_img is not None and prev_mass_center is not None:
            cprint(f"Computing anchor transform {i} -> {i+1}:", style='bold', color='cyan')
            tform = registration.sun.compute_transform(
                prev_preprocessed_img, preprocessed_img,
                prev_mass_center, max_iter, error_overlay_opacity,
            )
            sun_tforms_pairwise.append(tform)
            cprint(f"Anchor transform {i} -> {i+1} computed successfully.", color='green')

        anchor_headers.append(header)
        moon_centers.append(moon_center)
        moon_radii.append(moon_radius)

        prev_preprocessed_img = preprocessed_img
        prev_mass_center = mass_center

    # Compute transforms to each anchor from the first anchor (anchor[0] is identity)
    identity = sk.transform.EuclideanTransform()
    sun_tforms_from_first_anchor: list[sk.transform.EuclideanTransform] = [identity]
    for tform_pairwise in sun_tforms_pairwise:
        sun_tforms_from_first_anchor.append(
            cast(sk.transform.EuclideanTransform, sun_tforms_from_first_anchor[-1] + tform_pairwise)
        )
    # Recreate each anchor's moon registration transform in the first anchor frame: rotate around
    # the first anchor's moon by the sun rotation, then shift its moon onto anchor[i]'s.
    first_anchor_moon_center = moon_centers[0]
    moon_tforms_from_first_anchor = [
        transform.centered_rigid_transform(first_anchor_moon_center, float(t.rotation), moon_center - first_anchor_moon_center)
        for moon_center, t in zip(moon_centers, sun_tforms_from_first_anchor)
    ]

    return anchor_headers, moon_centers, moon_radii, sun_tforms_from_first_anchor, moon_tforms_from_first_anchor

def rebase_transforms_to_ref(
    tform_from_first_anchor_to_ref: sk.transform.EuclideanTransform,
    tforms_from_first_anchor: list[sk.transform.EuclideanTransform],
) -> list[sk.transform.EuclideanTransform]:
    """Convert transforms relative to the first anchor into transforms relative to the reference frame."""
    return [
        cast(sk.transform.EuclideanTransform, tform_from_first_anchor_to_ref.inverse + t)
        for t in tforms_from_first_anchor
    ]

def extract_anchor_values(
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

def build_anchor_values_interpolants(
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

def interpolate_anchor_values(
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


def interpolate_transforms_to_ref(
    sun_tforms_from_first_anchor: list[sk.transform.EuclideanTransform],
    moon_tforms_from_first_anchor: list[sk.transform.EuclideanTransform],
    first_anchor_moon_center: np.ndarray,
    anchor_timestamps: list[float],
    ref_moon_center: np.ndarray,
    ref_timestamp: float,
) -> tuple[sk.transform.EuclideanTransform, sk.transform.EuclideanTransform]:
    """Interpolate the moon and sun transforms from the first anchor to the reference frame.

    Follows the same principle used to register the remaining images: the moon center is detected
    directly, while the anchor values (rotation + sun-moon translation) are interpolated over
    timestamps. Because the reference frame is not yet defined at this stage, the anchor values are
    expressed relative to the first anchor frame; the transforms are then recreated from the detected
    reference moon center and the interpolated values.

    Returns (moon_tform_from_first_anchor_to_ref, sun_tform_from_first_anchor_to_ref).
    """
    rotations, sun_moon_translations = extract_anchor_values(sun_tforms_from_first_anchor, moon_tforms_from_first_anchor)
    rotation_interp, sun_moon_translation_interp = build_anchor_values_interpolants(anchor_timestamps, sun_moon_translations, rotations)
    rotation, sun_moon_translation = interpolate_anchor_values(ref_timestamp, rotation_interp, sun_moon_translation_interp)
    return recreate_transforms(rotation, sun_moon_translation, first_anchor_moon_center, ref_moon_center)


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
    moon_header = fits.update(header, registration.moon.keyword_cards(moon_registered_moon_center, moon_radius))
    sun_header = fits.update(header, registration.moon.keyword_cards(sun_registered_moon_center, moon_radius))
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
