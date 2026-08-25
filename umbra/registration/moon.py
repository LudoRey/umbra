from dataclasses import dataclass
from typing import cast

import numpy as np
import skimage as sk
import cv2
import warnings
import astropy.io.fits

from umbra.common import context
from umbra.common.terminal import cprint


# Apparent lunar radius in degrees. The upper bound is reached at perigee; the lower bound is the
# smallest apparent solar radius, below which the eclipse could not be total.
MIN_RADIUS_DEGREE = 0.262
MAX_RADIUS_DEGREE = 0.279


@dataclass
class Detection:
    """Moon geometry recovered from one frame by :func:`detect_moon`."""
    center: np.ndarray      # [x, y] in pixels
    radius: float           # semi-major axis of the fitted ellipse, in pixels
    flattening: float       # 1 - semi_minor/semi_major of the fitted ellipse
    inlier_fraction: float  # RANSAC inliers over detected edge pixels


def preprocess(
    img: np.ndarray,
    num_clipped_pixels: float,
) -> np.ndarray:
    cprint("Moon preprocessing:", style='bold')
    # Convert to grayscale float32
    img = img.astype(np.float32)
    if len(img.shape) == 3:
        img = img.mean(axis=2)

    # Rescale and clip pixels to make moon border more defined
    print(f"Clipping {num_clipped_pixels:.0f} pixels...", end=" ", flush=True)
    img, threshold = clip_brightest_pixels(img, num_clipped_pixels)
    context.checkstate()
    context.emit_image(img)
    print(f"Threshold : {threshold:.4f}.")
    return img

def detect_moon(
    img: np.ndarray,
    num_edge_pixels: float,
    image_scale: float,
) -> Detection:
    '''Fit an ellipse to the moon edge.'''
    # Compute image gradient (edges)
    print(f"Computing edge map...", end=" ", flush=True)
    edge_map = compute_canny_edge_map(img, num_edge_pixels)
    edge_coords = np.column_stack(np.nonzero(edge_map))
    context.checkstate()
    print(f"Found {len(edge_coords)} edge pixels.")

    # RANSAC fitting
    min_samples = 5
    if len(edge_coords) <= min_samples:
        raise RuntimeError("Not enough edge pixels to fit an ellipse.")
    cprint("Detecting moon:", style='bold')
    print("RANSAC fitting...", end=" ", flush=True)
    (moon_y, moon_x, semi_major, semi_minor, _), inliers_coords, outliers_coords = ransac_ellipse_fit(edge_coords, min_samples)
    context.checkstate()
    context.emit_image(make_ransac_img(img, inliers_coords, outliers_coords))
    inlier_fraction = len(inliers_coords) / len(edge_coords)
    moon_radius = float(semi_major)
    flattening = 1 - float(semi_minor) / moon_radius
    print(f"Found {len(inliers_coords)} inliers ({inlier_fraction:.0%} of edge pixels).")
    print("Detected moon:")
    print(f"- Center: ({moon_x:.2f}, {moon_y:.2f})")
    print(f"- Radius: {moon_radius:.2f}")
    print(f"- Flattening: {flattening:.4f}")
    validate_radius(moon_radius, image_scale)

    return Detection(np.array([moon_x, moon_y]), moon_radius, flattening, inlier_fraction)

def degree_to_pixels(angle: float, image_scale: float) -> float:
    '''Convert an angle in degrees to a length in pixels, for an image scale in arcseconds/pixel.'''
    return angle * 3600 / image_scale

def radius_bounds(image_scale: float, tolerance: float = 0.05) -> tuple[float, float]:
    '''Plausible detected-radius range in pixels (the acceptance range of :func:`validate_radius`).

    The Earth-moon distance is unknown, so any radius between MIN_RADIUS_DEGREE and
    MAX_RADIUS_DEGREE is accepted. The tolerance widens that range on both ends, to absorb
    image scale and detection inaccuracies.
    '''
    lower = (1 - tolerance) * degree_to_pixels(MIN_RADIUS_DEGREE, image_scale)
    upper = (1 + tolerance) * degree_to_pixels(MAX_RADIUS_DEGREE, image_scale)
    return lower, upper

def validate_radius(moon_radius: float, image_scale: float, tolerance: float = 0.05) -> None:
    '''Raise if the detected radius is implausible.'''
    lower, upper = radius_bounds(image_scale, tolerance)
    if not lower <= moon_radius <= upper:
        raise RuntimeError(
            f"Detected moon radius ({moon_radius:.2f} px) is outside the expected range "
            f"({lower:.2f} - {upper:.2f} px): check the image scale and the moon detection parameters."
        )

def clip_brightest_pixels(img: np.ndarray, num_clipped_pixels: float) -> tuple[np.ndarray, float]:
    # Compute threshold
    quantile_threshold = 1 - num_clipped_pixels / img.size
    threshold = np.quantile(img, quantile_threshold)
    img = np.clip(img, 0, threshold) / threshold
    return img, float(threshold)

def compute_canny_edge_map(img: np.ndarray, num_edge_pixels: float) -> np.ndarray:
    '''
    num_edge_pixels is the number of proposed edge pixels.
    The actual number of detected pixels is lower than that due to non-maximum suppression.
    '''
    # Convert to 8 bit
    img = (img * 255).astype(np.uint8)
    # Sobel filtering
    sobel_x = cv2.Sobel(img, cv2.CV_16SC1, 1, 0)
    sobel_y = cv2.Sobel(img, cv2.CV_16SC1, 0, 1)
    sobel_mag = np.abs(sobel_x) + np.abs(sobel_y) # faster approx., used by default in cv2.Canny
    # Compute threshold
    quantile_threshold = 1 - num_edge_pixels / sobel_mag.size
    threshold = np.quantile(sobel_mag, quantile_threshold)
    # Canny with no hysteresis, i.e. only non-maximum suppresion
    edge_map = cv2.Canny(sobel_x, sobel_y, float(threshold), float(threshold)) == 255
    return edge_map

class RadialEllipseModel(sk.measure.EllipseModel):
    '''An ellipse whose residuals are measured along the ray from its center.

    The base class returns the orthogonal distance to the curve. That distance has no closed
    form, so it is solved numerically one point at a time, which dominates the RANSAC loop.
    The radial distance is a closed form and vectorizes.

    The two differ by a factor cos(alpha), alpha being the angle between the ray and the
    curve normal, which is bounded by tan(alpha) <= (a**2 - b**2) / (2ab) and vanishes at the
    ends of both axes. At the 1% flattening of a moon low above the horizon that bounds the
    difference at 5e-5 of the distance, far below the pixel grid the edge coordinates live
    on. The radial distance is never the shorter of the two, so it can only shrink the
    inlier set.
    '''

    def residuals(self, data: np.ndarray) -> np.ndarray:
        xc, yc, semi_major, semi_minor, theta = self.params
        x, y = data[:, 0] - xc, data[:, 1] - yc
        # Polar equation of the ellipse, in the frame where its axes are the coordinate axes.
        psi = np.arctan2(y, x) - theta
        radius = semi_major * semi_minor / np.hypot(semi_minor * np.cos(psi), semi_major * np.sin(psi))
        return np.abs(np.hypot(x, y) - radius)

def ransac_ellipse_fit(
    coords: np.ndarray,
    min_samples: int = 5,
    residual_threshold: float = 1,
    max_trials: int = 10000,
) -> tuple[tuple[float, float, float, float, float], np.ndarray, np.ndarray]:
    '''
    The returned parameters are (yc, xc, semi_major, semi_minor, theta), mirroring the
    (row, column) order of `coords`, with semi_major >= semi_minor.

    Parameters:
    -min_samples: number of random samples used to estimate the model parameters at each iteration.
     5 is the minimum needed to define an ellipse.
    -residual_threshold: inliers are such that their distance to the ellipse is below the threshold.
     Might depend on pixel scale, but shouldnt really be lower than 1...
    -max_trials: number of RANSAC trials
    '''

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # might throw warnings when the samples are degenerate
        model, inliers = sk.measure.ransac(
            coords,
            RadialEllipseModel,
            min_samples=min_samples,
            residual_threshold=residual_threshold,
            max_trials=max_trials
        )
    if model is None or inliers is None:
        raise RuntimeError("RANSAC failed to fit an ellipse.")
    model = cast(RadialEllipseModel, model)
    inliers = cast(np.ndarray, inliers)
    inliers_coords = coords[inliers]
    outliers_coords = coords[~inliers]
    return model.params, inliers_coords, outliers_coords

def make_ransac_img(img: np.ndarray, inliers_coords: np.ndarray, outliers_coords: np.ndarray) -> np.ndarray:
    '''Takes a grayscale image (H,W) as input and returns a color image (H,W,3)
    where green pixels are inliers and red pixels are outliers.'''
    img = img[...,None].repeat(3, axis=-1)
    for i in range(3):
        img[outliers_coords[:,0], outliers_coords[:,1], i] = 1 if i == 0 else 0
        img[inliers_coords[:,0], inliers_coords[:,1], i] = 1 if i == 1 else 0
    return img

def keyword_cards(moon_center: np.ndarray, moon_radius: float) -> list[astropy.io.fits.Card]:
    return [
        astropy.io.fits.Card('MOON-X', moon_center[0], 'X-coordinate of the moon center.'),
        astropy.io.fits.Card('MOON-Y', moon_center[1], 'Y-coordinate of the moon center.'),
        astropy.io.fits.Card('MOON-R', moon_radius, 'Radius of the moon.'),
    ]
