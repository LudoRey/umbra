import numpy as np
import skimage as sk
import cv2
import warnings

from umbra.common.utils import cprint

def preprocess(img, num_clipped_pixels, *, checkstate, img_callback):
    cprint("Preprocessing:", style='bold')
    # Convert to grayscale float32
    img = img.astype(np.float32)
    if len(img.shape) == 3:
        img = img.mean(axis=2)

    # Rescale and clip pixels to make moon border more defined
    print(f"Clipping {num_clipped_pixels:.0f} pixels...", end=" ", flush=True)
    img, threshold = clip_brightest_pixels(img, num_clipped_pixels)
    checkstate()
    img_callback(img)
    print(f"Threshold : {threshold:.4f}.")
    return img

def detect_moon(img, num_edge_pixels, *, checkstate, img_callback):
    # Compute image gradient (edges)
    print(f"Computing edge map...", end=" ", flush=True)
    edge_map = compute_canny_edge_map(img, num_edge_pixels)
    edge_coords = np.column_stack(np.nonzero(edge_map))
    checkstate()
    print(f"Found {len(edge_coords)} edge pixels.")

    # RANSAC fitting
    min_samples = 5
    if len(edge_coords) <= min_samples:
        raise RuntimeError("Not enough edge pixels to fit a circle.")
    cprint("Detecting moon:", style='bold')
    print("RANSAC circle fitting...", end=" ", flush=True)
    (moon_y, moon_x, moon_radius), inliers_coords, outliers_coords = ransac_circle_fit(edge_coords, min_samples)
    checkstate()
    img_callback(make_ransac_img(img, inliers_coords, outliers_coords))
    print(f"Found {len(inliers_coords)} inliers.")
    cprint("Circle parameters :", color='green')
    cprint(f"- Center : ({moon_x:.2f}, {moon_y:.2f})", f"- Radius : {moon_radius:.2f}", color="green", sep="\n")
    
    return np.array([moon_x, moon_y]), moon_radius

def clip_brightest_pixels(img: np.ndarray, num_clipped_pixels: float):
    # Compute threshold
    quantile_threshold = 1 - num_clipped_pixels / img.size
    threshold = np.quantile(img, quantile_threshold)
    img = np.clip(img, 0, threshold) / threshold
    return img, threshold

def compute_canny_edge_map(img: np.ndarray, num_edge_pixels: float):
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
    edge_map = cv2.Canny(sobel_x, sobel_y, threshold, threshold) == 255
    return edge_map

def ransac_circle_fit(coords: np.ndarray, min_samples: int = 5, residual_threshold: float = 1, max_trials: int = 10000):
    '''
    Parameters:
    -min_samples: number of random samples used to estimate the model parameters at each iteration
    -residual_threshold: inliers are such that |sqrt(x**2 + y**2) - r| < threshold. Might depend on pixel scale, but shouldnt really be lower than 1...
    -max_trials: number of RANSAC trials
    '''

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # might throw warnings when the samples are colinear
        model, inliers = sk.measure.ransac(
            coords,
            sk.measure.CircleModel,
            min_samples=min_samples,
            residual_threshold=residual_threshold,
            max_trials=max_trials
        )
    inliers_coords = coords[inliers]
    outliers_coords = coords[~inliers]
    return model.params, inliers_coords, outliers_coords

def make_ransac_img(img: np.ndarray, inliers_coords: np.ndarray, outliers_coords: np.ndarray):
    '''Takes a grayscale image (H,W) as input and returns a color image (H,W,3)
    where green pixels are inliers and red pixels are outliers.'''
    img = img[...,None].repeat(3, axis=-1)
    for i in range(3):
        img[outliers_coords[:,0], outliers_coords[:,1], i] = 1 if i == 0 else 0
        img[inliers_coords[:,0], inliers_coords[:,1], i] = 1 if i == 1 else 0
    return img

def keyword_dict(moon_x, moon_y, moon_radius):
    return {'MOON-X': (moon_x, 'X-coordinate of the moon center.'),
            'MOON-Y': (moon_y, 'Y-coordinate of the moon center.'),
            'MOON-R': (moon_radius, 'Radius of the moon.')}