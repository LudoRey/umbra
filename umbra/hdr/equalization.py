"""Fitting one stack onto the brightness scale of another, as a function of the angle around the moon."""
import numpy as np

# The corona of the merged stacks comes from the sun-registered images, in which the moon drifts
# over totality. Inflating its radius by this much makes a disk centred on the moon at one moment
# cover it at every other, so the fit never sees a pixel the moon passed over.
MOON_RADIUS_FACTOR = 1.15

# Sampling walks a dense (angle, radius) grid; this many angles at a time keeps it a few megabytes.
ANGLES_PER_CHUNK = 1000

def evaluate_trigonometric_basis(theta, degree):
    out = [np.ones(theta.shape[0])]
    for n in range(1, degree+1):
        out.append(np.cos(n*theta))
        out.append(np.sin(n*theta))
    out = np.stack(out, axis=1) 
    return out

def sample_polar(mask, center, num_samples):
    """
    Sample pixels uniformly in polar coordinates around ``center``.

    Each sample draws an angle uniformly over the full turn, then a radius uniformly among the
    radii whose pixel lies inside ``mask``. Uneven illumination keeps that mask from being a
    clean annulus -- the thresholds bite at a different radius in every direction -- so drawing
    uniformly among its pixels would hand the most weight to the directions where it happens to
    be widest. Drawing the angle first weighs every direction the same, whatever shape the mask
    takes there.

    Parameters
    ----------
    mask : ndarray of bool, shape (height, width)
        The pixels a sample may land on.
    center : ndarray, shape (2,)
        Centre of the polar coordinates, as ``(x, y)``.
    num_samples : int
        Number of pixels to draw.

    Returns
    -------
    rows, cols : ndarray of int, shape (num_samples,)
        Indices of the sampled pixels. The same pixel may be drawn more than once, all the more
        near the centre, where a single pixel covers a wide span of angles.

    Raises
    ------
    ValueError
        If a drawn angle has no valid pixel anywhere along its ray.
    """
    height, width = mask.shape
    corners = np.array([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
    max_radius = np.hypot(*(corners - center).T).max()
    radius = np.arange(int(max_radius) + 1)
    theta = np.random.uniform(0, 2*np.pi, num_samples)

    rows, cols = np.empty(num_samples, dtype=np.intp), np.empty(num_samples, dtype=np.intp)
    for start in range(0, num_samples, ANGLES_PER_CHUNK):
        chunk_theta = theta[start:start+ANGLES_PER_CHUNK, None]
        x = np.rint(center[0] + radius*np.cos(chunk_theta)).astype(np.intp)
        y = np.rint(center[1] + radius*np.sin(chunk_theta)).astype(np.intp)
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        valid = inside & mask[np.where(inside, y, 0), np.where(inside, x, 0)]

        # Picking the largest of one random key per valid radius draws one of them uniformly.
        keys = np.where(valid, np.random.random(valid.shape), -1.0)
        picked = keys.argmax(axis=1)
        chunk_indices = np.arange(len(picked))
        empty_ray = keys[chunk_indices, picked] < 0
        if empty_ray.any():
            raise ValueError(
                f"The ray at {np.rad2deg(chunk_theta[empty_ray.argmax(), 0]):.0f} deg holds no sample to fit "
                "the brightness on. Widen the gap between the clipping thresholds.")
        rows[start:start+len(picked)] = y[chunk_indices, picked]
        cols[start:start+len(picked)] = x[chunk_indices, picked]
    return rows, cols

def linear_trigo_fit(x, theta, y, degree):
    # Construct trigonometric polynomial features (example of a term : x*sin(theta))
    trigo_basis = evaluate_trigonometric_basis(theta, degree)
    X = np.concatenate([trigo_basis, trigo_basis*x[:,None]], axis=1)

    # Add penalty to high order terms
    # num_features = 2*(2*degree+1)
    # penalty_matrix = reg_factor*np.tile(np.arange(2*degree+1), 2)*np.eye(num_features)
    # X = np.concatenate([X, penalty_matrix], axis=0)
    # y = np.concatenate([y, np.zeros(num_features)], axis=0)

    coeffs = np.linalg.lstsq(X, y, rcond=None)[0] # X @ coeffs ~= y

    offset_trigo_coeffs = coeffs[:1+2*degree]
    slope_trigo_coeffs = coeffs[1+2*degree:]

    return offset_trigo_coeffs, slope_trigo_coeffs

def evaluate_trigonometric_polynomial(theta, coeffs, degree, num_samples=10000):
    """
    Evaluate a trigonometric polynomial at arbitrary angles, using a LUT for better performance.

    The polynomial is collapsed against ``coeffs`` on a uniform grid of ``num_samples`` angles and
    the curve is then indexed, rather than evaluating the basis at every angle.

    Returns an array shaped like ``theta``.
    """
    grid = np.linspace(0, 2*np.pi, num_samples)
    curve = evaluate_trigonometric_basis(grid, degree) @ coeffs
    return curve[np.rint(theta * ((num_samples - 1) / (2*np.pi))).astype(np.int32)]

def equalize_brightness(img_x, img_theta, img_y, mask, center, degree=4, num_samples=1000):
    rows, cols = sample_polar(mask, center, num_samples)
    sample_x, sample_y = img_x[rows, cols].mean(axis=1), img_y[rows, cols].mean(axis=1)

    # Fit every angle at once using trigonometric interactions
    offset_trigo_coeffs, slope_trigo_coeffs = linear_trigo_fit(sample_x, img_theta[rows, cols], sample_y, degree)

    img_offset = evaluate_trigonometric_polynomial(img_theta, offset_trigo_coeffs, degree)
    img_slope = evaluate_trigonometric_polynomial(img_theta, slope_trigo_coeffs, degree)
    return img_offset[:,:,None] + img_slope[:,:,None]*img_x