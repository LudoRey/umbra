"""Fitting one stack onto the brightness scale of another, as a function of the angle around the moon."""
import numpy as np

# The moon drifts over totality in the sun-registered images the stacks come from. Inflating its
# radius by this much covers every position it took, so the fit never sees a pixel it passed over.
MOON_RADIUS_FACTOR = 1.15

# Sampling walks a dense (angle, radius) grid; this many angles at a time keeps it a few megabytes.
ANGLES_PER_CHUNK = 1000

def evaluate_trigonometric_basis(theta: np.ndarray, degree: int) -> np.ndarray:
    """The basis 1, cos(n*theta), sin(n*theta) up to ``degree``, one row per angle."""
    out = [np.ones(theta.shape[0])]
    for n in range(1, degree+1):
        out.append(np.cos(n*theta))
        out.append(np.sin(n*theta))
    return np.stack(out, axis=1)

def sample_polar(mask: np.ndarray, center: np.ndarray, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw pixels uniformly in polar coordinates around ``center``, as row and column indices.

    Each sample takes an angle uniformly over the full turn, then a radius uniformly among the
    radii whose pixel lies inside ``mask``. Uneven illumination keeps that mask from being a clean
    annulus, so drawing uniformly among its pixels would favour the directions where it is widest.
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
                "the brightness on. Widen the gap between the weighting thresholds.")
        rows[start:start+len(picked)] = y[chunk_indices, picked]
        cols[start:start+len(picked)] = x[chunk_indices, picked]
    return rows, cols

def linear_trigo_fit(x: np.ndarray, theta: np.ndarray, y: np.ndarray, degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares fit of ``y ~ offset(theta) + slope(theta)*x``, returning the coefficients of both."""
    # One feature per basis term, then the same terms interacted with x (a term is e.g. x*sin(theta)).
    trigo_basis = evaluate_trigonometric_basis(theta, degree)
    X = np.concatenate([trigo_basis, trigo_basis*x[:,None]], axis=1)

    coeffs = np.linalg.lstsq(X, y, rcond=None)[0] # X @ coeffs ~= y
    return coeffs[:1+2*degree], coeffs[1+2*degree:]

def evaluate_trigonometric_polynomial(theta: np.ndarray, coeffs: np.ndarray, degree: int,
                                      num_samples: int = 10000) -> np.ndarray:
    """
    Evaluate a trigonometric polynomial at every angle of ``theta``, which it returns the shape of.

    The curve is collapsed against ``coeffs`` on a uniform grid of ``num_samples`` angles and then
    indexed, rather than evaluating the basis at every angle.
    """
    grid = np.linspace(0, 2*np.pi, num_samples)
    curve = evaluate_trigonometric_basis(grid, degree) @ coeffs
    return curve[np.rint(theta * ((num_samples - 1) / (2*np.pi))).astype(np.int32)]

def equalize_brightness(img_x: np.ndarray, img_theta: np.ndarray, img_y: np.ndarray, mask: np.ndarray,
                        center: np.ndarray, degree: int = 4, num_samples: int = 1000) -> np.ndarray:
    """
    Rescale ``img_x`` onto the brightness of ``img_y``, through an affine map varying with the angle.

    The map is fitted on pixels drawn from ``mask``, which holds those both stacks record faithfully.
    """
    rows, cols = sample_polar(mask, center, num_samples)
    sample_x, sample_y = img_x[rows, cols].mean(axis=1), img_y[rows, cols].mean(axis=1)

    # Every angle is fitted at once, through the interactions of x with the trigonometric basis.
    offset_trigo_coeffs, slope_trigo_coeffs = linear_trigo_fit(sample_x, img_theta[rows, cols], sample_y, degree)

    img_offset = evaluate_trigonometric_polynomial(img_theta, offset_trigo_coeffs, degree)
    img_slope = evaluate_trigonometric_polynomial(img_theta, slope_trigo_coeffs, degree)
    return img_offset[:,:,None] + img_slope[:,:,None]*img_x