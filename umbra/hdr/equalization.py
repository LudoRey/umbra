"""Fitting one stack onto the brightness scale of another, as a function of the angle around the moon."""
import numpy as np

# The corona of the merged stacks comes from the sun-registered images, in which the moon drifts
# over totality. Inflating its radius by this much makes a disk centred on the moon at one moment
# cover it at every other, so the fit never sees a pixel the moon passed over.
MOON_RADIUS_FACTOR = 1.15

def evaluate_trigonometric_basis(theta, degree):
    out = [np.ones(theta.shape[0])]
    for n in range(1, degree+1):
        out.append(np.cos(n*theta))
        out.append(np.sin(n*theta))
    out = np.stack(out, axis=1) 
    return out

def resample_per_sector(theta, num_sectors, num_samples_per_sector):
    resampled_indices_per_sector = np.zeros([num_sectors, num_samples_per_sector], dtype=np.uint64)
    for sector_idx in range(num_sectors):
        # Define sector mask
        theta_min, theta_max = 2*np.pi*sector_idx/num_sectors, 2*np.pi*(sector_idx+1)/num_sectors
        sector_mask = (theta >= theta_min)*(theta < theta_max)
        # Resample sector indices
        sector_indices = np.nonzero(sector_mask)[0]
        if len(sector_indices) == 0:
            raise ValueError(
                f"Sector {sector_idx+1}/{num_sectors} ({np.rad2deg(theta_min):.0f}-{np.rad2deg(theta_max):.0f} deg) "
                "holds no sample to fit the brightness on. Widen the gap between the clipping thresholds, "
                "or reduce the extra radius excluded around the moon.")
        quotient, remainder = np.divmod(num_samples_per_sector, len(sector_indices))
        resampled_indices = np.concatenate([np.tile(sector_indices, quotient), np.random.choice(sector_indices, size=remainder, replace=False)])
        resampled_indices_per_sector[sector_idx] = resampled_indices
    return resampled_indices_per_sector

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

def equalize_brightness(img_x, img_theta, img_y, mask, degree=4, num_sectors=30, num_samples_per_sector=400, return_coeffs=False):
    valid_x = img_x.mean(axis=2)[mask]
    valid_y = img_y.mean(axis=2)[mask]
    valid_theta = img_theta[mask]

    # Select fixed amount of samples per sector (so that we sample ~uniformly over theta)
    resampled_indices_per_sector = resample_per_sector(valid_theta, num_sectors, num_samples_per_sector)
    samples_per_sector_x, samples_per_sector_theta, samples_per_sector_y = valid_x[resampled_indices_per_sector], valid_theta[resampled_indices_per_sector], valid_y[resampled_indices_per_sector]

    # Fit all sectors at once using trigonometric interactions
    offset_trigo_coeffs, slope_trigo_coeffs = linear_trigo_fit(samples_per_sector_x.reshape(-1), 
                                                                samples_per_sector_theta.reshape(-1), 
                                                                samples_per_sector_y.reshape(-1), 
                                                                degree)


    img_offset = evaluate_trigonometric_polynomial(img_theta, offset_trigo_coeffs, degree)
    img_slope = evaluate_trigonometric_polynomial(img_theta, slope_trigo_coeffs, degree)
    img_fitted_x = img_offset[:,:,None] + img_slope[:,:,None]*img_x
    if return_coeffs:
        return img_fitted_x, img_offset, img_slope 
    else:
        return img_fitted_x