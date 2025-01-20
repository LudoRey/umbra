import numpy as np

from skimage.feature import canny
from skimage import measure, transform

from astropy.coordinates import EarthLocation, get_body
from astropy.time import Time

from .disk import binary_disk
from .filters import tangential_filter, gaussian_filter

from typing import Callable

def centered_rigid_transform(center, rotation, translation):
    '''Rotate first around center, then translate'''
    t_uncenter = transform.AffineTransform(translation=center)
    t_center = t_uncenter.inverse
    t_rotate = transform.AffineTransform(rotation=rotation)
    t_translate = transform.AffineTransform(translation=translation)
    return t_center + t_rotate + t_uncenter + t_translate

def rotate_translate(img, theta, dx, dy):
    print(f"Rotating image by :")
    print(f"    - theta = {theta:.4f}")
    print(f"Translating image by :")
    print(f"    - dx = {dx:.2f}")
    print(f"    - dy = {dy:.2f}")
    tform = centered_rigid_transform(center=(img.shape[1]/2, img.shape[0]/2), rotation=theta, translation=(dx, dy))
    registered_img = transform.warp(img, tform.inverse)
    return registered_img

### Moon detection

def moon_detection(img, moon_radius_pixels):
    # Preparing image for detection
    if len(img.shape) == 3:
        # Convert to grayscale
        img = img.mean(axis=2)
    # Rescale and clip pixels to make moon border more defined
    # Because of brightness variations, the margin should be large enough so that a complete annulus is clipped
    clip_annulus_outer_moon_radii = 1.3 # outer radius (in moon radii) of the annulus
    clip_annulus_area_pixels = np.minimum(img.size, np.pi*(clip_annulus_outer_moon_radii*moon_radius_pixels)**2) - np.pi*moon_radius_pixels**2
    # Compute clipping value
    hist, bin_edges = np.histogram(img, bins=10000)
    cumhist = np.cumsum(hist)
    clip_idx = np.nonzero(cumhist > img.size - clip_annulus_area_pixels)[0][0]
    clip_value = bin_edges[clip_idx]
    num_clipped_pixels = img.size - cumhist[clip_idx-1]
    print(f"Rescaling and clipping pixels above {clip_value:.3f} (clipped {num_clipped_pixels} pixels)")
    img = np.clip(img, 0, clip_value)
    img /= clip_value
    
    # Canny
    print(f"Canny edge detection...")
    # Find the (appproximate) number of pixels that correspond to the moon circonference (M)
    # Canny will use a threshold that retains the M brightest pixels in the gradient image (before NMS, so there might be less of them after NMS)
    moon_circonference_pixels = 2*np.pi*moon_radius_pixels 
    moon_circonference_fraction = moon_circonference_pixels / img.size

    threshold = 1-moon_circonference_fraction # Single threshold : no hysteresis

    edges = canny(img, sigma=1, low_threshold=threshold, high_threshold=threshold, use_quantiles=True)
    print(f"Found {np.count_nonzero(edges)} edge pixels.")

    # RANSAC
    print("RANSAC fitting...")
    min_samples = 20 # Number of random samples used to estimate the model parameters at each iteration
    residual_threshold = 1 # Inliers are such that |sqrt(x**2 + y**2) - r| < threshold. Might depend on pixel scale, but shouldnt really be lower than 1...
    max_trials = 100 # Number of RANSAC trials 
    edges_coords = np.column_stack(np.nonzero(edges))
    model, inliers = measure.ransac(edges_coords, measure.CircleModel,
                                    min_samples=min_samples, residual_threshold=residual_threshold, max_trials=max_trials)

    y_c, x_c, radius = model.params
    print(f"Found {inliers.sum()} inliers.")
    print(f"Model parameters : ")
    print(f"    - x_c : {x_c:.2f}")
    print(f"    - y_c : {y_c:.2f}")
    print(f"    - radius : {radius:.2f}")

    return x_c, y_c, radius

### Sun registration

def get_moon_clipping_value(img, header):
    # Convert to grayscale
    if len(img.shape) == 3:
        img = img.min(axis=2)
    # Find clipping value that surrounds the 1.05R moon masks
    # The moon moves by less than 0.1R (~0.05R) during the eclipse : hence all moon masks will be contained by ext_moon_mask
    moon_mask = binary_disk(header["MOON-X"], header["MOON-Y"], header["MOON-R"]*1.05, img.shape) 
    ext_moon_mask = binary_disk(header["MOON-X"], header["MOON-Y"], header["MOON-R"]*1.15, img.shape) 
    moon_mask_border = ext_moon_mask & ~moon_mask
    clipping_value = np.min(img[moon_mask_border])
    return clipping_value

def prep_for_registration(img, header, clipping_value):
    print("Preparing image for registration...")
    # Convert to grayscale
    if len(img.shape) == 3:
        img = img.mean(axis=2)
    # Clip the moon and its surroundings
    moon_mask = binary_disk(header["MOON-X"], header["MOON-Y"], header["MOON-R"]*1.05, img.shape)
    clipping_mask = img >= clipping_value # should surround the moon_mask
    mask = clipping_mask | moon_mask
    img[mask] = clipping_value
    # High-pass filter
    img = img - tangential_filter(img, header["MOON-X"], header["MOON-Y"], sigma=10)
    # Low pass filter (attenuate interpolation effects during registration)
    img = gaussian_filter(img, sigma=2)
    # Normalize
    img /= img.std()
    return img # np.stack([mask, moon_mask, ext_moon_mask], axis=2, dtype=float)

def correlation(img1, img2):
    img1 = np.fft.fft2(img1) # Spectrum of image 1
    img2 = np.fft.fft2(img2) # Spectrum of image 2
    img = (img1 * np.conj(img2)) # Correlation spectrum
    img = np.real(np.fft.ifft2(img)) # Correlation image
    return img

class DiscreteRigidRegistrationObjective:
    def __init__(self, ref_img, img):
        # Precompute constants
        self.ref_img = ref_img 
        self.img = img
        # Cache
        self.x = None
        self.value_at_x = None
    
    def value(self, x):
        # Cached computation
        if not np.array_equal(x, self.x) or self.value_at_x is None:
            theta, tx, ty = self.convert_x_to_params(x)
            h, w = self.img.shape[0:2]
            inv_transform = centered_rigid_transform(center=(w/2,h/2), rotation=theta, translation=(tx, ty))
            registered_img = transform.warp(self.img, inv_transform) # Need to ensure that zero-padding is fine here !
            # Compute objective value and update cache
            self.value_at_x = 1/2*np.mean((registered_img - self.ref_img)**2)
        return self.value_at_x
    
    def grad(self, x, perturbation=0.01):
        # Forward difference
        value_at_x = self.value(x)
        objective_grad = np.zeros(3)
        for i in range(3):
            perturbed_x = x.copy()
            perturbed_x[i] += perturbation
            objective_grad[i] = (self.value(perturbed_x) - value_at_x) / perturbation
        return objective_grad
    
    def convert_x_to_params(self, x):
        theta, tx, ty = x[0]/1800 * np.pi, x[1], x[2] # parameters of registered_img -> img (we call it the *inverse* transform)
        return theta, tx, ty
    
    def convert_params_to_x(self, theta, tx, ty):
        x = np.array([theta/np.pi * 1800, tx, ty])
        return x

    # def hess(self, x, perturbation=0.01):
    #     value_at_x = self.value(x)
    #     objective_hess = np.zeros((3,3))
    #     for i in range(3):
    #         for j in range(3):
    #             if i <= j:
    #                 perturbed_x = x.copy()
    #                 perturbed_x[i] += perturbation
    #                 perturbed_x[j] += perturbation
    #                 objective_hess[i,j] = (self.value(perturbed_x) - value_at_x) / perturbation**2
    #                 objective_hess[j,i] = objective_hess[i,j]
    #     return objective_hess

def line_search_gradient_descent(x0: np.ndarray, func: Callable, grad: Callable, c=0.5, delta_initial=0.1, delta_final=1e-4):
    '''
    Gradient descent with Armijo line search. Uses an adaptive alpha_max scheme. 
    An important variable here describes how much we move : delta(alpha) := max|alpha*grad(x)|

    Parameters
    ----------
    - x0 : initial guess
    - func, grad : callables that return the function value and its gradient respectively.
    - c : positive value to ensure sufficient decrease
    - delta_initial : propose to initially move by delta_initial, i.e. in the first iteration, we set alpha_max s.t. delta(alpha_max) = delta_initial
    - delta_final : stop the optimization loop when we move by delta_final 
    '''
    stopping_flag = False
    x = x0
    iter = 0
    f = func(x0)
    g = grad(x0)
    # Determines initial alpha_max
    alpha_max = delta_initial/np.max(np.abs(g))
    while not stopping_flag:
        # Armijo line search
        alpha = alpha_max
        while not (f_next:= func(x - alpha*g)) <= f - c*alpha*np.dot(g, g):
            alpha /= 2
            if alpha <= 1e-10: # incorrect gradient (or magnitude is way too high)
                alpha = 0 # will end the linesearch and trigger the stopping criterion
        # Stopping criterion : if not moving by much and could not move more
        if alpha*np.max(np.abs(g)) <= delta_final and alpha != alpha_max:
            stopping_flag = True
        # Accept step
        x = x - alpha*g
        f = f_next
        g = grad(x)
        # Update alpha max (double or halve)
        if alpha == alpha_max:
            alpha_max *= 2
        else:
            alpha_max /= 2
        # Display info
        print(f"Iteration {iter}:")
        print(f"Value : {f:.3e}")
        print(f"x : {x}")
        print(f"Gradient : {g}")
        print(f"alpha : {alpha} \n")
        iter += 1
    return x

# Old astrometry approach
def get_moon_radius(time: Time, location: EarthLocation):
    moon_coords = get_body("moon", time, location)
    earth_coords = get_body("earth", time, location)
    moon_dist_km = earth_coords.separation_3d(moon_coords).km
    moon_real_radius_km = 1737.4
    moon_radius_degree = np.arctan(moon_real_radius_km / moon_dist_km) * 180 / np.pi
    return moon_radius_degree

def get_sun_moon_offset(time: Time, location: EarthLocation):
    # Get the moon and sun coordinates at the specified time and location
    moon_coords = get_body("moon", time, location)
    sun_coords = get_body("sun", time, location)
    # Compute the offset
    sun_offset_scalar = moon_coords.separation(sun_coords).arcsecond
    sun_offset_angle = moon_coords.position_angle(sun_coords).degree
    return sun_offset_scalar, sun_offset_angle

def convert_angular_offset_to_x_y(offset_scalar, offset_angle, camera_rotation, image_scale):
    '''
    offset_scalar is given in arcseconds
    offset_angle and camera_rotation are given in degrees
    image_scale is given in arcseconds/pixel
    '''
    # 1) offset is offset_angle degrees east of north
    # 2) up (-y) is camera_rotation degrees east of north
    # Combining 1) and 2), offset is offset_angle + (360 - camera_rotation) degrees counterclockwise of up
    # Modulo 360, offset is offset_angle - camera_rotation + 90 degrees counterclockwise of x
    offset_angle_to_x = (offset_angle - camera_rotation + 90) * np.pi / 180 # counterclockwise
    offset_x = np.cos(offset_angle_to_x)*offset_scalar / image_scale
    offset_y = -np.sin(offset_angle_to_x)*offset_scalar / image_scale
    return offset_x, offset_y