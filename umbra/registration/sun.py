import numpy as np
import cv2
import skimage as sk

from umbra.common import display, disk, filters, transform
from umbra.common.utils import cprint
from umbra.registration import objective, optim

def preprocess(img, moon_center, moon_radius, sigma_high_pass_tangential, *, img_callback, checkstate):
    '''
    Expects a moon-preprocessed image as input.
    Returns both the image and the center of mass (later used as rotation center)
    '''
    cprint("Preprocessing:", style='bold')
    # Hide moon
    print("Hiding moon...", end=" ", flush=True)
    img, mask = hide_moon(img, moon_center, moon_radius)
    checkstate()
    img_callback(img)
    print("Done.")

    # Preprocess image to register
    print("Applying bandpass filter...", end=" ", flush=True)
    mass_center = compute_mass_center(mask)
    img = apply_bandpass_filter(img, mass_center, sigma_high_pass_tangential)
    checkstate()
    img_callback(display.normalize(img))
    print("Done.")
    return img, mass_center

def get_clipping_value(img, moon_center, moon_radius):
    # Find clipping value that surrounds the 1.05R moon mask
    moon_mask = disk.binary_disk(*moon_center, moon_radius*1.05, img.shape)
    moon_mask_border = sk.morphology.binary_dilation(moon_mask) & ~moon_mask
    clipping_value = np.min(img[moon_mask_border]) # Possible bug : dead pixels
    return clipping_value

def hide_moon(img, moon_center, moon_radius):
    # Mask the moon and its surroundings 
    moon_mask = disk.binary_disk(*moon_center, moon_radius*1.05, img.shape)
    moon_mask_border = sk.morphology.binary_dilation(moon_mask) & ~moon_mask
    clipping_value = np.min(img[moon_mask_border]) # Possible bug : dead pixels
    clipping_mask = img >= clipping_value # should surround the moon_mask
    mask = clipping_mask | moon_mask
    img[mask] = clipping_value
    return img, mask

def compute_mass_center(mask):
    moments = cv2.moments(mask.astype(np.int32), binaryImage=True)
    return (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])

def apply_bandpass_filter(img, center, sigma_high_pass_tangential=10, sigma_low_pass=3):
    # Tangential high-pass filter
    img = img - filters.tangential_filter(img, center, sigma=sigma_high_pass_tangential)
    # Low-pass filter to match the bilinear interpolation smoothing that happens during registration
    img = filters.gaussian_filter(img, sigma=sigma_low_pass)
    # Normalize to [-1,1]
    img /= np.max(np.abs(img))
    return img

def compute_transform(ref_img, img, ref_mass_center, max_iter, error_overlay_opacity, *, img_callback, checkstate):
    '''
    Returns the parameters of the estimated transform "ref_img -> img".
    This transform is parametrized as a rigid transform, where the center of rotation is ref_mass_center.
    '''
    # Display the two images
    checkstate()
    img_callback(red_cyan_blend(ref_img, img, error_overlay_opacity=error_overlay_opacity))

    # Initialize transform parameters
    cprint("Initializing transform:", style='bold')
    print("Computing correlation peak...", end=" ", flush=True)
    tx, ty = objective.correlation_peak(img, ref_img) # translation ref_img -> img
    theta = 0
    print(f"({tx}, {ty})")
    print(f"Using center of mass as center of rotation: ({ref_mass_center[0]:.2f}, {ref_mass_center[1]:.2f})")
    
    # Optimize transform parameters
    cprint("Optimizing transform:", style='bold')
    obj = objective.RigidRegistrationObjective(ref_img, img, ref_mass_center, theta_factor=180/np.pi)
    delta_max = 1
    delta_min = np.array([1e-4, 1e-3, 1e-3]) # want more precision on the angle

    def optim_callback(iter, x, delta, f):
        checkstate()
        # GUI callback
        theta, tx, ty = obj.convert_x_to_params(x)
        tform = transform.centered_rigid_transform(center=ref_mass_center, rotation=theta, translation=(tx,ty))
        img_callback(red_cyan_blend(ref_img, transform.warp(img, tform.inverse.params), error_overlay_opacity=error_overlay_opacity))

        # Display info
        dtheta, dtx, dty = obj.convert_x_to_params(delta) if delta is not None else (None, None, None)
        print(f"Iteration {iter}:")
        print(f"- Rotation       : {np.rad2deg(theta):>9.3f} deg" + (f" ({np.rad2deg(dtheta):+.3f})" if dtheta is not None else "")) 
        print(f"- Translation (x): {tx:>9.2f} px " + (f" ({dtx:+.2f})" if dtx is not None else ""))  
        print(f"- Translation (y): {ty:>9.2f} px " + (f" ({dty:+.2f})" if dty is not None else ""))
        print(f"- Objective value: {f:.3e}")

    x = optim.line_search_newton(obj.convert_params_to_x(theta, tx, ty),
                                 obj.value, obj.grad, obj.hess,
                                 delta_max=delta_max, delta_min=delta_min,
                                 max_iter=max_iter,
                                 callback=optim_callback)
    theta, tx, ty = obj.convert_x_to_params(x)
    return theta, tx, ty

def compute_sun_moon_translation(sun_tform: sk.transform.EuclideanTransform, moon_tform: sk.transform.EuclideanTransform):
    '''
    Compute relative translation of the sun with respect to the moon (from ref to anchor, in ref's coordinate system).
    Resulting translation is such that sun_tform(p) = moon_tform(p + translation), where the tforms are from ref to anchor.
    '''
    # /!\ The coordinate systems have different orientations, and we want the translation in ref's coordinate system.
    # This is given by the difference of the sun/moon transforms "anchor -> ref" (because rotation is applied before translation)
    return - (sun_tform.inverse.translation - moon_tform.inverse.translation) # want ref to anchor, hence flipped sign

def red_cyan_blend(img1, img2, error_overlay_opacity=0.75):
    mean = (img1 + img2)/2
    diff = img1 - img2
    mean *= (1-error_overlay_opacity)
    diff *= error_overlay_opacity

    color_img = np.zeros((*img1.shape, 3), img1.dtype)
    color_img[...,0] = mean + diff
    color_img[...,1] = mean - diff
    color_img[...,2] = mean - diff
    
    color_img -= color_img.min() # in-place is much faster
    color_img /= color_img.max()
    return color_img