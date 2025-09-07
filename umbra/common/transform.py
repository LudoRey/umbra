import skimage as sk
import numpy as np
import cv2

border_mode_dict = {
    'constant': cv2.BORDER_CONSTANT,
    'edge': cv2.BORDER_REPLICATE,
    'reflect': cv2.BORDER_REFLECT,
    'wrap': cv2.BORDER_WRAP,
}
interp_mode_dict = {
    'nearest': cv2.INTER_NEAREST,
    'linear': cv2.INTER_LINEAR,
    'cubic': cv2.INTER_CUBIC,
    'lanczos4': cv2.INTER_LANCZOS4
}

def centered_rigid_transform(center, rotation, translation):
    '''
    Rotate first around center, then translate.
    Rotation is the *counterclockwise* angle in radians (but remember that the y-axis is upside down in images!)
    '''
    t_uncenter = sk.transform.EuclideanTransform(translation=center)
    t_center = t_uncenter.inverse
    t_rotate = sk.transform.EuclideanTransform(rotation=rotation)
    t_translate = sk.transform.EuclideanTransform(translation=translation)
    return t_center + t_rotate + t_uncenter + t_translate

def translation_transform(translation):
    return sk.transform.EuclideanTransform(rotation=0, translation=translation)

def warp(img: np.ndarray,
         matrix: np.ndarray,
         output_shape = None,
         interp_mode = 'linear',
         border_mode = 'constant',
         border_value = 0):
    
    if output_shape is None:
        output_shape = img.shape[0:2]

    if np.array_equal(matrix[2], [0,0,1]):
        _warp = cv2.warpAffine # faster
        matrix = matrix[:2] 
    else:
        _warp = cv2.warpPerspective

    return _warp(img, matrix, (output_shape[1], output_shape[0]),
                 flags=interp_mode_dict[interp_mode],
                 borderMode=border_mode_dict[border_mode],
                 borderValue=border_value)

def warp_cart_to_polar(img: np.ndarray,
                       center: tuple = None,
                       output_shape: tuple = None,
                       interp_mode = 'linear',
                       log_scaling = False):
    # Set defaults
    input_shape = img.shape
    if not output_shape:
        output_shape = input_shape
    if not center:
        center = (input_shape[1] // 2, input_shape[0] // 2)
    # Find maximum radius coordinate in input image
    corner_pts = np.array([[0, 0],[0, input_shape[1]],[input_shape[0], 0], [input_shape[0], input_shape[1]]])
    corner_dist = np.sqrt(np.sum((corner_pts - np.array([center[1], center[0]]))**2, axis=1))
    max_radius = corner_dist.max()+1
    # Set flags
    flags = interp_mode_dict[interp_mode]
    if log_scaling:
        flags = flags | cv2.WARP_POLAR_LOG
    flags = flags | cv2.WARP_FILL_OUTLIERS
    # Warp image
    img = cv2.warpPolar(img, (output_shape[1], output_shape[0]), center, max_radius, flags)
    # Border mode
    # warpPolar does not support borderMode, but we want to replicate the edge instead of filling with zeros
    outliers = cv2.warpPolar(np.full(input_shape, 255, np.uint8), (output_shape[1], output_shape[0]), center, max_radius, flags) != 255
    col_indices = np.argmax(outliers, axis=1)
    row_indices = np.arange(outliers.shape[0])
    for row_index, col_index in zip(row_indices, col_indices):
        if col_index != 0:
            img[row_index, col_index:] = img[row_index, col_index-1]
    return img
    
def warp_polar_to_cart(img: np.ndarray,
                       center: tuple = None,
                       output_shape: tuple = None,
                       interp_mode = 'linear',
                       log_scaling = False):
    # Set defaults
    if not output_shape:
        output_shape = img.shape
    if not center:
        center = (output_shape[1] // 2, output_shape[0] // 2)
    # Find maximum radius coordinate in output image
    corner_pts = np.array([[0, 0],[0, output_shape[1]],[output_shape[0], 0], [output_shape[0], output_shape[1]]])
    corner_dist = np.sqrt(np.sum((corner_pts - np.array([center[1], center[0]]))**2, axis=1))
    max_radius = corner_dist.max()+1
    # Set flags
    flags = interp_mode_dict[interp_mode]
    if log_scaling:
        flags = flags | cv2.WARP_POLAR_LOG
    flags = flags | cv2.WARP_INVERSE_MAP
    return cv2.warpPolar(img, (output_shape[1], output_shape[0]), center, max_radius, flags)

## Much slower skimage version

# def coords_cart_to_polar(cart_coords, x_c, y_c, theta_factor, rho_factor, log_scaling=False):
#     '''
#     To be used as the inverse_map argument of the warp function.
#     The input cart_coords is a (M, 2) array where each row contains (x, y) coordinates.
#     Returns a (M, 2) array where each row contains scaled (rho, theta) coordinates.
#     '''
#     x, y = cart_coords[:, 0], cart_coords[:, 1]
#     rho = np.sqrt((x-x_c)**2 + (y-y_c)**2)
#     if log_scaling:
#         rho = np.log(1 + rho)
#     with np.errstate(divide='ignore',invalid='ignore'):
#         theta = np.arctan((y-y_c)/(x-x_c)) + np.pi * (x-x_c < 0) + 2*np.pi * (x-x_c >= 0)*(y-y_c < 0) # tan is pi-periodic : arctan can be many things
#         theta[np.isnan(theta)] = 0
#     polar_coords = np.column_stack((rho*rho_factor, theta*theta_factor))
#     return polar_coords

# def coords_polar_to_cart(polar_coords, x_c, y_c, theta_factor, rho_factor, log_scaling=False):
#     '''
#     To be used as the inverse_map argument of the warp function.
#     The input polar_coords is a (M, 2) array where each row contains scaled (rho, theta) coordinates.
#     Returns a (M, 2) array where each row contains (x, y) coordinates.
#     '''
#     rho, theta = polar_coords[:, 0] / rho_factor, polar_coords[:, 1] / theta_factor
#     if log_scaling:
#         rho = np.exp(rho) - 1
#     x = rho * np.cos(theta) + x_c
#     y = rho * np.sin(theta) + y_c
#     cart_coords = np.column_stack((x, y))
#     return cart_coords

# def warp_cart_to_polar(img, x_c, y_c, output_shape, return_factors=False, log_scaling=False, **kwargs):
#     # Compute maximum/minimum distance to the center
#     corner_pts = np.array([[0, 0],[0, img.shape[1]],[img.shape[0], 0], [img.shape[0], img.shape[1]]])
#     corner_dist = np.sqrt(np.sum((corner_pts - np.array([y_c, x_c]))**2, axis=1)) # (4) array
#     max_rho = corner_dist.max()
#     if log_scaling:
#         max_rho = np.log(1 + max_rho)
#     # Define scaling factors
#     theta_factor = (output_shape[0] - 1) / (2 * np.pi)
#     rho_factor = (output_shape[1] - 1) / max_rho
#     # Warp image
#     warp_args = {'x_c': x_c, 'y_c': y_c, 'theta_factor': theta_factor, 'rho_factor': rho_factor, 'log_scaling': log_scaling}
#     warped_img = warp(img, inverse_map=coords_polar_to_cart, map_args=warp_args, output_shape=output_shape, **kwargs)
#     if return_factors:
#         return warped_img, theta_factor, rho_factor 
#     else:
#         return warped_img

# def warp_polar_to_cart(img, x_c, y_c, output_shape, log_scaling=False, **kwargs):
#     # Compute maximum distance to the center
#     corner_pts = np.array([[0, 0],[0, output_shape[1]],[output_shape[0], 0], [output_shape[0], output_shape[1]]])
#     corner_dist = np.sqrt(np.sum((corner_pts - np.array([y_c, x_c]))**2, axis=1))
#     max_rho = corner_dist.max()
#     if log_scaling:
#         max_rho = np.log(1 + max_rho)
#     # Define scaling factors
#     theta_factor = (img.shape[0] - 1) / (2 * np.pi)
#     rho_factor = (img.shape[1] - 1) / max_rho
#     # Warp image
#     warp_args = {'x_c': x_c, 'y_c': y_c, 'theta_factor': theta_factor, 'rho_factor': rho_factor, 'log_scaling': log_scaling}
#     warped_img = warp(img, inverse_map=coords_cart_to_polar, map_args=warp_args, output_shape=output_shape, mode='wrap', **kwargs) # wrap padding for 0 = 2pi
#     return warped_img