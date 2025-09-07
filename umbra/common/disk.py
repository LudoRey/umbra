import numpy as np

def binary_disk(x_c, y_c, radius, shape):
    x, y = np.arange(shape[1]), np.arange(shape[0])
    X, Y = np.meshgrid(x, y)
    binary_disk = np.sqrt((X-x_c)**2 + (Y-y_c)**2) <= radius
    return binary_disk 

def linear_falloff_disk(x_c, y_c, radius, shape, smoothness = 10, return_dist = False):
    x, y = np.arange(shape[1]), np.arange(shape[0])
    X, Y = np.meshgrid(x, y)
    R = np.sqrt((X-x_c)**2 + (Y-y_c)**2)
    if smoothness == 0:
        smooth_disk = (R <= radius).astype('float')
    else:
        smooth_disk = np.clip((radius-R+smoothness)/smoothness, 0, 1)
    if return_dist:
        return smooth_disk, R
    else:
        return smooth_disk