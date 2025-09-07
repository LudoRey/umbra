import numpy as np

def angle_map(x_c, y_c, shape):
    x, y = np.arange(shape[1]), np.arange(shape[0])
    X, Y = np.meshgrid(x, y)
    with np.errstate(divide='ignore',invalid='ignore'):
        theta = np.arctan((Y-y_c)/(X-x_c)) + np.pi * (X-x_c < 0) + 2*np.pi * (X-x_c >= 0)*(Y-y_c < 0)  # tan is pi-periodic : arctan can be many things
        theta[np.isnan(theta)] = 0 # if x_c and y_c are integers, we get arctan(0/0) = NaN at the center
    return theta

def radius_map(x_c, y_c, shape):
    x, y = np.arange(shape[1]), np.arange(shape[0])
    X, Y = np.meshgrid(x, y)
    rho = np.sqrt((X-x_c)**2 + (Y-y_c)**2)
    return rho