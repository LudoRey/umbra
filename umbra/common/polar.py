import numpy as np

def angle_map(x_c, y_c, shape):
    y = np.arange(shape[0], dtype=np.float32)
    x = np.arange(shape[1], dtype=np.float32)
    # Use broadcasting instead of meshgrid
    theta = np.arctan2(y[:, None] - y_c, x[None, :] - x_c)
    return theta % np.float32(2*np.pi)  # arctan2 spans (-pi, pi], the callers expect [0, 2*pi)

def radius_map(x_c, y_c, shape):
    y = np.arange(shape[0], dtype=np.float32)
    x = np.arange(shape[1], dtype=np.float32)
    # Use broadcasting instead of meshgrid
    rho = np.sqrt((y[:, None] - y_c) ** 2 + (x[None, :] - x_c) ** 2)
    return rho