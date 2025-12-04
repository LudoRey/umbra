import numpy as np

def weighted_average(
    stack: np.ndarray,
    weights: np.ndarray
) -> np.ndarray:
    """
    Compute the weighted average of a stack of images.

    Parameters
    ----------
    stack : np.ndarray
        Array of shape (N, H, W, C) representing the stack of images.
    weights : np.ndarray
        Array of shape (N, H, W, C) representing the weights for each pixel in the stack.

    Returns
    -------
    np.ndarray
        Array of shape (H, W, C) representing the weighted average image.
    """
    weighted_sum = np.sum(stack * weights, axis=0)
    sum_of_weights = np.sum(weights, axis=0)
    return weighted_sum / sum_of_weights