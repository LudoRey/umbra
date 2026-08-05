import numpy as np
from umbra.common import coords

def binary_disk(center: np.ndarray, radius: float, region: coords.Region) -> np.ndarray:
    """Create a binary disk mask over a specified region."""
    dist_map = distance_map(center, region)
    disk = dist_map <= radius
    return disk
    
def smooth_disk(center: np.ndarray, radius: float, region: coords.Region, smoothness: float) -> np.ndarray:
    """
    Create a feathered disk mask over a specified region.

    The mask is 1 inside ``radius`` and feathers linearly down to 0 over the next
    ``smoothness`` pixels. A smoothness of 0 gives the binary disk.
    """
    if smoothness == 0:
        return binary_disk(center, radius, region).astype(np.float32)
    dist_map = distance_map(center, region)
    return np.clip((radius + smoothness - dist_map) / smoothness, 0, 1)

def distance_map(center: np.ndarray, region: coords.Region) -> np.ndarray:
    """Create a distance map from a center point over a specified region."""
    y = np.arange(region.top, region.top + region.height, dtype=np.float32)
    x = np.arange(region.left, region.left + region.width, dtype=np.float32)
    center = center.astype(np.float32)
    dist_map = np.sqrt((y[:, None] - center[1]) ** 2 + (x[None, :] - center[0]) ** 2)
    return dist_map
