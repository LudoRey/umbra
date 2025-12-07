import numpy as np
from umbra.common import coords

def binary_disk(center: coords.Point | tuple[float, float], radius: float, region: coords.Region | tuple[int, int]) -> np.ndarray:
    """
    Create a binary disk mask.
    
    Parameters:
    -----------
    center : coords.Point or tuple of float
        The coordinates of the disk center. If a tuple is provided, it should be in the form (x, y).
    radius : float
        The radius of the disk.
    region : coords.Region or tuple of int
        The region defining the size of the output mask. If a tuple is provided, it should be in the form (height, width).
    """
    if isinstance(center, tuple):
        center = coords.Point(x=center[0], y=center[1])
    if isinstance(region, tuple):
        region = coords.Region(height=region[0], width=region[1])
    dist_map = distance_map(center, region)
    disk = dist_map <= radius
    return disk
    
def distance_map(center: coords.Point, region: coords.Region):
    """
    Create a distance map from a center point over a specified region.
    
    Parameters:
    -----------
    center : coords.Point
        The coordinates of the center point.
    region : coords.Region
        The region defining the size of the output distance map.
    """
    y = np.arange(region.top, region.top + region.height, dtype=np.float32)
    x = np.arange(region.left, region.left + region.width, dtype=np.float32)
    dist_map = np.sqrt((y[:, None] - center.y) ** 2 + (x[None, :] - center.x) ** 2)
    return dist_map