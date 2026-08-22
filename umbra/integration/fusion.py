import numpy as np

from umbra.common import channels, coords, disk


def merge(
    sun_img: np.ndarray,
    moon_img: np.ndarray,
    center: np.ndarray,
    radius: float,
    blend_smoothness: float,
) -> np.ndarray:
    """
    Merge a sun-registered and a moon-registered stack across the moon's limb.

    There are three regimes: the moon image inside the disk, the sun image outside it, and
    the darker of the two over a transition annulus straddling the limb, where the moon image
    may hold corona leaked over the limb and the sun image the pixels the moon rejection
    dropped. The annulus spans radius +/- smoothness, so the minimum takes over completely at
    the limb itself.

    Parameters
    ----------
    sun_img, moon_img : np.ndarray
        Stacks of shape (H, W) or (H, W, C), sun-registered and moon-registered respectively.
    center : np.ndarray
        The (x, y) position of the moon in the moon-registered stack.
    radius : float
        The moon radius, in pixels.
    blend_smoothness : float
        Half-width of the transition annulus, in pixels: the blend reaches this far on
        each side of the limb.

    Returns
    -------
    merged_img : np.ndarray
        The merged stack, of the same shape as the inputs.

    Notes
    ------
    The three weights sum to 1 everywhere, so each handover is continuous even where the two
    stacks disagree: a hard switch would ring wherever the minimum picks the other side.
    """
    region = coords.Region.from_shape(sun_img.shape[:2])
    moon_disk = disk.smooth_disk(center, radius - blend_smoothness, region, blend_smoothness)
    moon_or_min_disk = disk.smooth_disk(center, radius, region, blend_smoothness)
    moon_weights = channels.broadcast_over_channels(moon_disk, sun_img)
    moon_or_min_weights = channels.broadcast_over_channels(moon_or_min_disk, sun_img)
    min_weights = moon_or_min_weights - moon_weights
    return (moon_weights * moon_img
            + min_weights * np.minimum(moon_img, sun_img)
            + (1 - moon_or_min_weights) * sun_img)
