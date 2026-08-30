import cv2
import numpy as np


def estimate_depth(image, horizon=0.45, blur=51):
    """
    Approximate scene depth from a single road image.

    Returns
    -------
    depth : float32 array (H,W)
        0 = close
        1 = far
    """

    h, w = image.shape[:2]

    # Vertical depth prior
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)
    depth = np.tile(y[:, None], (1, w))

    # Horizon adjustment
    depth = np.clip((depth - horizon) / (1.0 - horizon), 0, 1)

    # Road widening prior
    x = np.linspace(-1, 1, w, dtype=np.float32)
    x = np.abs(x)
    road_prior = 1.0 - (x ** 2)

    depth *= (0.6 + 0.4 * road_prior)

    # Smooth map
    depth = cv2.GaussianBlur(depth, (blur, blur), 0)

    # Normalize
    depth -= depth.min()
    depth /= (depth.max() + 1e-8)

    return depth.astype(np.float32)


def transmission(depth, strength):
    """
    Atmospheric transmission map.
    Larger strength = heavier weather.
    """

    return np.exp(-strength * depth)


def blend_with_airlight(image, depth, strength, airlight=(230, 230, 230)):
    """
    Blend image with atmospheric light using estimated depth.
    Used by fog, haze, smoke and dust.
    """

    img = image.astype(np.float32)

    t = transmission(depth, strength)
    t = np.repeat(t[:, :, None], 3, axis=2)

    A = np.array(airlight, dtype=np.float32)

    result = img * t + A * (1.0 - t)

    return np.clip(result, 0, 255).astype(np.uint8)