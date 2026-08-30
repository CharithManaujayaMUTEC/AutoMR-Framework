import cv2
import numpy as np

from .depth import estimate_depth


# ==========================================================
# RANDOM NUMBER GENERATOR
# ==========================================================

_rng = np.random.default_rng()


# ==========================================================
# RAIN
# ==========================================================

def generate_rain_layer(
    image,
    intensity=0.5,
    drop_length=25,
    angle=-15,
):
    """
    Generate a rain layer whose density increases with depth.
    """

    h, w = image.shape[:2]

    depth = estimate_depth(image)

    layer = np.zeros((h, w), dtype=np.float32)

    max_drops = int(h * w * 0.0008 * intensity)

    ys = _rng.integers(0, h, max_drops)
    xs = _rng.integers(0, w, max_drops)

    for x, y in zip(xs, ys):

        # Far regions receive more rain
        d = depth[y, x]

        if _rng.random() > d:
            continue

        length = int(drop_length * (0.4 + 0.6 * d))

        dx = int(np.sin(np.deg2rad(angle)) * length)
        dy = int(np.cos(np.deg2rad(angle)) * length)

        cv2.line(
            layer,
            (x, y),
            (x + dx, y + dy),
            1,
            1,
        )

    layer = cv2.GaussianBlur(layer, (3, 3), 0)

    return layer


# ==========================================================
# SNOW
# ==========================================================

def generate_snow_layer(
    image,
    intensity=0.5,
):
    """
    Snow particles increase with depth and become smaller.
    """

    h, w = image.shape[:2]

    depth = estimate_depth(image)

    layer = np.zeros((h, w), dtype=np.float32)

    flakes = int(h * w * 0.00035 * intensity)

    ys = _rng.integers(0, h, flakes)
    xs = _rng.integers(0, w, flakes)

    for x, y in zip(xs, ys):

        d = depth[y, x]

        radius = max(
            1,
            int(5 * (1 - d) + 1),
        )

        alpha = 0.3 + 0.7 * d

        cv2.circle(
            layer,
            (x, y),
            radius,
            alpha,
            -1,
        )

    layer = cv2.GaussianBlur(layer, (5, 5), 0)

    return layer


# ==========================================================
# DUST / SAND
# ==========================================================

def generate_dust_layer(
    image,
    intensity=0.5,
):
    """
    Generates many tiny dust particles instead of large blobs.
    """

    h, w = image.shape[:2]

    depth = estimate_depth(image)

    layer = np.zeros((h, w), dtype=np.float32)

    count = int(h * w * 0.0009 * intensity)

    ys = _rng.integers(0, h, count)
    xs = _rng.integers(0, w, count)

    for x, y in zip(xs, ys):

        d = depth[y, x]

        radius = max(
            1,
            int(3 * (1 - d) + 1),
        )

        alpha = 0.2 + 0.5 * d

        cv2.circle(
            layer,
            (x, y),
            radius,
            alpha,
            -1,
        )

    layer = cv2.GaussianBlur(layer, (7, 7), 0)

    return layer