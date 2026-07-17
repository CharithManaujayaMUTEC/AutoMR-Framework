import cv2
import numpy as np

from ..utils import create_rng


# ==========================================================
# Global Spatial Translation
# ==========================================================

def shift_right(
    image,
    pixels=5,
    seed=None,
):
    """
    Global Spatial Translation.

    Controlled parameter
    --------------------
    pixels : maximum translation magnitude

    Randomized
    ----------
    • translation direction (360°)
    • translation distance

    Reproducible when a seed is provided.
    """

    rng = create_rng(seed)

    pixels = max(1, int(pixels))

    # ---------------------------------
    # Random translation direction
    # ---------------------------------
    angle = rng.uniform(
        0.0,
        2.0 * np.pi,
    )

    distance = rng.uniform(
        pixels * 0.5,
        pixels,
    )

    dx = int(
        np.round(
            distance * np.cos(angle)
        )
    )

    dy = int(
        np.round(
            distance * np.sin(angle)
        )
    )

    # ---------------------------------
    # Translation matrix
    # ---------------------------------
    M = np.float32([
        [1, 0, dx],
        [0, 1, dy],
    ])

    # ---------------------------------
    # Apply translation
    # ---------------------------------
    translated = cv2.warpAffine(
        image,
        M,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )

    # ---------------------------------
    # Return translated image
    # ---------------------------------
    return translated.astype(np.uint8)