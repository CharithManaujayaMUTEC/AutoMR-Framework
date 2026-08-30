import cv2
import numpy as np

from .depth import (
    estimate_depth,
    blend_with_airlight,
)


# ==========================================================
# FOG
# ==========================================================

def apply_fog(
    image,
    intensity=0.5,
):
    """
    Realistic fog using atmospheric scattering.

    intensity:
        0.0 -> no fog
        1.0 -> dense fog
    """

    depth = estimate_depth(image)

    strength = 1.5 * intensity

    fog = blend_with_airlight(
        image=image,
        depth=depth,
        strength=strength,
        airlight=(235, 235, 235),
    )

    return fog


# ==========================================================
# HAZE
# ==========================================================

def apply_haze(
    image,
    intensity=0.5,
):
    """
    Mild atmospheric haze.
    """

    depth = estimate_depth(image)

    strength = 0.8 * intensity

    haze = blend_with_airlight(
        image=image,
        depth=depth,
        strength=strength,
        airlight=(220, 225, 235),
    )

    # Slight desaturation
    hsv = cv2.cvtColor(
        haze,
        cv2.COLOR_BGR2HSV,
    )

    hsv[:, :, 1] = (
        hsv[:, :, 1].astype(np.float32)
        * (1.0 - 0.25 * intensity)
    ).clip(0, 255)

    haze = cv2.cvtColor(
        hsv.astype(np.uint8),
        cv2.COLOR_HSV2BGR,
    )

    return haze


# ==========================================================
# SMOKE
# ==========================================================

def apply_smoke(
    image,
    intensity=0.5,
):
    """
    Smoke generated using multi-scale blurred noise.
    """

    h, w = image.shape[:2]

    noise = np.random.rand(h, w).astype(np.float32)

    noise = cv2.GaussianBlur(
        noise,
        (0, 0),
        sigmaX=25,
    )

    noise -= noise.min()
    noise /= noise.max() + 1e-8

    depth = estimate_depth(image)

    alpha = (
        noise
        * depth
        * intensity
        * 0.7
    )

    alpha = np.repeat(
        alpha[:, :, None],
        3,
        axis=2,
    )

    smoke_color = np.full_like(
        image,
        220,
        dtype=np.float32,
    )

    img = image.astype(np.float32)

    result = (
        img * (1.0 - alpha)
        + smoke_color * alpha
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


# ==========================================================
# DUST
# ==========================================================

def apply_dust(
    image,
    intensity=0.5,
):
    """
    Dust storm with brown atmospheric scattering.
    """

    depth = estimate_depth(image)

    dust = blend_with_airlight(
        image=image,
        depth=depth,
        strength=intensity,
        airlight=(175, 170, 145),
    )

    overlay = np.full_like(
        dust,
        (170, 165, 130),
    )

    alpha = 0.15 * intensity

    dust = cv2.addWeighted(
        dust,
        1 - alpha,
        overlay,
        alpha,
        0,
    )

    return dust


# ==========================================================
# SANDSTORM
# ==========================================================

def apply_sandstorm(
    image,
    intensity=0.5,
):
    """
    Heavy sandstorm.

    Combines:
        - Dust scattering
        - Strong color cast
        - Contrast reduction
    """

    result = apply_dust(
        image,
        intensity * 1.5,
    )

    hsv = cv2.cvtColor(
        result,
        cv2.COLOR_BGR2HSV,
    )

    hsv[:, :, 1] = (
        hsv[:, :, 1].astype(np.float32)
        * (0.75 - 0.25 * intensity)
    ).clip(0, 255)

    hsv[:, :, 2] = (
        hsv[:, :, 2].astype(np.float32)
        * (0.95 + 0.10 * intensity)
    ).clip(0, 255)

    result = cv2.cvtColor(
        hsv.astype(np.uint8),
        cv2.COLOR_HSV2BGR,
    )

    result = cv2.GaussianBlur(
        result,
        (5, 5),
        0,
    )

    return result