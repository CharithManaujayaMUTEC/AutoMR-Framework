import cv2
import numpy as np

from .weather_particles import (
    generate_rain_layer,
    generate_snow_layer,
    generate_dust_layer,
)

from .atmospheric import (
    apply_fog,
    apply_haze,
    apply_smoke,
    apply_dust,
    apply_sandstorm,
)


# ==========================================================
# RAIN
# ==========================================================

def render_rain(
    image,
    intensity=0.5,
):
    """
    Render realistic rain.

    Combines:
        - Rain streaks
        - Slight atmospheric haze
        - Contrast reduction
    """

    img = image.astype(np.float32)

    rain = generate_rain_layer(
        image,
        intensity=intensity,
    )

    rain = cv2.GaussianBlur(
        rain,
        (0, 0),
        sigmaX=1.2,
    )

    rain = np.repeat(
        rain[:, :, None],
        3,
        axis=2,
    )

    result = img + rain * 180

    result = np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)

    result = apply_haze(
        result,
        intensity * 0.30,
    )

    result = cv2.convertScaleAbs(
        result,
        alpha=(1 - 0.08 * intensity),
        beta=0,
    )

    return result


# ==========================================================
# SNOW
# ==========================================================

def render_snow(
    image,
    intensity=0.5,
):
    """
    Render realistic snowfall.
    """

    img = image.astype(np.float32)

    snow = generate_snow_layer(
        image,
        intensity=intensity,
    )

    snow = np.repeat(
        snow[:, :, None],
        3,
        axis=2,
    )

    result = img + snow * 255

    result = np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)

    result = apply_fog(
        result,
        intensity * 0.20,
    )

    return result


# ==========================================================
# DUST
# ==========================================================

def render_dust(
    image,
    intensity=0.5,
):
    """
    Render dusty environment.
    """

    result = apply_dust(
        image,
        intensity,
    )

    particles = generate_dust_layer(
        image,
        intensity,
    )

    particles = np.repeat(
        particles[:, :, None],
        3,
        axis=2,
    )

    result = (
        result.astype(np.float32)
        + particles * 70
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


# ==========================================================
# SANDSTORM
# ==========================================================

def render_sandstorm(
    image,
    intensity=0.5,
):
    """
    Render heavy sandstorm.
    """

    result = apply_sandstorm(
        image,
        intensity,
    )

    particles = generate_dust_layer(
        image,
        intensity * 1.5,
    )

    particles = np.repeat(
        particles[:, :, None],
        3,
        axis=2,
    )

    result = (
        result.astype(np.float32)
        + particles * 40
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


# ==========================================================
# FOG
# ==========================================================

def render_fog(
    image,
    intensity=0.5,
):
    return apply_fog(
        image,
        intensity,
    )


# ==========================================================
# HAZE
# ==========================================================

def render_haze(
    image,
    intensity=0.5,
):
    return apply_haze(
        image,
        intensity,
    )


# ==========================================================
# SMOKE
# ==========================================================

def render_smoke(
    image,
    intensity=0.5,
):
    return apply_smoke(
        image,
        intensity,
    )