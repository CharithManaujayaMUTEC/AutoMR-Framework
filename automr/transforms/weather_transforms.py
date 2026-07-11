import cv2
import numpy as np

from .utils import (
    create_rng,
    blend_images,
)

# ==========================================================
# Shared Utilities
# ==========================================================


def _depth_map(h, w, gamma=2.0):
    """
    Simple pseudo depth map.

    Top    -> far
    Bottom -> near
    """

    y = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1)

    depth = (1.0 - y) ** gamma

    return np.repeat(depth, w, axis=1)


# ----------------------------------------------------------
# Low-frequency random field
# ----------------------------------------------------------

def _density_field(
    h,
    w,
    rng,
    scale=96,
    blur=81,
):
    """
    Generates a smooth random density field covering
    the entire image.

    No elliptical blobs.
    """

    gh = max(2, h // scale)
    gw = max(2, w // scale)

    field = rng.random((gh, gw), dtype=np.float32)

    field = cv2.resize(
        field,
        (w, h),
        interpolation=cv2.INTER_CUBIC,
    )

    if blur % 2 == 0:
        blur += 1

    field = cv2.GaussianBlur(
        field,
        (blur, blur),
        0,
    )

    field -= field.min()

    field /= (
        field.max() + 1e-6
    )

    return field.astype(np.float32)


# ----------------------------------------------------------
# Atmospheric scattering
# ----------------------------------------------------------

def _atmospheric_scatter(
    image,
    airlight,
    beta,
    depth,
):
    """
    Koschmieder atmospheric scattering model.
    """

    transmission = np.exp(
        -beta * depth
    ).astype(np.float32)

    transmission = transmission[:, :, None]

    out = (
        image * transmission
        + airlight * (1.0 - transmission)
    )

    return out.astype(np.float32)


# ----------------------------------------------------------
# Global contrast reduction
# ----------------------------------------------------------

def _reduce_contrast(
    image,
    amount,
):
    """
    Slight global contrast reduction.
    """

    mean = np.mean(
        image,
        axis=(0, 1),
        keepdims=True,
    )

    return (
        mean
        + (image - mean) * (1.0 - amount)
    )


# ----------------------------------------------------------
# Color cast
# ----------------------------------------------------------

def _color_cast(
    image,
    color,
    strength,
):
    """
    Applies a global color cast.
    """

    color = np.asarray(
        color,
        dtype=np.float32,
    )

    return (
        image * (1.0 - strength)
        + color * strength
    )

# ==========================================================
# Rain
# ==========================================================

def add_rain(
    image,
    intensity=0.5,
    seed=None,
):
    """
    Physically-inspired rain.

    Characteristics
    ---------------
    • Whole-image rainfall
    • Spatial density variation
    • Perspective streak length
    • Wind
    • Motion blur
    • Slight atmospheric haze
    """

    intensity = float(intensity)

    rng = create_rng(seed)

    img = image.astype(np.float32)

    h, w = img.shape[:2]

    depth = _depth_map(h, w)

    density = _density_field(
        h,
        w,
        rng,
        scale=96,
        blur=81,
    )

    rain = np.zeros_like(img)

    angle = float(
        rng.uniform(-20, 20)
    )

    dx = np.sin(
        np.deg2rad(angle)
    )

    dy = np.cos(
        np.deg2rad(angle)
    )

    n_streaks = int(
        h * w * intensity * 0.0015
    )

    ys = rng.integers(
        0,
        h,
        n_streaks,
    )

    xs = rng.integers(
        0,
        w,
        n_streaks,
    )

    for x, y in zip(xs, ys):

        if rng.random() > density[y, x]:
            continue

        d = depth[y, x]

        length = int(
            8 + 22 * d
        )

        thickness = (
            1
            if d > 0.4
            else 2
        )

        brightness = int(
            170 + 55 * d
        )

        x2 = int(x + dx * length)
        y2 = int(y + dy * length)

        cv2.line(
            rain,
            (x, y),
            (x2, y2),
            (
                brightness,
                brightness,
                brightness,
            ),
            thickness,
            cv2.LINE_AA,
        )

    rain = cv2.GaussianBlur(
        rain,
        (3, 3),
        0,
    )

    airlight = np.full_like(
        img,
        230,
        dtype=np.float32,
    )

    scene = _atmospheric_scatter(
        img,
        airlight,
        beta=0.30 * intensity,
        depth=depth,
    )

    scene = _reduce_contrast(
        scene,
        0.15 * intensity,
    )

    out = cv2.addWeighted(
        scene,
        1.0,
        rain,
        0.55,
        0,
        dtype=cv2.CV_32F,
    )

    return np.clip(
        out,
        0,
        255,
    ).astype(np.uint8)

# ==========================================================
# Snow
# ==========================================================

def add_snow(
    image,
    intensity=0.5,
    seed=None,
):
    """
    Physically-inspired snowfall.

    Characteristics
    ---------------
    • Whole-image snowfall
    • Multi-layer flakes
    • Perspective scaling
    • Atmospheric whitening
    """

    intensity = float(intensity)

    rng = create_rng(seed)

    img = image.astype(np.float32)

    h, w = img.shape[:2]

    depth = _depth_map(h, w)

    density = _density_field(
        h,
        w,
        rng,
        scale=120,
        blur=101,
    )

    snow = np.zeros_like(img)

    layers = 3

    for layer in range(layers):

        count = int(
            h * w * intensity *
            (0.0005 + layer * 0.00035)
        )

        xs = rng.integers(0, w, count)
        ys = rng.integers(0, h, count)

        for x, y in zip(xs, ys):

            if rng.random() > density[y, x]:
                continue

            d = depth[y, x]

            radius = max(
                1,
                int(
                    1
                    + (1.0 - d) * (2 + layer)
                ),
            )

            brightness = int(
                rng.uniform(220, 255)
            )

            cv2.circle(
                snow,
                (int(x), int(y)),
                radius,
                (
                    brightness,
                    brightness,
                    brightness,
                ),
                -1,
                lineType=cv2.LINE_AA,
            )

    snow = cv2.GaussianBlur(
        snow,
        (5, 5),
        0,
    )

    airlight = np.full_like(
        img,
        245,
        dtype=np.float32,
    )

    scene = _atmospheric_scatter(
        img,
        airlight,
        beta=0.22 * intensity,
        depth=depth,
    )

    scene = _reduce_contrast(
        scene,
        0.10 * intensity,
    )

    out = cv2.addWeighted(
        scene,
        1.0,
        snow,
        0.65,
        0,
        dtype=cv2.CV_32F,
    )

    return np.clip(
        out,
        0,
        255,
    ).astype(np.uint8)

# ==========================================================
# Fog
# ==========================================================

def add_fog(
    image,
    intensity=0.5,
    seed=None,
):
    """
    Physically-inspired fog.

    Characteristics
    ---------------
    • Whole-scene fog
    • Depth-aware attenuation
    • Smooth density variation
    • Atmospheric scattering
    """

    intensity = float(intensity)

    rng = create_rng(seed)

    img = image.astype(np.float32)

    h, w = img.shape[:2]

    depth = _depth_map(h, w)

    density = _density_field(
        h,
        w,
        rng,
        scale=180,
        blur=151,
    )

    beta = (
        1.2
        + 1.2 * density
    ) * intensity

    airlight = np.full_like(
        img,
        245,
        dtype=np.float32,
    )

    fog = _atmospheric_scatter(
        img,
        airlight,
        beta,
        depth,
    )

    fog = _reduce_contrast(
        fog,
        0.20 * intensity,
    )

    fog = cv2.GaussianBlur(
        fog,
        (15, 15),
        0,
    )

    return np.clip(
        fog,
        0,
        255,
    ).astype(np.uint8)


# ==========================================================
# Haze
# ==========================================================

def add_haze(
    image,
    intensity=0.5,
    seed=None,
):
    """
    Light atmospheric haze.

    Characteristics
    ---------------
    • Mild atmospheric scattering
    • Slight whitening
    • Contrast reduction
    • Smooth spatial variation
    """

    intensity = float(intensity)

    rng = create_rng(seed)

    img = image.astype(np.float32)

    h, w = img.shape[:2]

    depth = _depth_map(h, w)

    density = _density_field(
        h,
        w,
        rng,
        scale=220,
        blur=181,
    )

    beta = (
        0.35
        + 0.35 * density
    ) * intensity

    airlight = np.full_like(
        img,
        255,
        dtype=np.float32,
    )

    haze = _atmospheric_scatter(
        img,
        airlight,
        beta,
        depth,
    )

    haze = _reduce_contrast(
        haze,
        0.10 * intensity,
    )

    haze = _color_cast(
        haze,
        [245, 248, 255],
        0.05 * intensity,
    )

    haze = cv2.GaussianBlur(
        haze,
        (9, 9),
        0,
    )

    return np.clip(
        haze,
        0,
        255,
    ).astype(np.uint8)

# ==========================================================
# Dust
# ==========================================================

def add_dust(
    image,
    intensity=0.5,
    seed=None,
):
    """
    Atmospheric airborne dust.

    Characteristics
    ---------------
    • Whole-image dust field
    • Small drifting particles
    • Slight brown color cast
    • Contrast reduction
    """

    intensity = float(intensity)

    rng = create_rng(seed)

    img = image.astype(np.float32)

    h, w = img.shape[:2]

    depth = _depth_map(h, w)

    density = _density_field(
        h,
        w,
        rng,
        scale=140,
        blur=121,
    )

    dust = np.zeros_like(img)

    particles = int(
        h * w * intensity * 0.0012
    )

    xs = rng.integers(0, w, particles)
    ys = rng.integers(0, h, particles)

    for x, y in zip(xs, ys):

        if rng.random() > density[y, x]:
            continue

        r = max(
            1,
            int(
                1 + (1.0 - depth[y, x]) * 3
            ),
        )

        c = int(
            rng.uniform(170, 220)
        )

        cv2.circle(
            dust,
            (int(x), int(y)),
            r,
            (
                c,
                c - 10,
                c - 20,
            ),
            -1,
            cv2.LINE_AA,
        )

    dust = cv2.GaussianBlur(
        dust,
        (7, 7),
        0,
    )

    scene = _color_cast(
        img,
        [195, 185, 160],
        0.06 * intensity,
    )

    scene = _reduce_contrast(
        scene,
        0.08 * intensity,
    )

    out = cv2.addWeighted(
        scene,
        1.0,
        dust,
        0.35,
        0,
        dtype=cv2.CV_32F,
    )

    return np.clip(
        out,
        0,
        255,
    ).astype(np.uint8)


# ==========================================================
# Sandstorm
# ==========================================================

def add_sandstorm(
    image,
    intensity=0.5,
    seed=None,
):
    """
    Realistic sandstorm.

    Characteristics
    ---------------
    • Atmospheric scattering
    • Yellow/brown color cast
    • Dust particles
    • Reduced visibility
    """

    intensity = float(intensity)

    rng = create_rng(seed)

    img = image.astype(np.float32)

    h, w = img.shape[:2]

    depth = _depth_map(h, w)

    density = _density_field(
        h,
        w,
        rng,
        scale=110,
        blur=101,
    )

    beta = (
        0.8
        + density * 1.2
    ) * intensity

    airlight = np.full(
        img.shape,
        (190, 175, 130),
        dtype=np.float32,
    )

    scene = _atmospheric_scatter(
        img,
        airlight,
        beta,
        depth,
    )

    scene = _reduce_contrast(
        scene,
        0.25 * intensity,
    )

    scene = _color_cast(
        scene,
        [188, 173, 130],
        0.18 * intensity,
    )

    particles = np.zeros_like(scene)

    count = int(
        h * w * intensity * 0.0015
    )

    xs = rng.integers(0, w, count)
    ys = rng.integers(0, h, count)

    for x, y in zip(xs, ys):

        if rng.random() > density[y, x]:
            continue

        r = max(
            1,
            int(
                1 + (1.0 - depth[y, x]) * 2
            ),
        )

        c = int(
            rng.uniform(170, 210)
        )

        cv2.circle(
            particles,
            (int(x), int(y)),
            r,
            (
                c,
                c - 15,
                c - 45,
            ),
            -1,
            cv2.LINE_AA,
        )

    particles = cv2.GaussianBlur(
        particles,
        (5, 5),
        0,
    )

    out = cv2.addWeighted(
        scene,
        1.0,
        particles,
        0.30,
        0,
        dtype=cv2.CV_32F,
    )

    return np.clip(
        out,
        0,
        255,
    ).astype(np.uint8)

# ==========================================================
# Smoke
# ==========================================================

def add_smoke(
    image,
    intensity=0.5,
    seed=None,
):
    """
    Physically-inspired smoke.

    Characteristics
    ---------------
    • Global turbulent smoke field
    • Irregular cloud structures
    • Smooth density variation
    • Atmospheric scattering
    """

    intensity = float(intensity)

    rng = create_rng(seed)

    img = image.astype(np.float32)

    h, w = img.shape[:2]

    depth = _depth_map(h, w)

    # Multi-scale turbulence
    density = (
        0.55 * _density_field(h, w, rng, scale=200, blur=151)
        + 0.30 * _density_field(h, w, rng, scale=90, blur=81)
        + 0.15 * _density_field(h, w, rng, scale=45, blur=41)
    )

    density -= density.min()
    density /= density.max() + 1e-6

    # Keep only denser regions
    density = np.clip((density - 0.35) / 0.65, 0.0, 1.0)

    smoke_color = np.full(
        img.shape,
        180,
        dtype=np.float32,
    )

    beta = (
        0.45
        + density * 1.2
    ) * intensity

    smoke = _atmospheric_scatter(
        img,
        smoke_color,
        beta,
        depth,
    )

    smoke = _reduce_contrast(
        smoke,
        0.22 * intensity,
    )

    alpha = (
        density ** 1.8
    ) * 0.65 * intensity

    out = blend_images(
        img,
        smoke,
        alpha,
    )

    out = cv2.GaussianBlur(
        out,
        (15, 15),
        0,
    )

    return np.clip(
        out,
        0,
        255,
    ).astype(np.uint8)