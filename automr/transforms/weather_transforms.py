import cv2
import numpy as np

from .utils import (
    create_rng,
    create_random_mask,
    blend_images,
)


# ==========================================================
# Shared Utilities
# ==========================================================

def _depth_map(h, w, gamma=2.0):
    """
    Simple pseudo-depth map.

    Top of image  -> farther away
    Bottom        -> closer
    """

    y = np.linspace(0, 1, h).reshape(h, 1)

    depth = (1.0 - y) ** gamma

    return np.repeat(
        depth,
        w,
        axis=1,
    )


def _soft_mask(
    h,
    w,
    rng,
):
    """
    Creates a random soft blending mask.
    """

    return create_random_mask(
        (h, w),
        rng=rng,
        min_regions=1,
        max_regions=2,
        min_scale=0.25,
        max_scale=0.60,
        blur_choices=(41, 61, 81),
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
    Depth-aware localized rain.
    """

    intensity = float(intensity)
    rng = create_rng(seed)

    img = image.astype(np.float32)

    h, w = img.shape[:2]

    depth = _depth_map(h, w)

    rain = np.zeros_like(img)

    alpha = np.zeros((h, w), np.float32)

    cells = int(rng.integers(3, 8))

    for _ in range(cells):

        mask = _soft_mask(
            h,
            w,
            rng,
        )

        alpha += mask

        angle = rng.uniform(-30, 30)

        dx = np.sin(np.deg2rad(angle))
        dy = np.cos(np.deg2rad(angle))

        density = int(
            h * w * intensity * 0.003
        )

        ys, xs = np.where(mask > 0.15)

        if len(xs) == 0:
            continue

        ids = rng.choice(
            len(xs),
            size=min(density, len(xs)),
            replace=False,
        )

        for i in ids:

            x = xs[i]
            y = ys[i]

            length = int(
                6 + 20 * depth[y, x]
            )

            thickness = (
                1
                if depth[y, x] > 0.5
                else 2
            )

            brightness = int(
                180 + 60 * depth[y, x]
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
            )

    rain = cv2.GaussianBlur(
        rain,
        (3, 3),
        0,
    )

    alpha = np.clip(alpha, 0, 1)

    alpha *= intensity

    return np.clip(

        blend_images(
            img,
            rain,
            alpha,
        ),

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
    Depth-aware snow.
    """

    intensity = float(intensity)
    rng = create_rng(seed)

    img = image.astype(np.float32)

    h, w = img.shape[:2]

    depth = _depth_map(h, w)

    snow = img.copy()

    flakes = int(
        h * w * intensity * 0.002
    )

    for _ in range(flakes):

        x = rng.integers(0, w)
        y = rng.integers(0, h)

        radius = max(
            1,
            int(
                1 + 5 * (1 - depth[y, x])
            ),
        )

        brightness = int(
            rng.integers(
                220,
                256,
            )
        )

        cv2.circle(
            snow,
            (x, y),
            radius,
            (
                brightness,
                brightness,
                brightness,
            ),
            -1,
        )

    snow = cv2.GaussianBlur(
        snow,
        (3, 3),
        0,
    )

    return np.clip(

        cv2.addWeighted(
            img,
            1 - 0.45 * intensity,
            snow,
            0.45 * intensity,
            0,
        ),

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
    Depth-aware atmospheric fog.
    """
    intensity = float(intensity)
    rng = create_rng(seed)

    img = image.astype(np.float32)

    h, w = img.shape[:2]

    depth = _depth_map(h, w)

    atmosphere = np.full_like(
        img,
        255,
        dtype=np.float32,
    )

    beta = rng.uniform(
        1.2,
        2.2,
    ) * intensity

    transmission = np.exp(
        -beta * depth
    )

    transmission = transmission[:, :, None]

    fog = (
        img * transmission
        + atmosphere * (1 - transmission)
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
# Sandstorm
# ==========================================================

def add_sandstorm(
    image,
    intensity=0.5,
    seed=None,
):
    """
    Depth-aware sandstorm.
    """

    intensity = float(intensity)
    rng = create_rng(seed)

    img = image.astype(np.float32)

    h, w = img.shape[:2]

    depth = _depth_map(h, w)

    sand_color = np.array(
        [175, 165, 120],
        dtype=np.float32,
    )

    sand = np.ones_like(img) * sand_color

    beta = rng.uniform(
        1.0,
        2.0,
    ) * intensity

    transmission = np.exp(
        -beta * depth
    )

    transmission = transmission[:, :, None]

    out = (
        img * transmission
        + sand * (1 - transmission)
    )

    particles = int(
        h * w * intensity * 0.002
    )

    for _ in range(particles):

        x = rng.integers(0, w)
        y = rng.integers(0, h)

        radius = rng.integers(1, 4)

        color = int(
            rng.integers(170, 205)
        )

        cv2.circle(
            out,
            (x, y),
            radius,
            (
                color,
                color - 10,
                color - 40,
            ),
            -1,
        )

    out = cv2.GaussianBlur(
        out,
        (9, 9),
        0,
    )

    return np.clip(
        out,
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
    Random airborne dust.
    """

    intensity = float(intensity)
    rng = create_rng(seed)

    img = image.astype(np.float32)

    h, w = img.shape[:2]

    dust = img.copy()

    count = int(
        h * w * intensity * 0.001
    )

    for _ in range(count):

        x = rng.integers(0, w)
        y = rng.integers(0, h)

        radius = rng.integers(2, 6)

        color = int(
            rng.integers(150, 220)
        )

        cv2.circle(
            dust,
            (x, y),
            radius,
            (
                color,
                color,
                color,
            ),
            -1,
        )

    dust = cv2.GaussianBlur(
        dust,
        (7, 7),
        0,
    )

    return np.clip(
        cv2.addWeighted(
            img,
            1 - 0.40 * intensity,
            dust,
            0.40 * intensity,
            0,
        ),
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
    intensity = float(intensity)

    img = image.astype(np.float32)

    white = np.full(
        img.shape,
        255.0,
        dtype=np.float32,
    )

    out = cv2.addWeighted(
        img,
        1.0 - 0.3 * intensity,
        white,
        0.3 * intensity,
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
    Random localized smoke clouds.
    """

    intensity = float(intensity)
    rng = create_rng(seed)

    img = image.astype(np.float32)

    h, w = img.shape[:2]

    smoke = np.zeros_like(img)

    alpha = np.zeros(
        (h, w),
        dtype=np.float32,
    )

    clouds = int(rng.integers(4, 10))

    for _ in range(clouds):

        mask = _soft_mask(
            h,
            w,
            rng,
        )

        alpha += mask

        color = rng.integers(
            150,
            220,
        )

        smoke += (
            mask[:, :, None]
            * color
        )

    alpha = np.clip(
        alpha,
        0,
        1,
    ) * intensity

    smoke = cv2.GaussianBlur(
        smoke,
        (25, 25),
        0,
    )

    out = blend_images(
        img,
        smoke,
        alpha,
    )

    return np.clip(
        out,
        0,
        255,
    ).astype(np.uint8)