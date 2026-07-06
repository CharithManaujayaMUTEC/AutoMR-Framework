import cv2
import numpy as np


# ==========================================================
# Shared Utilities
# ==========================================================

def _depth_map(h, w, gamma=2.0):
    """
    Simple pseudo-depth map.
    Top of image = farther away
    Bottom = closer to camera
    """
    y = np.linspace(0, 1, h).reshape(h, 1)
    depth = (1 - y) ** gamma
    return np.repeat(depth, w, axis=1)


def _soft_mask(h, w):

    mask = np.zeros((h, w), np.uint8)

    axis_x = np.random.randint(
        max(40, w // 12),
        max(100, w // 3)
    )

    axis_y = np.random.randint(
        max(40, h // 12),
        max(100, h // 3)
    )

    center = (
        np.random.randint(0, w),
        np.random.randint(0, h)
    )

    angle = np.random.uniform(0, 360)

    cv2.ellipse(
        mask,
        center,
        (axis_x, axis_y),
        angle,
        0,
        360,
        255,
        -1,
    )

    k = np.random.choice([41, 61, 81])

    mask = cv2.GaussianBlur(mask, (k, k), 0)

    return mask.astype(np.float32) / 255.0


# ==========================================================
# Rain
# ==========================================================

def add_rain(image, intensity=0.5):

    img = image.astype(np.float32)

    h, w = img.shape[:2]

    depth = _depth_map(h, w)

    rain = np.zeros_like(img)

    alpha = np.zeros((h, w), np.float32)

    cells = np.random.randint(3, 8)

    for _ in range(cells):

        mask = _soft_mask(h, w)

        alpha += mask

        angle = np.random.uniform(-30, 30)

        dx = np.sin(np.deg2rad(angle))

        dy = np.cos(np.deg2rad(angle))

        density = int(h * w * intensity * 0.003)

        ys, xs = np.where(mask > 0.2)

        if len(xs) == 0:
            continue

        ids = np.random.choice(
            len(xs),
            min(density, len(xs)),
            replace=False
        )

        for i in ids:

            x = xs[i]
            y = ys[i]

            length = int(8 + 20 * depth[y, x])

            x2 = int(x + dx * length)

            y2 = int(y + dy * length)

            cv2.line(
                rain,
                (x, y),
                (x2, y2),
                (220, 220, 220),
                1
            )

    alpha = np.clip(alpha, 0, 1) * intensity

    out = img * (1 - alpha[:, :, None]) + rain * alpha[:, :, None]

    return np.clip(out, 0, 255).astype(np.uint8)


# ==========================================================
# Snow
# ==========================================================

def add_snow(image, intensity=0.5):

    img = image.astype(np.float32)

    h, w = img.shape[:2]

    depth = _depth_map(h, w)

    snow = img.copy()

    flakes = int(h * w * intensity * 0.002)

    for _ in range(flakes):

        x = np.random.randint(w)
        y = np.random.randint(h)

        radius = max(1, int(1 + 5 * (1 - depth[y, x])))

        cv2.circle(
            snow,
            (x, y),
            radius,
            (255, 255, 255),
            -1
        )

    snow = cv2.GaussianBlur(snow, (3, 3), 0)

    out = cv2.addWeighted(
        img,
        1 - 0.5 * intensity,
        snow,
        0.5 * intensity,
        0
    )

    return np.clip(out, 0, 255).astype(np.uint8)


# ==========================================================
# Fog
# ==========================================================

def add_fog(image, intensity=0.5):

    img = image.astype(np.float32)

    h, w = img.shape[:2]

    depth = _depth_map(h, w)

    atmosphere = np.full_like(img, 255)

    beta = intensity * 2.0

    transmission = np.exp(-beta * depth)

    transmission = transmission[:, :, None]

    out = img * transmission + atmosphere * (1 - transmission)

    out = cv2.GaussianBlur(out, (15, 15), 0)

    return np.clip(out, 0, 255).astype(np.uint8)


# ==========================================================
# Sandstorm
# ==========================================================

def add_sandstorm(image, intensity=0.5):

    img = image.astype(np.float32)

    h, w = img.shape[:2]

    depth = _depth_map(h, w)

    sand = np.full_like(
        img,
        (170, 160, 120),
        dtype=np.float32
    )

    beta = intensity * 1.6

    transmission = np.exp(-beta * depth)

    transmission = transmission[:, :, None]

    out = img * transmission + sand * (1 - transmission)

    particles = int(h * w * intensity * 0.002)

    for _ in range(particles):

        x = np.random.randint(w)
        y = np.random.randint(h)

        r = np.random.randint(1, 4)

        cv2.circle(
            out,
            (x, y),
            r,
            (185, 175, 130),
            -1
        )

    out = cv2.GaussianBlur(out, (9, 9), 0)

    return np.clip(out, 0, 255).astype(np.uint8)


# ==========================================================
# Dust
# ==========================================================

def add_dust(image, intensity=0.5):

    img = image.astype(np.float32)

    h, w = img.shape[:2]

    dust = img.copy()

    count = int(h * w * intensity * 0.001)

    for _ in range(count):

        x = np.random.randint(w)
        y = np.random.randint(h)

        r = np.random.randint(2, 6)

        color = np.random.randint(140, 210)

        cv2.circle(
            dust,
            (x, y),
            r,
            (color, color, color),
            -1
        )

    dust = cv2.GaussianBlur(dust, (7, 7), 0)

    return np.clip(
        cv2.addWeighted(
            img,
            1 - 0.4 * intensity,
            dust,
            0.4 * intensity,
            0
        ),
        0,
        255,
    ).astype(np.uint8)


# ==========================================================
# Haze
# ==========================================================

def add_haze(image, intensity=0.5):

    img = image.astype(np.float32)

    white = np.full_like(img, 255)

    out = cv2.addWeighted(
        img,
        1 - 0.3 * intensity,
        white,
        0.3 * intensity,
        0
    )

    return np.clip(out, 0, 255).astype(np.uint8)


# ==========================================================
# Smoke
# ==========================================================

def add_smoke(image, intensity=0.5):

    img = image.astype(np.float32)

    h, w = img.shape[:2]

    smoke = np.zeros_like(img)

    cells = np.random.randint(4, 10)

    for _ in range(cells):

        mask = _soft_mask(h, w)

        color = np.random.randint(150, 220)

        smoke += mask[:, :, None] * color

    smoke = np.clip(smoke, 0, 255)

    out = cv2.addWeighted(
        img,
        1 - 0.5 * intensity,
        smoke,
        0.5 * intensity,
        0
    )

    return np.clip(out, 0, 255).astype(np.uint8)