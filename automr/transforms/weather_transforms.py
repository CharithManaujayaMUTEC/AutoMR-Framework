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
# Camera Lens Effects
# ==========================================================

def _apply_lens_droplets(
    image,
    rng,
    intensity,
    tint=(255,255,255),
):
    """
    Realistic camera-lens droplets.

    Used for:
        rain
        snow
        fog
        dust
        smoke
    """

    img = image.astype(np.float32)

    h, w = img.shape[:2]

    alpha = np.zeros((h,w), np.float32)

    blurred = cv2.GaussianBlur(
        img,
        (0,0),
        sigmaX=8,
    )

    out = img.copy()

    droplets = int(
        rng.uniform(
            10,
            80
        ) * intensity
    )

    for _ in range(droplets):

        cx = int(rng.integers(0,w))
        cy = int(rng.integers(0,h))

        r = rng.uniform(
            6,
            35
        ) * (0.5 + intensity)

        mask = np.zeros((h,w), np.float32)

        cv2.circle(
            mask,
            (cx,cy),
            int(r),
            1,
            -1,
        )

        mask = cv2.GaussianBlur(
            mask,
            (0,0),
            sigmaX=r*0.25,
        )

        alpha = np.maximum(
            alpha,
            mask*0.55,
        )

        # highlight

        hx = int(cx-r*0.3)
        hy = int(cy-r*0.3)

        cv2.circle(
            out,
            (hx,hy),
            max(1,int(r*0.18)),
            tint,
            -1,
        )

    out = (
        out*(1-alpha[:,:,None])
        + blurred*alpha[:,:,None]
    )

    return out

# ==========================================================
# Rain
# ==========================================================

def add_rain(
    image,
    intensity=0.5,
    seed=None,
):
    """
    Realistic dashcam rain.

    Features
    --------
    • Perspective rain
    • Wind variation
    • Lens droplets
    • Slight refraction
    • Atmospheric darkening
    """

    intensity = float(intensity)
    rng = create_rng(seed)

    img = image.astype(np.float32)

    h, w = img.shape[:2]

    depth = _depth_map(h, w)

    # --------------------------------------------------
    # Slight rainy atmosphere
    # --------------------------------------------------

    out = img * (1.0 - 0.15 * intensity)

    # --------------------------------------------------
    # Rain streak layer
    # --------------------------------------------------

    rain = np.zeros_like(out)

    wind = rng.uniform(-18, 18)

    n = int(
        h * w * (0.0008 + 0.0025 * intensity)
    )

    for _ in range(n):

        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))

        d = depth[y, x]

        length = int(
            rng.uniform(8, 28)
            * (1.8 - d)
        )

        thickness = max(
            1,
            int(
                rng.uniform(1, 3)
                * (1.8 - d)
            ),
        )

        angle = wind + rng.normal(0, 5)

        dx = np.sin(np.deg2rad(angle))
        dy = np.cos(np.deg2rad(angle))

        x2 = int(x + dx * length)
        y2 = int(y + dy * length)

        b = int(
            rng.uniform(170, 255)
        )

        cv2.line(
            rain,
            (x, y),
            (x2, y2),
            (b, b, b),
            thickness,
            cv2.LINE_AA,
        )

    rain = cv2.GaussianBlur(
        rain,
        (3, 3),
        0,
    )

    out = cv2.addWeighted(
        out,
        1.0,
        rain,
        0.35 * intensity,
        0,
        dtype=cv2.CV_32F,
    )

    # --------------------------------------------------
    # Lens droplets
    # --------------------------------------------------

    droplets = int(
        20 + intensity * 120
    )

    for _ in range(droplets):

        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))

        r = int(
            rng.uniform(
                3,
                18 + 30 * intensity,
            )
        )

        mask = np.zeros(
            (h, w),
            np.uint8,
        )

        cv2.circle(
            mask,
            (cx, cy),
            r,
            255,
            -1,
            cv2.LINE_AA,
        )

        blur = max(
            9,
            int(r * 2 + 1),
        )

        if blur % 2 == 0:
            blur += 1

        mask = cv2.GaussianBlur(
            mask,
            (blur, blur),
            0,
        )

        alpha = (
            mask.astype(np.float32)
            / 255.0
        )

        alpha *= rng.uniform(
            0.15,
            0.45,
        ) * intensity

        # slight magnification
        scale = rng.uniform(
            1.01,
            1.05,
        )

        M = cv2.getRotationMatrix2D(
            (cx, cy),
            0,
            scale,
        )

        warped = cv2.warpAffine(
            out,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )

        out = (
            out * (1 - alpha[:, :, None])
            + warped * alpha[:, :, None]
        )

        cv2.circle(
            out,
            (cx, cy),
            r,
            (
                255,
                255,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

    # --------------------------------------------------
    # Final blur
    # --------------------------------------------------

    out = cv2.GaussianBlur(
        out,
        (3, 3),
        0,
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
    Realistic atmospheric fog.

    Features
    --------
    • Depth-aware scattering
    • Patchy density
    • Soft turbulence
    • Contrast reduction
    """

    intensity = float(intensity)
    rng = create_rng(seed)

    img = image.astype(np.float32)

    h, w = img.shape[:2]

    depth = _depth_map(h, w)

    # ---------------------------------------------
    # Large smooth fog density
    # ---------------------------------------------

    noise = rng.normal(
        0,
        1,
        (h, w),
    ).astype(np.float32)

    noise = cv2.GaussianBlur(
        noise,
        (0, 0),
        sigmaX=max(h, w) / 12,
    )

    noise -= noise.min()
    noise /= noise.max() + 1e-6

    density = (
        0.5
        + 0.5 * noise
    )

    beta = (
        1.4
        + rng.uniform(-0.2, 0.2)
    ) * intensity

    transmission = np.exp(
        -beta * depth * density
    )

    transmission = transmission[:, :, None]

    atmosphere = np.full_like(
        img,
        255,
        dtype=np.float32,
    )

    out = (
        img * transmission
        + atmosphere * (1.0 - transmission)
    )

    # ---------------------------------------------
    # Slight desaturation
    # ---------------------------------------------

    gray = cv2.cvtColor(
        out.astype(np.uint8),
        cv2.COLOR_BGR2GRAY,
    ).astype(np.float32)

    gray = gray[:, :, None]

    out = (
        out * (1 - 0.20 * intensity)
        + gray * (0.20 * intensity)
    )

    # ---------------------------------------------
    # Reduce distant contrast
    # ---------------------------------------------

    blur = cv2.GaussianBlur(
        out,
        (0, 0),
        sigmaX=8,
    )

    far = (
        1.0 - depth
    )[:, :, None]

    out = (
        out * (1 - far * 0.25 * intensity)
        + blur * (far * 0.25 * intensity)
    )

    return np.clip(
        out,
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
    Realistic atmospheric haze.

    Features
    --------
    • Uneven atmospheric scattering
    • Reduced contrast
    • Slight bluish tint
    • Large-scale density variation
    """

    intensity = float(intensity)

    rng = create_rng(seed)

    img = image.astype(np.float32)

    h, w = img.shape[:2]

    # --------------------------------------------------
    # Large-scale haze density
    # --------------------------------------------------

    noise = rng.normal(
        0,
        1,
        (h, w),
    ).astype(np.float32)

    noise = cv2.GaussianBlur(
        noise,
        (0, 0),
        sigmaX=max(h, w) / 8,
    )

    noise -= noise.min()
    noise /= noise.max() + 1e-6

    density = (
        0.55
        + 0.45 * noise
    )

    density *= intensity

    density = density[:, :, None]

    # --------------------------------------------------
    # Slight blue-gray atmosphere
    # --------------------------------------------------

    atmosphere = np.full_like(
        img,
        (
            235,
            238,
            245,
        ),
        dtype=np.float32,
    )

    out = (
        img * (1 - density * 0.45)
        + atmosphere * (density * 0.45)
    )

    # --------------------------------------------------
    # Reduce contrast
    # --------------------------------------------------

    mean = cv2.GaussianBlur(
        out,
        (0, 0),
        sigmaX=25,
    )

    contrast = (
        1.0
        - 0.30 * intensity
    )

    out = mean + contrast * (out - mean)

    # --------------------------------------------------
    # Soft atmospheric blur
    # --------------------------------------------------

    blur = cv2.GaussianBlur(
        out,
        (0, 0),
        sigmaX=2 + 4 * intensity,
    )

    out = cv2.addWeighted(
        out,
        0.8,
        blur,
        0.2,
        0,
        dtype=cv2.CV_32F,
    )

    # --------------------------------------------------
    # Slight desaturation
    # --------------------------------------------------

    gray = cv2.cvtColor(
        out.astype(np.uint8),
        cv2.COLOR_BGR2GRAY,
    ).astype(np.float32)

    gray = gray[:, :, None]

    out = (
        out * (1 - 0.12 * intensity)
        + gray * (0.12 * intensity)
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
    Realistic airborne dust.

    Features
    --------
    • Fine suspended particles
    • Multi-scale dust clouds
    • Warm brown atmospheric tint
    • Reduced visibility
    • Slight lens softness
    """

    intensity = float(intensity)

    rng = create_rng(seed)

    img = image.astype(np.float32)

    h, w = img.shape[:2]

    out = img.copy()

    # --------------------------------------------------
    # Multi-scale dust density
    # --------------------------------------------------

    n1 = rng.normal(0, 1, (h, w)).astype(np.float32)
    n2 = rng.normal(0, 1, (h, w)).astype(np.float32)
    n3 = rng.normal(0, 1, (h, w)).astype(np.float32)

    n1 = cv2.GaussianBlur(n1, (0, 0), sigmaX=max(h, w) / 5)
    n2 = cv2.GaussianBlur(n2, (0, 0), sigmaX=max(h, w) / 12)
    n3 = cv2.GaussianBlur(n3, (0, 0), sigmaX=max(h, w) / 30)

    density = (
        0.55 * n1 +
        0.30 * n2 +
        0.15 * n3
    )

    density -= density.min()
    density /= density.max() + 1e-6

    density = np.power(density, 1.8)

    density *= intensity

    density = density[:, :, None]

    # --------------------------------------------------
    # Warm dust atmosphere
    # --------------------------------------------------

    dust_color = np.full_like(
        img,
        (175, 170, 145),
        dtype=np.float32,
    )

    out = (
        img * (1 - density * 0.45)
        + dust_color * (density * 0.45)
    )

    # --------------------------------------------------
    # Floating particles
    # --------------------------------------------------

    particles = np.zeros_like(out)

    count = int(
        h * w * intensity * 0.002
    )

    for _ in range(count):

        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))

        r = int(rng.uniform(1, 4))

        c = int(rng.uniform(170, 230))

        cv2.circle(
            particles,
            (x, y),
            r,
            (c, c - 8, c - 25),
            -1,
            cv2.LINE_AA,
        )

    particles = cv2.GaussianBlur(
        particles,
        (0, 0),
        sigmaX=1.5,
    )

    out = cv2.addWeighted(
        out,
        1.0,
        particles,
        0.45 * intensity,
        0,
        dtype=cv2.CV_32F,
    )

    # --------------------------------------------------
    # Reduce sharpness
    # --------------------------------------------------

    blur = cv2.GaussianBlur(
        out,
        (0, 0),
        sigmaX=2 + 4 * intensity,
    )

    out = cv2.addWeighted(
        out,
        0.85,
        blur,
        0.15,
        0,
        dtype=cv2.CV_32F,
    )

    # --------------------------------------------------
    # Slight contrast reduction
    # --------------------------------------------------

    mean = cv2.GaussianBlur(
        out,
        (0, 0),
        sigmaX=25,
    )

    out = mean + (
        1 - 0.25 * intensity
    ) * (out - mean)

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

    Features
    --------
    • Directional blowing sand
    • Multi-scale dust density
    • Warm atmospheric tint
    • Flying sand particles
    • Motion blur
    """

    intensity = float(intensity)

    rng = create_rng(seed)

    img = image.astype(np.float32)

    h, w = img.shape[:2]

    out = img.copy()

    wind = rng.uniform(-25, 25)

    # --------------------------------------------------
    # Large atmospheric dust
    # --------------------------------------------------

    n1 = rng.normal(0, 1, (h, w)).astype(np.float32)
    n2 = rng.normal(0, 1, (h, w)).astype(np.float32)
    n3 = rng.normal(0, 1, (h, w)).astype(np.float32)

    n1 = cv2.GaussianBlur(n1, (0, 0), sigmaX=max(h, w) / 4)
    n2 = cv2.GaussianBlur(n2, (0, 0), sigmaX=max(h, w) / 12)
    n3 = cv2.GaussianBlur(n3, (0, 0), sigmaX=max(h, w) / 28)

    density = (
        0.55 * n1 +
        0.30 * n2 +
        0.15 * n3
    )

    density -= density.min()
    density /= density.max() + 1e-6

    density = np.power(density, 1.7)

    density *= intensity

    density = density[:, :, None]

    sand_color = np.full_like(
        out,
        (175, 165, 125),
        dtype=np.float32,
    )

    out = (
        out * (1 - density * 0.55)
        + sand_color * (density * 0.55)
    )

    # --------------------------------------------------
    # Blowing sand streaks
    # --------------------------------------------------

    streaks = np.zeros_like(out)

    count = int(
        h * w * intensity * 0.0025
    )

    theta = np.deg2rad(wind)

    dx = np.cos(theta)
    dy = np.sin(theta)

    for _ in range(count):

        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))

        length = int(rng.uniform(8, 28))

        x2 = int(x + dx * length)
        y2 = int(y + dy * length)

        color = int(rng.uniform(170, 220))

        cv2.line(
            streaks,
            (x, y),
            (x2, y2),
            (
                color,
                color - 15,
                color - 40,
            ),
            1,
            cv2.LINE_AA,
        )

    # --------------------------------------------------
    # Motion blur
    # --------------------------------------------------

    kernel = np.zeros((21, 21), np.float32)

    kernel[10, :] = 1

    M = cv2.getRotationMatrix2D(
        (10.5, 10.5),
        wind,
        1,
    )

    kernel = cv2.warpAffine(
        kernel,
        M,
        (21, 21),
    )

    kernel /= kernel.sum()

    streaks = cv2.filter2D(
        streaks,
        -1,
        kernel,
    )

    out = cv2.addWeighted(
        out,
        1.0,
        streaks,
        0.65 * intensity,
        0,
        dtype=cv2.CV_32F,
    )

    # --------------------------------------------------
    # Visibility reduction
    # --------------------------------------------------

    blur = cv2.GaussianBlur(
        out,
        (0, 0),
        sigmaX=3 + 5 * intensity,
    )

    out = cv2.addWeighted(
        out,
        0.82,
        blur,
        0.18,
        0,
        dtype=cv2.CV_32F,
    )

    # --------------------------------------------------
    # Contrast reduction
    # --------------------------------------------------

    mean = cv2.GaussianBlur(
        out,
        (0, 0),
        sigmaX=30,
    )

    out = mean + (
        1 - 0.30 * intensity
    ) * (out - mean)

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
    Realistic turbulent smoke.

    Features
    --------
    • Procedural turbulence
    • Large smoke plumes
    • Soft edges
    • Slight blue-gray tint
    • Atmospheric scattering
    """

    intensity = float(intensity)

    rng = create_rng(seed)

    img = image.astype(np.float32)

    h, w = img.shape[:2]

    # --------------------------------------------------
    # Multi-scale turbulence
    # --------------------------------------------------

    noise1 = rng.normal(
        0,
        1,
        (h, w),
    ).astype(np.float32)

    noise2 = rng.normal(
        0,
        1,
        (h, w),
    ).astype(np.float32)

    noise3 = rng.normal(
        0,
        1,
        (h, w),
    ).astype(np.float32)

    noise1 = cv2.GaussianBlur(
        noise1,
        (0, 0),
        sigmaX=max(h, w) / 4,
    )

    noise2 = cv2.GaussianBlur(
        noise2,
        (0, 0),
        sigmaX=max(h, w) / 10,
    )

    noise3 = cv2.GaussianBlur(
        noise3,
        (0, 0),
        sigmaX=max(h, w) / 25,
    )

    smoke = (
        0.55 * noise1
        + 0.30 * noise2
        + 0.15 * noise3
    )

    smoke -= smoke.min()
    smoke /= smoke.max() + 1e-6

    smoke = np.power(
        smoke,
        1.8,
    )

    smoke *= intensity

    smoke = smoke[:, :, None]

    # --------------------------------------------------
    # Smoke color
    # --------------------------------------------------

    color = np.full_like(
        img,
        (
            180,
            182,
            186,
        ),
        dtype=np.float32,
    )

    out = (
        img * (1 - smoke * 0.55)
        + color * (smoke * 0.55)
    )

    # --------------------------------------------------
    # Reduce local contrast
    # --------------------------------------------------

    blur = cv2.GaussianBlur(
        out,
        (0, 0),
        sigmaX=10,
    )

    out = (
        out * (1 - smoke * 0.35)
        + blur * (smoke * 0.35)
    )

    # --------------------------------------------------
    # Slight desaturation
    # --------------------------------------------------

    gray = cv2.cvtColor(
        out.astype(np.uint8),
        cv2.COLOR_BGR2GRAY,
    ).astype(np.float32)

    gray = gray[:, :, None]

    out = (
        out * (1 - smoke * 0.20)
        + gray * (smoke * 0.20)
    )

    # --------------------------------------------------
    # Soft atmospheric blur
    # --------------------------------------------------

    out = cv2.GaussianBlur(
        out,
        (0, 0),
        sigmaX=1 + 3 * intensity,
    )

    return np.clip(
        out,
        0,
        255,
    ).astype(np.uint8)