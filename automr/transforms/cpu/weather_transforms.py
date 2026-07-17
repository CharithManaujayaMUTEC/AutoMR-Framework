import cv2
import numpy as np

from ..utils import (
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


# ----------------------------------------------------------
# Directional motion blur kernel
# ----------------------------------------------------------

def _motion_blur_kernel(length, angle_deg):
    """
    Build a 1-D motion blur kernel of given length
    at the given angle (degrees from vertical).
    """

    size = int(length) | 1          # ensure odd
    size = max(3, size)
    kernel = np.zeros((size, size), np.float32)
    kernel[size // 2, :] = 1.0

    M = cv2.getRotationMatrix2D(
        (size / 2 - 0.5, size / 2 - 0.5),
        angle_deg,
        1.0,
    )

    kernel = cv2.warpAffine(kernel, M, (size, size))
    total  = kernel.sum()

    if total > 0:
        kernel /= total

    return kernel


# ==========================================================
# Camera Lens Effects
# ==========================================================

def _apply_lens_droplets(
    image,
    rng,
    intensity,
    tint=(255, 255, 255),
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

    alpha = np.zeros((h, w), np.float32)

    blurred = cv2.GaussianBlur(
        img,
        (0, 0),
        sigmaX=8,
    )

    out = img.copy()

    droplets = int(
        rng.uniform(10, 80) * intensity
    )

    for _ in range(droplets):

        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))

        r = rng.uniform(6, 35) * (0.5 + intensity)

        mask = np.zeros((h, w), np.float32)

        cv2.circle(
            mask,
            (cx, cy),
            int(r),
            1,
            -1,
        )

        mask = cv2.GaussianBlur(
            mask,
            (0, 0),
            sigmaX=r * 0.25,
        )

        alpha = np.maximum(alpha, mask * 0.55)

        hx = int(cx - r * 0.3)
        hy = int(cy - r * 0.3)

        cv2.circle(
            out,
            (hx, hy),
            max(1, int(r * 0.18)),
            tint,
            -1,
        )

    out = (
        out * (1 - alpha[:, :, None])
        + blurred * alpha[:, :, None]
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
    • Perspective rain  (far streaks shorter + thinner)
    • Depth-weighted streak brightness
    • Directional motion blur on the streak layer
    • Lens droplets with refraction warp
    • Wet-road darkening on the bottom third
    • Atmospheric darkening + slight blue-cool tint
    """

    intensity = float(intensity)
    rng       = create_rng(seed)

    img = image.astype(np.float32)
    h, w = img.shape[:2]

    depth = _depth_map(h, w)

    # --------------------------------------------------
    # Atmospheric darkening + cool-blue tint
    # --------------------------------------------------

    out = img * (1.0 - 0.18 * intensity)

    out[:, :, 0] = np.clip(
        out[:, :, 0] + 6 * intensity, 0, 255
    )   # slight blue push (BGR channel 0)

    # --------------------------------------------------
    # Wet-road effect  —  darken + desaturate lower third
    # --------------------------------------------------

    road_y = int(h * 0.68)

    road_region = out[road_y:, :].copy()

    gray_road = cv2.cvtColor(
        road_region.astype(np.uint8),
        cv2.COLOR_BGR2GRAY,
    ).astype(np.float32)[:, :, None]

    road_region = (
        road_region * (1.0 - 0.35 * intensity)
        + gray_road  *  0.15 * intensity
    )

    # subtle wet sheen: blend a slightly brightened version
    sheen = np.clip(road_region * 1.12, 0, 255)

    road_region = cv2.addWeighted(
        road_region, 0.80,
        sheen,        0.20,
        0,
        dtype=cv2.CV_32F,
    )

    out[road_y:] = road_region

    # --------------------------------------------------
    # Rain streak layer  (perspective-scaled)
    # --------------------------------------------------

    rain  = np.zeros_like(out)
    wind  = rng.uniform(-20, 20)          # degrees from vertical

    n = int(h * w * (0.0010 + 0.0030 * intensity))

    for _ in range(n):

        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))

        d      = depth[y, x]
        near_f = 1.0 - d              # 0 at horizon, 1 at bottom

        # perspective: near streaks longer + thicker
        length    = int(rng.uniform(6, 14) + near_f * 28 * intensity)
        thickness = max(1, int(rng.uniform(1, 2) + near_f * 1.5))

        angle = wind + rng.normal(0, 4)
        dx    = np.sin(np.deg2rad(angle))
        dy    = np.cos(np.deg2rad(angle))

        x2 = int(x + dx * length)
        y2 = int(y + dy * length)

        # far streaks dimmer
        b = int(rng.uniform(140, 220) * (0.55 + 0.45 * near_f))

        cv2.line(
            rain,
            (x, y),
            (x2, y2),
            (b, b, b),
            thickness,
            cv2.LINE_AA,
        )

    # Directional motion blur aligned to wind angle
    blur_len = max(3, int(8 + 14 * intensity))
    mk       = _motion_blur_kernel(blur_len, angle_deg=wind)
    rain     = cv2.filter2D(rain, -1, mk)

    # Soft Gaussian on top for natural look
    rain = cv2.GaussianBlur(rain, (3, 3), 0)

    out = cv2.addWeighted(
        out,
        1.0,
        rain,
        0.40 * intensity,
        0,
        dtype=cv2.CV_32F,
    )

    # --------------------------------------------------
    # Lens droplets
    # --------------------------------------------------

    droplets = int(20 + intensity * 120)

    for _ in range(droplets):

        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))

        r = int(rng.uniform(3, 18 + 30 * intensity))

        mask = np.zeros((h, w), np.uint8)

        cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)

        blur = max(9, int(r * 2 + 1))

        if blur % 2 == 0:
            blur += 1

        mask = cv2.GaussianBlur(mask, (blur, blur), 0)

        alpha = (
            mask.astype(np.float32) / 255.0
        ) * rng.uniform(0.18, 0.50) * intensity

        scale = rng.uniform(1.01, 1.06)

        M = cv2.getRotationMatrix2D((cx, cy), 0, scale)

        warped = cv2.warpAffine(
            out, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )

        out = (
            out    * (1 - alpha[:, :, None])
            + warped * alpha[:, :, None]
        )

        cv2.circle(out, (cx, cy), r, (255, 255, 255), 1, cv2.LINE_AA)

    # --------------------------------------------------
    # Final soft pass
    # --------------------------------------------------

    out = cv2.GaussianBlur(out, (3, 3), 0)

    return np.clip(out, 0, 255).astype(np.uint8)


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
    • Whole-image snowfall across multiple depth layers
    • Perspective scaling (near flakes large + bright)
    • Bokeh-blur on far flakes (out-of-focus look)
    • Sparkle highlight on near flakes
    • Ground accumulation brightening on bottom band
    • Atmospheric whitening via Koschmieder scattering
    """

    intensity = float(intensity)
    rng       = create_rng(seed)

    img = image.astype(np.float32)
    h, w = img.shape[:2]

    depth = _depth_map(h, w)

    density = _density_field(h, w, rng, scale=120, blur=101)

    snow = np.zeros_like(img)

    layers = 3

    for layer in range(layers):

        count = int(
            h * w * intensity
            * (0.0006 + layer * 0.00045)
        )

        xs = rng.integers(0, w, count)
        ys = rng.integers(0, h, count)

        for x, y in zip(xs, ys):

            if rng.random() > density[y, x]:
                continue

            d      = depth[y, x]
            near_f = 1.0 - d           # 0 = far, 1 = near

            # perspective radius
            radius = max(
                1,
                int(1 + near_f * (3 + layer * 1.5)),
            )

            brightness = int(rng.uniform(210, 255))

            cv2.circle(
                snow,
                (int(x), int(y)),
                radius,
                (brightness, brightness, brightness),
                -1,
                lineType=cv2.LINE_AA,
            )

            # sparkle highlight on near, large flakes
            if near_f > 0.55 and radius >= 3:
                cv2.circle(
                    snow,
                    (int(x) - max(1, radius // 3),
                     int(y) - max(1, radius // 3)),
                    max(1, radius // 3),
                    (255, 255, 255),
                    -1,
                    cv2.LINE_AA,
                )

        # Far flakes get a bokeh blur (larger sigma for more distant layer)
        if layer == 0:
            snow = cv2.GaussianBlur(snow, (0, 0), sigmaX=2.5)
        elif layer == 1:
            snow = cv2.GaussianBlur(snow, (0, 0), sigmaX=1.2)
        # near layer (layer == 2): no blur — sharp flakes in foreground

    # --------------------------------------------------
    # Ground accumulation  —  brighten bottom band
    # --------------------------------------------------

    accum_y = int(h * 0.80)
    band    = out_band = snow[accum_y:].copy()

    accum_white = np.full_like(band, 240, dtype=np.float32)

    snow[accum_y:] = cv2.addWeighted(
        band.astype(np.float32),
        1.0,
        accum_white,
        0.12 * intensity,
        0,
        dtype=cv2.CV_32F,
    )

    # --------------------------------------------------
    # Atmospheric scattering (whitening at distance)
    # --------------------------------------------------

    airlight = np.full_like(img, 245, dtype=np.float32)

    scene = _atmospheric_scatter(
        img,
        airlight,
        beta=0.25 * intensity,
        depth=depth,
    )

    scene = _reduce_contrast(scene, 0.12 * intensity)

    out = cv2.addWeighted(
        scene,
        1.0,
        snow,
        0.70,
        0,
        dtype=cv2.CV_32F,
    )

    return np.clip(out, 0, 255).astype(np.uint8)


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
    • Depth-aware Koschmieder scattering
    • Horizon-concentrated density (fog pools in distance)
    • Patchy turbulence layer for natural variation
    • Soft distant blur (loss of edge sharpness at range)
    • Contrast reduction proportional to depth
    • Slight cool-grey desaturation
    """

    intensity = float(intensity)
    rng       = create_rng(seed)

    img = image.astype(np.float32)
    h, w = img.shape[:2]

    depth = _depth_map(h, w)

    # --------------------------------------------------
    # Horizon density bias
    # Far pixels (high depth value) get more fog density
    # --------------------------------------------------

    horizon_bias = np.power(depth, 0.6)       # stronger near horizon

    noise = rng.normal(0, 1, (h, w)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=max(h, w) / 10)
    noise -= noise.min()
    noise /= noise.max() + 1e-6

    # Combine spatial noise with horizon bias
    density = (
        0.55 * horizon_bias
        + 0.45 * noise
    )

    density = np.clip(density, 0, 1)

    beta = (1.5 + rng.uniform(-0.2, 0.2)) * intensity

    transmission = np.exp(-beta * depth * density)
    transmission = transmission[:, :, None]

    atmosphere = np.full_like(img, 255, dtype=np.float32)

    out = (
        img        * transmission
        + atmosphere * (1.0 - transmission)
    )

    # --------------------------------------------------
    # Soft distant-edge blur  (fog erases fine detail at range)
    # --------------------------------------------------

    far_mask = depth[:, :, None]              # strong at top/horizon

    blur_heavy = cv2.GaussianBlur(out, (0, 0), sigmaX=6 + 10 * intensity)

    out = (
        out        * (1 - far_mask * 0.55 * intensity)
        + blur_heavy * (far_mask  * 0.55 * intensity)
    )

    # --------------------------------------------------
    # Desaturation + cool-grey cast
    # --------------------------------------------------

    gray = cv2.cvtColor(
        out.astype(np.uint8), cv2.COLOR_BGR2GRAY
    ).astype(np.float32)[:, :, None]

    out = (
        out  * (1 - 0.22 * intensity)
        + gray *  0.22 * intensity
    )

    # slight cool tint (blue channel lift in distant regions)
    out[:, :, 0] = np.clip(
        out[:, :, 0] + 8 * intensity * depth, 0, 255
    )

    return np.clip(out, 0, 255).astype(np.uint8)


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
    • Uneven atmospheric scattering via depth-weighted density
    • Chromatic haze shift  (blue channel slightly more scattered)
    • Reduced contrast proportional to haze density
    • Slight blue-white atmospheric tint
    • Soft overall desaturation
    • Distance-progressive blur
    """

    intensity = float(intensity)
    rng       = create_rng(seed)

    img = image.astype(np.float32)
    h, w = img.shape[:2]

    depth = _depth_map(h, w)

    # --------------------------------------------------
    # Large-scale haze density
    # --------------------------------------------------

    noise = rng.normal(0, 1, (h, w)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=max(h, w) / 8)
    noise -= noise.min()
    noise /= noise.max() + 1e-6

    # Weight density toward distant areas
    depth_bias = np.power(depth, 0.5)
    density    = 0.50 * depth_bias + 0.50 * noise
    density    = np.clip(density, 0, 1) * intensity

    density_3  = density[:, :, None]

    # --------------------------------------------------
    # Chromatic scattering  (blue scattered most, red least)
    # --------------------------------------------------

    atm_b = np.full((h, w), 245, np.float32)   # blue-white sky
    atm_g = np.full((h, w), 238, np.float32)
    atm_r = np.full((h, w), 228, np.float32)

    atmosphere = np.stack([atm_b, atm_g, atm_r], axis=2)

    # Each channel has a slightly different scattering coefficient
    alpha_b = np.clip(density * 0.52, 0, 1)[:, :, None]
    alpha_g = np.clip(density * 0.44, 0, 1)[:, :, None]
    alpha_r = np.clip(density * 0.36, 0, 1)[:, :, None]

    alpha_bgr = np.concatenate([alpha_b, alpha_g, alpha_r], axis=2)

    out = img * (1 - alpha_bgr) + atmosphere * alpha_bgr

    # --------------------------------------------------
    # Contrast reduction
    # --------------------------------------------------

    mean = cv2.GaussianBlur(out, (0, 0), sigmaX=25)
    contrast = 1.0 - 0.32 * intensity
    out = mean + contrast * (out - mean)

    # --------------------------------------------------
    # Distance-progressive blur
    # --------------------------------------------------

    far_mask    = depth[:, :, None]
    blur_far    = cv2.GaussianBlur(out, (0, 0), sigmaX=3 + 6 * intensity)

    out = (
        out      * (1 - far_mask * 0.50 * intensity)
        + blur_far * (far_mask  * 0.50 * intensity)
    )

    # --------------------------------------------------
    # Slight overall desaturation
    # --------------------------------------------------

    gray = cv2.cvtColor(
        out.astype(np.uint8), cv2.COLOR_BGR2GRAY
    ).astype(np.float32)[:, :, None]

    out = (
        out  * (1 - 0.14 * intensity)
        + gray *  0.14 * intensity
    )

    return np.clip(out, 0, 255).astype(np.uint8)


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
    • Fine suspended particles with warm brown tint
    • Multi-scale dust density  (ground-level density bias)
    • Ground-hugging effect  (denser near bottom of frame)
    • Floating particle layer with depth-scaled sizes
    • Reduced sharpness proportional to density
    • Slight warm contrast reduction
    """

    intensity = float(intensity)
    rng       = create_rng(seed)

    img = image.astype(np.float32)
    h, w = img.shape[:2]

    out = img.copy()

    depth = _depth_map(h, w)

    # Ground-hugging density bias
    # Dust is heaviest near the ground  (bottom of image = near)
    y_norm = np.linspace(0, 1, h, dtype=np.float32).reshape(h, 1)
    ground_bias = np.power(y_norm, 0.6)           # stronger toward bottom
    ground_bias = np.repeat(ground_bias, w, axis=1)

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
        0.50 * n1
        + 0.30 * n2
        + 0.20 * n3
    )

    density -= density.min()
    density /= density.max() + 1e-6
    density  = np.power(density, 1.7)

    # Combine noise density with ground-hugging bias
    density = 0.60 * density + 0.40 * ground_bias
    density  = np.clip(density, 0, 1) * intensity
    density_3 = density[:, :, None]

    # --------------------------------------------------
    # Warm dust atmosphere
    # --------------------------------------------------

    dust_color = np.full_like(img, (175, 170, 145), dtype=np.float32)

    out = (
        img        * (1 - density_3 * 0.50)
        + dust_color * (density_3  * 0.50)
    )

    # --------------------------------------------------
    # Floating particles (depth-scaled radius)
    # --------------------------------------------------

    particles = np.zeros_like(out)

    count = int(h * w * intensity * 0.0025)

    for _ in range(count):

        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))

        d      = depth[y, x]
        near_f = 1.0 - d

        r = max(1, int(rng.uniform(1, 3) + near_f * 3))

        c = int(rng.uniform(165, 225))

        cv2.circle(
            particles,
            (x, y),
            r,
            (c, c - 10, c - 28),
            -1,
            cv2.LINE_AA,
        )

    particles = cv2.GaussianBlur(particles, (0, 0), sigmaX=1.5)

    out = cv2.addWeighted(
        out,
        1.0,
        particles,
        0.50 * intensity,
        0,
        dtype=cv2.CV_32F,
    )

    # --------------------------------------------------
    # Density-weighted sharpness reduction
    # --------------------------------------------------

    blur = cv2.GaussianBlur(out, (0, 0), sigmaX=2 + 5 * intensity)

    out = (
        out   * (1 - density_3 * 0.45)
        + blur  *   (density_3 * 0.45)
    )

    # --------------------------------------------------
    # Slight contrast reduction
    # --------------------------------------------------

    mean = cv2.GaussianBlur(out, (0, 0), sigmaX=25)

    out = mean + (1 - 0.28 * intensity) * (out - mean)

    return np.clip(out, 0, 255).astype(np.uint8)


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
    • Directional blowing sand with wind angle
    • Ground-level sand density  (densest at bottom, lifted upward)
    • Multi-scale turbulent density field
    • Flying sand streaks with directional motion blur
    • Warm yellow-orange atmospheric tint
    • Progressive visibility reduction with depth
    """

    intensity = float(intensity)
    rng       = create_rng(seed)

    img = image.astype(np.float32)
    h, w = img.shape[:2]

    out = img.copy()

    depth = _depth_map(h, w)

    wind = rng.uniform(-25, 25)

    # Ground-level density:  sand is denser near the ground (bottom of image)
    y_norm = np.linspace(0, 1, h, dtype=np.float32).reshape(h, 1)
    ground_bias = np.power(y_norm, 0.5)
    ground_bias = np.repeat(ground_bias, w, axis=1)

    # --------------------------------------------------
    # Multi-scale turbulent density
    # --------------------------------------------------

    n1 = rng.normal(0, 1, (h, w)).astype(np.float32)
    n2 = rng.normal(0, 1, (h, w)).astype(np.float32)
    n3 = rng.normal(0, 1, (h, w)).astype(np.float32)

    n1 = cv2.GaussianBlur(n1, (0, 0), sigmaX=max(h, w) / 4)
    n2 = cv2.GaussianBlur(n2, (0, 0), sigmaX=max(h, w) / 12)
    n3 = cv2.GaussianBlur(n3, (0, 0), sigmaX=max(h, w) / 28)

    density = (
        0.55 * n1
        + 0.30 * n2
        + 0.15 * n3
    )

    density -= density.min()
    density /= density.max() + 1e-6
    density  = np.power(density, 1.7)

    # Bias toward the ground
    density = 0.55 * density + 0.45 * ground_bias
    density  = np.clip(density, 0, 1) * intensity
    density_3 = density[:, :, None]

    # --------------------------------------------------
    # Warm sand atmosphere  (deeper orange at high density)
    # --------------------------------------------------

    sand_color = np.full_like(out, (45, 80, 162), dtype=np.float32)

    out = (
        out        * (1 - density_3 * 0.60)
        + sand_color * (density_3  * 0.60)
    )

    # --------------------------------------------------
    # Blowing sand streaks
    # --------------------------------------------------

    streaks = np.zeros_like(out)

    count  = int(h * w * intensity * 0.0030)
    theta  = np.deg2rad(wind)
    dx_dir = np.cos(theta)
    dy_dir = np.sin(theta)

    for _ in range(count):

        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))

        d      = depth[y, x]
        near_f = 1.0 - d

        # near streaks longer + brighter
        length = int(rng.uniform(5, 14) + near_f * 22 * intensity)
        color  = int(rng.uniform(160, 215))

        x2 = int(x + dx_dir * length)
        y2 = int(y + dy_dir * length)

        cv2.line(
            streaks,
            (x, y),
            (x2, y2),
            (int(color * 0.38), int(color * 0.58), color),
            1,
            cv2.LINE_AA,
        )

    # --------------------------------------------------
    # Directional motion blur along wind direction
    # --------------------------------------------------

    mk      = _motion_blur_kernel(int(12 + 14 * intensity), angle_deg=wind)
    streaks = cv2.filter2D(streaks, -1, mk)

    out = cv2.addWeighted(
        out,
        1.0,
        streaks,
        0.70 * intensity,
        0,
        dtype=cv2.CV_32F,
    )

    # --------------------------------------------------
    # Density-weighted visibility reduction
    # --------------------------------------------------

    blur = cv2.GaussianBlur(out, (0, 0), sigmaX=3 + 6 * intensity)

    out = (
        out   * (1 - density_3 * 0.40)
        + blur  *   (density_3 * 0.40)
    )

    # --------------------------------------------------
    # Contrast reduction
    # --------------------------------------------------

    mean = cv2.GaussianBlur(out, (0, 0), sigmaX=30)

    out = mean + (1 - 0.32 * intensity) * (out - mean)

    return np.clip(out, 0, 255).astype(np.uint8)


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
    • Procedural multi-scale turbulence
    • Upward-drift density bias  (smoke rises from below)
    • Large smoke plumes with soft edges
    • Depth-weighted density  (closer smoke denser)
    • Blue-grey cool tint
    • Local contrast reduction inside plumes
    • Atmospheric scattering near dense regions
    """

    intensity = float(intensity)
    rng       = create_rng(seed)

    img = image.astype(np.float32)
    h, w = img.shape[:2]

    depth = _depth_map(h, w)

    # Upward-drift bias:  smoke originates near the bottom and rises
    y_norm     = np.linspace(1.0, 0.0, h, dtype=np.float32).reshape(h, 1)
    rise_bias  = np.power(y_norm, 0.5)           # stronger at bottom
    rise_bias  = np.repeat(rise_bias, w, axis=1)

    # --------------------------------------------------
    # Multi-scale turbulence
    # --------------------------------------------------

    noise1 = rng.normal(0, 1, (h, w)).astype(np.float32)
    noise2 = rng.normal(0, 1, (h, w)).astype(np.float32)
    noise3 = rng.normal(0, 1, (h, w)).astype(np.float32)

    noise1 = cv2.GaussianBlur(noise1, (0, 0), sigmaX=max(h, w) / 4)
    noise2 = cv2.GaussianBlur(noise2, (0, 0), sigmaX=max(h, w) / 10)
    noise3 = cv2.GaussianBlur(noise3, (0, 0), sigmaX=max(h, w) / 25)

    smoke = (
        0.55 * noise1
        + 0.30 * noise2
        + 0.15 * noise3
    )

    smoke -= smoke.min()
    smoke /= smoke.max() + 1e-6
    smoke  = np.power(smoke, 1.8)

    # Combine turbulence with upward-drift and depth
    near_bias = 1.0 - depth              # denser smoke in near regions
    smoke = (
        0.50 * smoke
        + 0.30 * rise_bias
        + 0.20 * near_bias
    )

    smoke  = np.clip(smoke, 0, 1) * intensity
    smoke  = smoke[:, :, None]

    # --------------------------------------------------
    # Smoke color  (cool blue-grey)
    # --------------------------------------------------

    color = np.full_like(img, (188, 186, 182), dtype=np.float32)

    out = (
        img   * (1 - smoke * 0.58)
        + color * (smoke * 0.58)
    )

    # --------------------------------------------------
    # Local contrast reduction inside dense plumes
    # --------------------------------------------------

    blur = cv2.GaussianBlur(out, (0, 0), sigmaX=10)

    out = (
        out  * (1 - smoke * 0.38)
        + blur * (smoke * 0.38)
    )

    # --------------------------------------------------
    # Slight desaturation inside smoke
    # --------------------------------------------------

    gray = cv2.cvtColor(
        out.astype(np.uint8), cv2.COLOR_BGR2GRAY
    ).astype(np.float32)[:, :, None]

    out = (
        out  * (1 - smoke * 0.22)
        + gray * (smoke * 0.22)
    )

    # --------------------------------------------------
    # Soft overall atmospheric blur
    # --------------------------------------------------

    out = cv2.GaussianBlur(out, (0, 0), sigmaX=1.0 + 3.5 * intensity)

    return np.clip(out, 0, 255).astype(np.uint8)