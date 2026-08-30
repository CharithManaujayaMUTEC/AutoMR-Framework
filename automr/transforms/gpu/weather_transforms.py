import torch
import torch.nn.functional as F
import kornia.filters as KF
import numpy as np

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================================================
# Shared Utilities
# ==========================================================

def _depth_map(h, w, gamma=2.0):

    y = torch.linspace(
        0.0,
        1.0,
        h,
        device=DEVICE,
    ).view(h, 1)

    depth = (1.0 - y) ** gamma

    return depth.repeat(1, w)


def _density_field(
    batch,
    scale=96,
):

    n, c, h, w = batch.shape

    gh = max(2, h // scale)
    gw = max(2, w // scale)

    noise = torch.rand(
        (n, 1, gh, gw),
        device=DEVICE,
    )

    noise = F.interpolate(
        noise,
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    )

    noise = KF.gaussian_blur2d(
        noise,
        (81, 81),
        (25.0, 25.0),
    )

    noise = noise - noise.amin(
        dim=(-2, -1),
        keepdim=True,
    )

    noise = noise / (
        noise.amax(
            dim=(-2, -1),
            keepdim=True,
        )
        + 1e-6
    )

    return noise


def _atmospheric_scatter(
    image,
    airlight,
    beta,
    depth,
):

    transmission = torch.exp(
        -beta * depth
    ).unsqueeze(0).unsqueeze(0)

    return (
        image * transmission
        + airlight * (1.0 - transmission)
    )


def _reduce_contrast(
    image,
    amount,
):

    mean = image.mean(
        dim=(-2, -1),
        keepdim=True,
    )

    return mean + (image - mean) * (1.0 - amount)


def _color_cast(
    image,
    color,
    strength,
):

    color = torch.tensor(
        color,
        device=DEVICE,
        dtype=torch.float32,
    ).view(1, 3, 1, 1)

    return (
        image * (1.0 - strength)
        + color * strength
    )


# ----------------------------------------------------------
# Ground-hugging bias
# Stronger weight toward bottom of image (y=1 → ground)
# ----------------------------------------------------------

def _ground_bias(h, w, power=0.6):
    """
    Returns (1, 1, h, w) tensor.
    Value 1.0 at bottom row, 0.0 at top row.
    """

    y = torch.linspace(
        0.0,
        1.0,
        h,
        device=DEVICE,
    ).view(1, 1, h, 1)

    return y.pow(power).expand(1, 1, h, w)


# ----------------------------------------------------------
# Upward-drift bias
# Stronger weight toward bottom, fading toward top
# (smoke/steam originates near ground and rises)
# ----------------------------------------------------------

def _rise_bias(h, w, power=0.5):
    """
    Returns (1, 1, h, w) tensor.
    Value 1.0 at bottom row, tapering toward top.
    """

    y = torch.linspace(
        1.0,
        0.0,
        h,
        device=DEVICE,
    ).view(1, 1, h, 1)

    return y.pow(power).expand(1, 1, h, w)


# ----------------------------------------------------------
# Horizon-concentration bias
# Far pixels (top) get higher weight — used for fog / haze
# ----------------------------------------------------------

def _horizon_bias(depth, power=0.6):
    """
    Returns (h, w) tensor.  Peaks at top (far/horizon).
    """

    return depth.pow(power)


# ==========================================================
# Lens Droplets
# ==========================================================

def _apply_lens_droplets(
    image,
    intensity,
):

    blur = KF.gaussian_blur2d(
        image,
        (17, 17),
        (8.0, 8.0),
    )

    alpha = torch.rand(
        (
            image.size(0),
            1,
            image.size(2),
            image.size(3),
        ),
        device=DEVICE,
    )

    alpha = KF.gaussian_blur2d(
        alpha,
        (81, 81),
        (25.0, 25.0),
    )

    alpha = alpha * (
        intensity * 0.55
    )

    return (
        image * (1.0 - alpha)
        + blur * alpha
    )


# ==========================================================
# Rain (Batch)
# ==========================================================

def add_rain_batch(
    images,
    intensity=0.5,
):
    """
    Realistic dashcam rain.

    Upgrades
    --------
    • Perspective streak layer  (near = longer/brighter, far = dim/short)
    • Directional motion blur aligned to wind angle
    • Wet-road darkening on lower third
    • Cool blue-grey atmospheric tint
    • Lens droplets + refraction
    """

    images = images.float().to(DEVICE)

    n, c, h, w = images.shape

    depth = _depth_map(h, w)      # (h, w)  1=far/top  0=near/bottom

    # near_factor: 0 at top (far), 1 at bottom (near)
    near_f = (
        1.0 - depth
    ).unsqueeze(0).unsqueeze(0)   # (1,1,h,w)

    # --------------------------------------------------
    # Atmospheric darkening + cool-blue tint
    # --------------------------------------------------

    out = images * (1.0 - 0.18 * intensity)

    # Lift blue channel (BGR: channel index 0)
    out[:, 0] = (out[:, 0] + 6.0 * intensity).clamp(0, 255)

    # --------------------------------------------------
    # Wet-road darkening  —  lower 32 % of frame
    # --------------------------------------------------

    road_y = int(h * 0.68)

    road = out[:, :, road_y:, :]

    # Darken + desaturate
    gray_road = road.mean(dim=1, keepdim=True)

    road = (
        road * (1.0 - 0.35 * intensity)
        + gray_road * 0.15 * intensity
    )

    # Subtle wet sheen
    sheen = (road * 1.12).clamp(0, 255)

    road = road * 0.80 + sheen * 0.20

    out[:, :, road_y:, :] = road

    # --------------------------------------------------
    # Perspective streak layer
    # --------------------------------------------------

    rain = torch.rand_like(out)

    # Scale streaks: near rows → high value, far rows → low value
    rain = rain * (0.40 + 0.60 * near_f)

    # Directional motion blur simulating angled rainfall
    rain = KF.motion_blur(
        rain,
        kernel_size=max(3, int(17 + 18 * intensity)) | 1,
        angle=80.0,        # close to vertical
        direction=0.3,     # slight diagonal for wind effect
    )

    rain = rain * (255.0 * 0.35 * intensity)

    # Perspective brightness: near streaks visibly brighter
    rain = rain * (0.55 + 0.45 * near_f)

    out = (out + rain).clamp(0, 255)

    # --------------------------------------------------
    # Lens droplets
    # --------------------------------------------------

    out = _apply_lens_droplets(out, intensity)

    # --------------------------------------------------
    # Final soft pass
    # --------------------------------------------------

    out = KF.gaussian_blur2d(out, (3, 3), (1.2, 1.2))

    return out.clamp(0, 255)


# ==========================================================
# Snow (Batch)
# ==========================================================

def add_snow_batch(
    images,
    intensity=0.5,
):
    """
    Physically-inspired snowfall.

    Upgrades
    --------
    • Three depth layers with per-layer bokeh blur
    • Near flakes sharp, far flakes soft (out-of-focus)
    • Perspective radius scaling
    • Ground-accumulation brightening on bottom band
    • Atmospheric scattering + contrast reduction
    """

    images = images.float().to(DEVICE)

    n, c, h, w = images.shape

    depth = _depth_map(h, w)       # (h, w)

    density = _density_field(images, scale=120)   # (n,1,h,w)

    near_f = (
        1.0 - depth
    ).unsqueeze(0).unsqueeze(0)    # (1,1,h,w)

    # --------------------------------------------------
    # Three layered snow passes
    # --------------------------------------------------

    snow = torch.zeros_like(images)

    for layer in range(3):

        threshold = 0.998 - intensity * (0.004 + layer * 0.003)

        flakes = (
            torch.rand(n, 1, h, w, device=DEVICE) > threshold
        ).float()

        # Perspective brightness + size proxy via blur sigma
        flakes = flakes * (
            210.0 + 45.0 * near_f
        )

        # Bokeh by layer: far=heavy blur, mid=light, near=sharp
        if layer == 0:
            # far — large soft circles
            flakes = KF.gaussian_blur2d(flakes, (9, 9), (3.5, 3.5))
        elif layer == 1:
            # mid
            flakes = KF.gaussian_blur2d(flakes, (5, 5), (1.8, 1.8))
        else:
            # near — tiny sharp Gaussian (just anti-alias)
            flakes = KF.gaussian_blur2d(flakes, (3, 3), (0.8, 0.8))

        snow = snow + flakes.expand_as(images)

    # --------------------------------------------------
    # Sparkle on near pixels  (channel-0 highlight lift)
    # --------------------------------------------------

    sparkle_mask = (near_f > 0.55).float()

    snow[:, 0] = (snow[:, 0] + 30.0 * sparkle_mask.squeeze(1)).clamp(0, 255)

    # --------------------------------------------------
    # Ground accumulation  —  brighten bottom band
    # --------------------------------------------------

    accum_y = int(h * 0.80)

    snow[:, :, accum_y:, :] = (
        snow[:, :, accum_y:, :] + 28.0 * intensity
    ).clamp(0, 255)

    # --------------------------------------------------
    # Atmospheric scattering
    # --------------------------------------------------

    airlight = torch.full_like(images, 245.0)

    scene = _atmospheric_scatter(
        images,
        airlight,
        beta=0.25 * intensity,
        depth=depth,
    )

    scene = _reduce_contrast(scene, 0.12 * intensity)

    out = (scene + snow * 0.70).clamp(0, 255)

    return out.clamp(0, 255)


# ==========================================================
# Fog (Batch)
# ==========================================================

def add_fog_batch(
    images,
    intensity=0.5,
):
    """
    Realistic atmospheric fog.

    Upgrades
    --------
    • Horizon-concentrated density  (fog pools at distance)
    • Distance-progressive edge blur  (detail loss at range)
    • Cool-grey tint on distant regions
    • Desaturation proportional to depth
    """

    images = images.float().to(DEVICE)

    n, c, h, w = images.shape

    depth = _depth_map(h, w)      # (h,w)

    # Horizon bias: fog thickest near the top/horizon
    h_bias = _horizon_bias(depth, power=0.6).unsqueeze(0).unsqueeze(0)  # (1,1,h,w)

    noise = torch.rand(n, 1, h, w, device=DEVICE)

    noise = KF.gaussian_blur2d(noise, (101, 101), (32.0, 32.0))

    noise = noise - noise.amin(dim=(-2, -1), keepdim=True)
    noise = noise / (noise.amax(dim=(-2, -1), keepdim=True) + 1e-6)

    # Combine horizon bias with spatial noise
    density = (
        0.55 * h_bias
        + 0.45 * noise
    ).clamp(0, 1)

    beta = 1.5 * intensity

    transmission = torch.exp(
        -beta
        * depth.unsqueeze(0).unsqueeze(0)
        * density
    )

    atmosphere = torch.full_like(images, 255.0)

    out = (
        images * transmission
        + atmosphere * (1.0 - transmission)
    )

    # --------------------------------------------------
    # Distance-progressive edge blur  (detail loss at range)
    # --------------------------------------------------

    far_mask = depth.unsqueeze(0).unsqueeze(0)   # strong at top

    blur_heavy = KF.gaussian_blur2d(
        out,
        (21, 21),
        (7.0 + 9.0 * intensity, 7.0 + 9.0 * intensity),
    )

    out = (
        out        * (1.0 - far_mask * 0.55 * intensity)
        + blur_heavy * (far_mask      * 0.55 * intensity)
    )

    # --------------------------------------------------
    # Desaturation + cool-grey tint
    # --------------------------------------------------

    gray = out.mean(dim=1, keepdim=True)

    out = (
        out  * (1.0 - 0.22 * intensity)
        + gray *  0.22 * intensity
    )

    # Cool tint: lift blue (channel 0) in far regions
    out[:, 0] = (
        out[:, 0] + 8.0 * intensity * depth.unsqueeze(0)
    ).clamp(0, 255)

    return out.clamp(0, 255)


# ==========================================================
# Haze (Batch)
# ==========================================================

def add_haze_batch(
    images,
    intensity=0.5,
):
    """
    Realistic atmospheric haze.

    Upgrades
    --------
    • Chromatic scattering  (blue scattered more than red)
    • Depth-weighted density  (haze thickens at distance)
    • Distance-progressive blur
    • Slight desaturation inside dense haze
    """

    images = images.float().to(DEVICE)

    n, c, h, w = images.shape

    depth = _depth_map(h, w)      # (h,w)

    depth_bias = depth.pow(0.5).unsqueeze(0).unsqueeze(0)   # (1,1,h,w)

    noise = torch.rand(n, 1, h, w, device=DEVICE)

    noise = KF.gaussian_blur2d(noise, (101, 101), (40.0, 40.0))

    noise = noise - noise.amin(dim=(-2, -1), keepdim=True)
    noise = noise / (noise.amax(dim=(-2, -1), keepdim=True) + 1e-6)

    # Blend depth-bias with spatial noise
    density = (
        0.50 * depth_bias
        + 0.50 * noise
    ).clamp(0, 1) * intensity

    # --------------------------------------------------
    # Chromatic scattering  (Rayleigh: blue > green > red)
    # --------------------------------------------------

    atm_b = torch.full((n, 1, h, w), 245.0, device=DEVICE)
    atm_g = torch.full((n, 1, h, w), 238.0, device=DEVICE)
    atm_r = torch.full((n, 1, h, w), 228.0, device=DEVICE)

    atmosphere = torch.cat([atm_b, atm_g, atm_r], dim=1)   # (n,3,h,w)

    alpha_b = (density * 0.52).clamp(0, 1)
    alpha_g = (density * 0.44).clamp(0, 1)
    alpha_r = (density * 0.36).clamp(0, 1)

    alpha_bgr = torch.cat([alpha_b, alpha_g, alpha_r], dim=1)   # (n,3,h,w)

    out = images * (1.0 - alpha_bgr) + atmosphere * alpha_bgr

    # --------------------------------------------------
    # Contrast reduction
    # --------------------------------------------------

    mean = KF.gaussian_blur2d(out, (51, 51), (25.0, 25.0))

    contrast = 1.0 - 0.32 * intensity

    out = mean + contrast * (out - mean)

    # --------------------------------------------------
    # Distance-progressive blur
    # --------------------------------------------------

    far_mask = depth.unsqueeze(0).unsqueeze(0)

    blur_far = KF.gaussian_blur2d(
        out,
        (15, 15),
        (4.0 + 6.0 * intensity, 4.0 + 6.0 * intensity),
    )

    out = (
        out      * (1.0 - far_mask * 0.50 * intensity)
        + blur_far * (far_mask      * 0.50 * intensity)
    )

    # --------------------------------------------------
    # Slight desaturation
    # --------------------------------------------------

    gray = out.mean(dim=1, keepdim=True)

    out = (
        out  * (1.0 - 0.14 * intensity)
        + gray *  0.14 * intensity
    )

    return out.clamp(0, 255)


# ==========================================================
# Dust (Batch)
# ==========================================================

def add_dust_batch(
    images,
    intensity=0.5,
):
    """
    Realistic airborne dust.

    Upgrades
    --------
    • Ground-hugging density bias  (dust heaviest near ground)
    • Depth-scaled particle size  (near = larger particles)
    • Density-weighted sharpness reduction
    • Warm brown tint
    """

    images = images.float().to(DEVICE)

    n, c, h, w = images.shape

    depth = _depth_map(h, w)

    near_f = (1.0 - depth).unsqueeze(0).unsqueeze(0)   # (1,1,h,w)

    # Ground-hugging bias: dust settles near the ground (bottom of frame)
    g_bias = _ground_bias(h, w, power=0.6)               # (1,1,h,w)

    n1 = torch.rand(n, 1, h, w, device=DEVICE)
    n2 = torch.rand_like(n1)
    n3 = torch.rand_like(n1)

    n1 = KF.gaussian_blur2d(n1, (101, 101), (35.0, 35.0))
    n2 = KF.gaussian_blur2d(n2, (51,  51),  (15.0, 15.0))
    n3 = KF.gaussian_blur2d(n3, (21,  21),  (6.0,  6.0))

    density = (
        0.50 * n1
        + 0.30 * n2
        + 0.20 * n3
    )

    density = density - density.amin(dim=(-2, -1), keepdim=True)
    density = density / (density.amax(dim=(-2, -1), keepdim=True) + 1e-6)
    density = density.pow(1.8)

    # Combine noise density with ground-hugging bias
    density = (
        0.60 * density
        + 0.40 * g_bias
    ).clamp(0, 1) * intensity

    dust_color = torch.tensor(
        [175.0, 170.0, 145.0],
        device=DEVICE,
    ).view(1, 3, 1, 1)

    out = (
        images     * (1.0 - density * 0.50)
        + dust_color * (density      * 0.50)
    )

    # --------------------------------------------------
    # Floating particles  (depth-scaled size via blur)
    # --------------------------------------------------

    particles = (
        torch.rand(n, 1, h, w, device=DEVICE) > 0.9975
    ).float() * 255.0

    # Near particles get a wider blur radius (appear larger)
    particles_sharp = KF.gaussian_blur2d(particles, (5,  5), (1.5, 1.5))
    particles_soft  = KF.gaussian_blur2d(particles, (11, 11), (4.0, 4.0))

    particles_out = (
        particles_sharp * near_f
        + particles_soft  * (1.0 - near_f)
    ).expand_as(images)

    # Warm colour tint on particles (B < G < R shift)
    p_tinted = torch.cat([
        particles_out[:, 0:1] * 0.85,   # blue
        particles_out[:, 1:2] * 0.92,   # green
        particles_out[:, 2:3] * 1.00,   # red — warmest
    ], dim=1)

    out = out + p_tinted * (0.50 * intensity)

    # --------------------------------------------------
    # Density-weighted sharpness reduction
    # --------------------------------------------------

    blur = KF.gaussian_blur2d(
        out,
        (13, 13),
        (3.0 + 5.0 * intensity, 3.0 + 5.0 * intensity),
    )

    out = (
        out  * (1.0 - density * 0.45)
        + blur * (density      * 0.45)
    )

    # --------------------------------------------------
    # Slight contrast reduction
    # --------------------------------------------------

    mean = KF.gaussian_blur2d(out, (51, 51), (25.0, 25.0))

    out = mean + (1.0 - 0.28 * intensity) * (out - mean)

    return out.clamp(0, 255)


# ==========================================================
# Sandstorm (Batch)
# ==========================================================

def add_sandstorm_batch(
    images,
    intensity=0.5,
):
    """
    Realistic sandstorm.

    Upgrades
    --------
    • Ground-level density bias  (sand heaviest at bottom)
    • Perspective streak scaling  (near = longer/brighter)
    • Directional motion blur along true wind angle
    • Warm orange-yellow atmospheric tint
    • Density-weighted visibility reduction
    """

    images = images.float().to(DEVICE)

    n, c, h, w = images.shape

    depth = _depth_map(h, w)

    near_f = (1.0 - depth).unsqueeze(0).unsqueeze(0)   # (1,1,h,w)

    # Ground-level bias: sand rolls along the ground
    g_bias = _ground_bias(h, w, power=0.5)              # (1,1,h,w)

    n1 = torch.rand(n, 1, h, w, device=DEVICE)
    n2 = torch.rand_like(n1)
    n3 = torch.rand_like(n1)

    n1 = KF.gaussian_blur2d(n1, (101, 101), (40.0, 40.0))
    n2 = KF.gaussian_blur2d(n2, (51,  51),  (15.0, 15.0))
    n3 = KF.gaussian_blur2d(n3, (21,  21),  (6.0,  6.0))

    density = (
        0.55 * n1
        + 0.30 * n2
        + 0.15 * n3
    )

    density = density - density.amin(dim=(-2, -1), keepdim=True)
    density = density / (density.amax(dim=(-2, -1), keepdim=True) + 1e-6)
    density = density.pow(1.7)

    # Combine noise with ground-hugging bias
    density = (
        0.55 * density
        + 0.45 * g_bias
    ).clamp(0, 1) * intensity

    sand_color = torch.tensor(
        [45.0, 80.0, 162.0],
        device=DEVICE,
    ).view(1, 3, 1, 1)

    out = (
        images     * (1.0 - density * 0.60)
        + sand_color * (density      * 0.60)
    )

    # --------------------------------------------------
    # Perspective streak layer
    # Near rows → long/bright streaks; far rows → dim/short
    # --------------------------------------------------

    streaks = torch.rand_like(out)

    # Scale streak intensity by proximity
    streaks = streaks * (0.35 + 0.65 * near_f)

    # Directional blur at wind angle (slight diagonal)
    wind_angle = 20.0

    streaks = KF.motion_blur(
        streaks,
        kernel_size=max(3, int(21 + 16 * intensity)) | 1,
        angle=wind_angle,
        direction=0.4,
    )

    streaks = streaks * (255.0 * intensity)

    # Near sand is warmer (redder) than far
    streaks_tinted = torch.cat([
        streaks[:, 0:1] * 0.28,   # blue  — heavily suppressed
        streaks[:, 1:2] * 0.52,   # green — mid
        streaks[:, 2:3] * 1.00,   # red   — dominant
    ], dim=1)

    out = out + streaks_tinted * 0.50

    # --------------------------------------------------
    # Density-weighted visibility reduction
    # --------------------------------------------------

    blur = KF.gaussian_blur2d(
        out,
        (13, 13),
        (4.0 + 6.0 * intensity, 4.0 + 6.0 * intensity),
    )

    out = (
        out  * (1.0 - density * 0.40)
        + blur * (density      * 0.40)
    )

    # --------------------------------------------------
    # Contrast reduction
    # --------------------------------------------------

    mean = KF.gaussian_blur2d(out, (61, 61), (30.0, 30.0))

    out = mean + (1.0 - 0.32 * intensity) * (out - mean)

    return out.clamp(0, 255)


# ==========================================================
# Smoke (Batch)
# ==========================================================

def add_smoke_batch(
    images,
    intensity=0.5,
):
    """
    Realistic turbulent smoke.

    Upgrades
    --------
    • Upward-drift density bias  (smoke rises from below)
    • Near-depth density bias  (closer smoke denser)
    • Multi-scale turbulence with depth blending
    • Cool blue-grey tint
    • Density-weighted local contrast reduction
    • Density-weighted desaturation
    """

    images = images.float().to(DEVICE)

    n, c, h, w = images.shape

    depth = _depth_map(h, w)

    near_f = (1.0 - depth).unsqueeze(0).unsqueeze(0)   # (1,1,h,w)

    # Upward-drift: smoke rises from bottom (near) toward top
    r_bias = _rise_bias(h, w, power=0.5)                # (1,1,h,w)

    noise1 = torch.rand(n, 1, h, w, device=DEVICE)
    noise2 = torch.rand_like(noise1)
    noise3 = torch.rand_like(noise1)

    noise1 = KF.gaussian_blur2d(noise1, (101, 101), (40.0, 40.0))
    noise2 = KF.gaussian_blur2d(noise2, (51,  51),  (15.0, 15.0))
    noise3 = KF.gaussian_blur2d(noise3, (21,  21),  (6.0,  6.0))

    smoke = (
        0.55 * noise1
        + 0.30 * noise2
        + 0.15 * noise3
    )

    smoke = smoke - smoke.amin(dim=(-2, -1), keepdim=True)
    smoke = smoke / (smoke.amax(dim=(-2, -1), keepdim=True) + 1e-6)
    smoke = smoke.pow(1.8)

    # Blend turbulence with upward-drift and near-depth bias
    smoke = (
        0.50 * smoke
        + 0.30 * r_bias
        + 0.20 * near_f
    ).clamp(0, 1) * intensity

    # --------------------------------------------------
    # Smoke colour  (cool blue-grey, slightly differentiated channels)
    # --------------------------------------------------

    color = torch.tensor(
        [188.0, 185.0, 181.0],
        device=DEVICE,
    ).view(1, 3, 1, 1)

    out = (
        images * (1.0 - smoke * 0.58)
        + color  * (smoke      * 0.58)
    )

    # --------------------------------------------------
    # Density-weighted local contrast reduction inside plumes
    # --------------------------------------------------

    blur = KF.gaussian_blur2d(out, (21, 21), (10.0, 10.0))

    out = (
        out  * (1.0 - smoke * 0.38)
        + blur * (smoke      * 0.38)
    )

    # --------------------------------------------------
    # Density-weighted desaturation
    # --------------------------------------------------

    gray = out.mean(dim=1, keepdim=True)

    out = (
        out  * (1.0 - smoke * 0.22)
        + gray * (smoke      * 0.22)
    )

    # --------------------------------------------------
    # Soft atmospheric blur
    # --------------------------------------------------

    out = KF.gaussian_blur2d(
        out,
        (9, 9),
        (1.0 + 3.5 * intensity, 1.0 + 3.5 * intensity),
    )

    return out.clamp(0, 255)


# ==========================================================
# Single-image wrappers
# ==========================================================

def _to_batch(image):

    if isinstance(image, np.ndarray):
        x = torch.from_numpy(image)

    elif torch.is_tensor(image):
        x = image

    else:
        raise TypeError(type(image))

    # HWC -> CHW
    if x.ndim == 3 and x.shape[-1] in (1, 3, 4):
        x = x.permute(2, 0, 1)

    # CHW -> NCHW
    if x.ndim == 3:
        x = x.unsqueeze(0)

    return x.float().to(DEVICE)

def _from_batch(batch):

    img = (
        batch[0]
        .clamp(0, 255)
        .permute(1, 2, 0)
        .byte()
        .cpu()
        .numpy()
    )

    return img


# ==========================================================
# Public API
# ==========================================================

def add_rain(
    image,
    intensity=0.5,
    seed=None,
):
    batch = _to_batch(image)
    out = add_rain_batch(batch, intensity)
    return _from_batch(out)


def add_snow(
    image,
    intensity=0.5,
    seed=None,
):
    batch = _to_batch(image)
    out = add_snow_batch(batch, intensity)
    return _from_batch(out)


def add_fog(
    image,
    intensity=0.5,
    seed=None,
):
    batch = _to_batch(image)
    out = add_fog_batch(batch, intensity)
    return _from_batch(out)


def add_haze(
    image,
    intensity=0.5,
    seed=None,
):
    batch = _to_batch(image)
    out = add_haze_batch(batch, intensity)
    return _from_batch(out)


def add_dust(
    image,
    intensity=0.5,
    seed=None,
):
    batch = _to_batch(image)
    out = add_dust_batch(batch, intensity)
    return _from_batch(out)


def add_sandstorm(
    image,
    intensity=0.5,
    seed=None,
):
    batch = _to_batch(image)
    out = add_sandstorm_batch(batch, intensity)
    return _from_batch(out)


def add_smoke(
    image,
    intensity=0.5,
    seed=None,
):
    batch = _to_batch(image)
    out = add_smoke_batch(batch, intensity)
    return _from_batch(out)


# ==========================================================
# Exported Symbols
# ==========================================================

__all__ = [
    "add_rain",
    "add_snow",
    "add_fog",
    "add_haze",
    "add_dust",
    "add_sandstorm",
    "add_smoke",
    "add_rain_batch",
    "add_snow_batch",
    "add_fog_batch",
    "add_haze_batch",
    "add_dust_batch",
    "add_sandstorm_batch",
    "add_smoke_batch",
]