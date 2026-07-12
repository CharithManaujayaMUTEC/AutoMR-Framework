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

    images = images.float().to(DEVICE)

    n, c, h, w = images.shape

    depth = _depth_map(h, w)

    out = images * (
        1.0 - 0.15 * intensity
    )

    rain = torch.rand_like(out)

    rain = KF.motion_blur(
        rain,
        kernel_size=21,
        angle=75.0,
        direction=0.0,
    )

    rain = rain * (
        255.0 * intensity
    )

    out = out + rain * 0.35

    out = _apply_lens_droplets(
        out,
        intensity,
    )

    out = KF.gaussian_blur2d(
        out,
        (3, 3),
        (1.2, 1.2),
    )

    return out.clamp(
        0,
        255,
    )

# ==========================================================
# Snow (Batch)
# ==========================================================

def add_snow_batch(
    images,
    intensity=0.5,
):

    images = images.float().to(DEVICE)

    n, c, h, w = images.shape

    depth = _depth_map(h, w)

    density = _density_field(
        images,
        scale=120,
    )

    snow = torch.rand_like(images)

    snow = (snow > (0.998 - intensity * 0.004)).float()

    snow = KF.gaussian_blur2d(
        snow,
        (5, 5),
        (1.5, 1.5),
    )

    airlight = torch.full_like(
        images,
        245.0,
    )

    scene = _atmospheric_scatter(
        images,
        airlight,
        beta=0.22 * intensity,
        depth=depth,
    )

    scene = _reduce_contrast(
        scene,
        0.10 * intensity,
    )

    out = scene + snow * 255.0 * 0.65

    return out.clamp(
        0,
        255,
    )


# ==========================================================
# Fog (Batch)
# ==========================================================

def add_fog_batch(
    images,
    intensity=0.5,
):

    images = images.float().to(DEVICE)

    n, c, h, w = images.shape

    depth = _depth_map(h, w)

    noise = torch.rand(
        (n, 1, h, w),
        device=DEVICE,
    )

    noise = KF.gaussian_blur2d(
        noise,
        (101, 101),
        (32.0, 32.0),
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

    density = (
        0.5
        + 0.5 * noise
    )

    beta = 1.4 * intensity

    transmission = torch.exp(
        -beta
        * depth.unsqueeze(0).unsqueeze(0)
        * density
    )

    atmosphere = torch.full_like(
        images,
        255.0,
    )

    out = (
        images * transmission
        + atmosphere * (1.0 - transmission)
    )

    gray = out.mean(
        dim=1,
        keepdim=True,
    )

    out = (
        out * (1.0 - 0.20 * intensity)
        + gray * (0.20 * intensity)
    )

    blur = KF.gaussian_blur2d(
        out,
        (17, 17),
        (8.0, 8.0),
    )

    far = (
        1.0 - depth
    ).unsqueeze(0).unsqueeze(0)

    out = (
        out * (1.0 - far * 0.25 * intensity)
        + blur * (far * 0.25 * intensity)
    )

    return out.clamp(
        0,
        255,
    )

# ==========================================================
# Haze (Batch)
# ==========================================================

def add_haze_batch(
    images,
    intensity=0.5,
):

    images = images.float().to(DEVICE)

    n, c, h, w = images.shape

    noise = torch.rand(
        (n, 1, h, w),
        device=DEVICE,
    )

    noise = KF.gaussian_blur2d(
        noise,
        (101, 101),
        (40.0, 40.0),
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

    density = (
        0.55
        + 0.45 * noise
    ) * intensity

    atmosphere = torch.tensor(
        [235.0, 238.0, 245.0],
        device=DEVICE,
    ).view(1, 3, 1, 1)

    out = (
        images * (1.0 - density * 0.45)
        + atmosphere * (density * 0.45)
    )

    mean = KF.gaussian_blur2d(
        out,
        (51, 51),
        (25.0, 25.0),
    )

    contrast = (
        1.0
        - 0.30 * intensity
    )

    out = mean + contrast * (out - mean)

    blur = KF.gaussian_blur2d(
        out,
        (9, 9),
        (3.0, 3.0),
    )

    out = (
        out * 0.8
        + blur * 0.2
    )

    gray = out.mean(
        dim=1,
        keepdim=True,
    )

    out = (
        out * (1.0 - 0.12 * intensity)
        + gray * (0.12 * intensity)
    )

    return out.clamp(
        0,
        255,
    )


# ==========================================================
# Dust (Batch)
# ==========================================================

def add_dust_batch(
    images,
    intensity=0.5,
):

    images = images.float().to(DEVICE)

    n, c, h, w = images.shape

    n1 = torch.rand(
        (n, 1, h, w),
        device=DEVICE,
    )

    n2 = torch.rand_like(n1)
    n3 = torch.rand_like(n1)

    n1 = KF.gaussian_blur2d(
        n1,
        (101, 101),
        (35.0, 35.0),
    )

    n2 = KF.gaussian_blur2d(
        n2,
        (51, 51),
        (15.0, 15.0),
    )

    n3 = KF.gaussian_blur2d(
        n3,
        (21, 21),
        (6.0, 6.0),
    )

    density = (
        0.55 * n1
        + 0.30 * n2
        + 0.15 * n3
    )

    density = density - density.amin(
        dim=(-2, -1),
        keepdim=True,
    )

    density = density / (
        density.amax(
            dim=(-2, -1),
            keepdim=True,
        )
        + 1e-6
    )

    density = (
        density ** 1.8
    ) * intensity

    dust_color = torch.tensor(
        [175.0, 170.0, 145.0],
        device=DEVICE,
    ).view(1, 3, 1, 1)

    out = (
        images * (1.0 - density * 0.45)
        + dust_color * (density * 0.45)
    )

    particles = torch.rand_like(out)

    particles = (
        particles > 0.998
    ).float() * 255.0

    particles = KF.gaussian_blur2d(
        particles,
        (7, 7),
        (2.0, 2.0),
    )

    out = out + particles * (
        0.45 * intensity
    )

    blur = KF.gaussian_blur2d(
        out,
        (11, 11),
        (4.0, 4.0),
    )

    out = (
        out * 0.85
        + blur * 0.15
    )

    mean = KF.gaussian_blur2d(
        out,
        (51, 51),
        (25.0, 25.0),
    )

    out = mean + (
        1.0 - 0.25 * intensity
    ) * (out - mean)

    return out.clamp(
        0,
        255,
    )

# ==========================================================
# Sandstorm (Batch)
# ==========================================================

def add_sandstorm_batch(
    images,
    intensity=0.5,
):

    images = images.float().to(DEVICE)

    n, c, h, w = images.shape

    n1 = torch.rand(
        (n, 1, h, w),
        device=DEVICE,
    )

    n2 = torch.rand_like(n1)
    n3 = torch.rand_like(n1)

    n1 = KF.gaussian_blur2d(
        n1,
        (101, 101),
        (40.0, 40.0),
    )

    n2 = KF.gaussian_blur2d(
        n2,
        (51, 51),
        (15.0, 15.0),
    )

    n3 = KF.gaussian_blur2d(
        n3,
        (21, 21),
        (6.0, 6.0),
    )

    density = (
        0.55 * n1 +
        0.30 * n2 +
        0.15 * n3
    )

    density = density - density.amin(
        dim=(-2, -1),
        keepdim=True,
    )

    density = density / (
        density.amax(
            dim=(-2, -1),
            keepdim=True,
        ) + 1e-6
    )

    density = (
        density ** 1.7
    ) * intensity

    sand_color = torch.tensor(
        [175.0, 165.0, 125.0],
        device=DEVICE,
    ).view(1, 3, 1, 1)

    out = (
        images * (1.0 - density * 0.55)
        + sand_color * (density * 0.55)
    )

    streaks = torch.rand_like(out)

    streaks = KF.motion_blur(
        streaks,
        kernel_size=25,
        angle=20.0,
        direction=0.0,
    )

    streaks *= (
        255.0 * intensity
    )

    out += streaks * 0.45

    blur = KF.gaussian_blur2d(
        out,
        (11, 11),
        (5.0, 5.0),
    )

    out = (
        out * 0.82
        + blur * 0.18
    )

    mean = KF.gaussian_blur2d(
        out,
        (61, 61),
        (30.0, 30.0),
    )

    out = mean + (
        1.0 - 0.30 * intensity
    ) * (out - mean)

    return out.clamp(
        0,
        255,
    )


# ==========================================================
# Smoke (Batch)
# ==========================================================

def add_smoke_batch(
    images,
    intensity=0.5,
):

    images = images.float().to(DEVICE)

    n, c, h, w = images.shape

    noise1 = torch.rand(
        (n, 1, h, w),
        device=DEVICE,
    )

    noise2 = torch.rand_like(noise1)
    noise3 = torch.rand_like(noise1)

    noise1 = KF.gaussian_blur2d(
        noise1,
        (101, 101),
        (40.0, 40.0),
    )

    noise2 = KF.gaussian_blur2d(
        noise2,
        (51, 51),
        (15.0, 15.0),
    )

    noise3 = KF.gaussian_blur2d(
        noise3,
        (21, 21),
        (6.0, 6.0),
    )

    smoke = (
        0.55 * noise1 +
        0.30 * noise2 +
        0.15 * noise3
    )

    smoke = smoke - smoke.amin(
        dim=(-2, -1),
        keepdim=True,
    )

    smoke = smoke / (
        smoke.amax(
            dim=(-2, -1),
            keepdim=True,
        ) + 1e-6
    )

    smoke = (
        smoke ** 1.8
    ) * intensity

    color = torch.tensor(
        [180.0, 182.0, 186.0],
        device=DEVICE,
    ).view(1, 3, 1, 1)

    out = (
        images * (1.0 - smoke * 0.55)
        + color * (smoke * 0.55)
    )

    blur = KF.gaussian_blur2d(
        out,
        (21, 21),
        (10.0, 10.0),
    )

    out = (
        out * (1.0 - smoke * 0.35)
        + blur * (smoke * 0.35)
    )

    gray = out.mean(
        dim=1,
        keepdim=True,
    )

    out = (
        out * (1.0 - smoke * 0.20)
        + gray * (smoke * 0.20)
    )

    out = KF.gaussian_blur2d(
        out,
        (9, 9),
        (4.0, 4.0),
    )

    return out.clamp(
        0,
        255,
    )

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