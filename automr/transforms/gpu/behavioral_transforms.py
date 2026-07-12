# ==========================================================
# automr/transforms/gpu/behavioral_transforms.py
# GPU VERSION
# ==========================================================

import torch
import kornia.filters as KF

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================================
# Visibility (Batch)
# ==========================================================

def reduce_visibility_batch(
    images,
    factor=0.5,
):
    """
    GPU visibility reduction.

    images : (N,C,H,W)
    """

    images = images.float().to(DEVICE)

    factor = float(max(0.0, min(factor, 1.5)))

    white = torch.full_like(
        images,
        255.0,
    )

    alpha = torch.ones(
        (images.size(0), 1, images.size(2), images.size(3)),
        device=DEVICE,
    )

    alpha = KF.gaussian_blur2d(
        alpha,
        (81, 81),
        (25.0, 25.0),
    )

    alpha = alpha * factor

    output = (
        images * (1.0 - alpha)
        + white * alpha
    )

    return output.clamp(
        0,
        255,
    )


# ==========================================================
# Darkness (Batch)
# ==========================================================

def darken_batch(
    images,
    factor=0.5,
):
    """
    GPU darkness.

    images : (N,C,H,W)
    """

    images = images.float().to(DEVICE)

    factor = float(max(0.05, min(factor, 1.0)))

    alpha = torch.ones(
        (images.size(0), 1, images.size(2), images.size(3)),
        device=DEVICE,
    )

    alpha = KF.gaussian_blur2d(
        alpha,
        (81, 81),
        (25.0, 25.0),
    )

    dark = images * factor

    output = (
        images * (1.0 - alpha)
        + dark * alpha
    )

    return output.clamp(
        0,
        255,
    )


# ==========================================================
# Single-image wrappers
# ==========================================================

def reduce_visibility(
    image,
    factor=0.5,
    **kwargs,
):
    x = (
        torch.from_numpy(image)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(DEVICE)
    )

    y = reduce_visibility_batch(
        x,
        factor=factor,
    )

    return (
        y.squeeze(0)
        .permute(1, 2, 0)
        .byte()
        .cpu()
        .numpy()
    )


def darken(
    image,
    factor=0.5,
    **kwargs,
):
    x = (
        torch.from_numpy(image)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(DEVICE)
    )

    y = darken_batch(
        x,
        factor=factor,
    )

    return (
        y.squeeze(0)
        .permute(1, 2, 0)
        .byte()
        .cpu()
        .numpy()
    )