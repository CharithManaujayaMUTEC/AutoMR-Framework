# ==========================================================
# automr/transforms/gpu/behavioral_transforms.py
# GPU VERSION
# ==========================================================

import torch
import kornia.filters as KF

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def _to_tensor(image):
    if isinstance(image, torch.Tensor):
        x = image

    else:
        x = torch.from_numpy(image)

    if x.ndim == 3:
        # HWC
        if x.shape[-1] in (1, 3):
            x = x.permute(2, 0, 1)

        # CHW
        elif x.shape[0] in (1, 3):
            pass

        else:
            raise ValueError(f"Unexpected image shape: {tuple(x.shape)}")

        x = x.unsqueeze(0)

    elif x.ndim == 4:
        # NHWC
        if x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2)

        # NCHW
        elif x.shape[1] in (1, 3):
            pass

        else:
            raise ValueError(f"Unexpected batch shape: {tuple(x.shape)}")

    else:
        raise ValueError(f"Expected 3D or 4D image, got {x.ndim}D")

    return x.float().to(DEVICE)

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

def reduce_visibility(image, factor=0.5, **kwargs):
    x = _to_tensor(image)
    y = reduce_visibility_batch(x, factor=factor)

    return (
        y.squeeze(0)
        .permute(1, 2, 0)
        .clamp(0, 255)
        .byte()
        .cpu()
        .numpy()
    )


def darken(image, factor=0.5, **kwargs):
    x = _to_tensor(image)
    y = darken_batch(x, factor=factor)

    return (
        y.squeeze(0)
        .permute(1, 2, 0)
        .clamp(0, 255)
        .byte()
        .cpu()
        .numpy()
    )