import numpy as np
import torch
import kornia.geometry.transform as K

from ..backend import DEVICE
from ..utils import create_rng


# ==========================================================
# GPU Global Spatial Translation
# ==========================================================

def shift_right(
    image,
    pixels=5,
    seed=None,
):
    """
    GPU Global Spatial Translation.

    Controlled parameter
    --------------------
    pixels : maximum translation magnitude

    Randomized
    ----------
    • translation direction (360°)
    • translation distance

    Reproducible when a seed is provided.
    """

    rng = create_rng(seed)

    # ---------------------------------
    # Convert image to tensor
    # ---------------------------------
    if isinstance(image, np.ndarray):

        img = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(DEVICE)
        )

    else:

        img = image.unsqueeze(0).to(DEVICE)

    _, _, H, W = img.shape

    pixels = max(1, int(pixels))

    # ---------------------------------
    # Random translation direction
    # ---------------------------------
    angle = rng.uniform(
        0.0,
        2.0 * np.pi,
    )

    distance = rng.uniform(
        pixels * 0.5,
        pixels,
    )

    dx = float(distance * np.cos(angle))
    dy = float(distance * np.sin(angle))

    # ---------------------------------
    # Translation matrix
    # ---------------------------------
    transform = torch.eye(
        3,
        device=DEVICE,
    ).unsqueeze(0)

    transform[:, 0, 2] = dx
    transform[:, 1, 2] = dy

    # ---------------------------------
    # Apply translation
    # ---------------------------------
    translated = K.warp_perspective(
        img,
        transform,
        dsize=(H, W),
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    )

    # ---------------------------------
    # Return translated image
    # ---------------------------------
    return (
        translated[0]
        .clamp(0, 255)
        .permute(1, 2, 0)
        .byte()
        .cpu()
        .numpy()
    )