import numpy as np
import torch

from automr.transforms.backend import DEVICE


# ==========================================================
# NumPy -> Backend
# ==========================================================

def to_backend(image):
    """
    Converts NumPy image to backend tensor.
    """

    if isinstance(image, torch.Tensor):
        return image.to(DEVICE)

    return (
        torch.from_numpy(image)
        .permute(2, 0, 1)
        .float()
        .to(DEVICE)
    )


# ==========================================================
# Backend -> NumPy
# ==========================================================

def from_backend(image):
    """
    Converts backend tensor back to NumPy.
    """

    if isinstance(image, torch.Tensor):
        return (
            image
            .clamp(0, 255)
            .permute(1, 2, 0)
            .byte()
            .cpu()
            .numpy()
        )

    return image.astype(np.uint8)


# ==========================================================
# Ensure NumPy
# ==========================================================

def ensure_numpy(image):
    """
    Always returns NumPy image.
    """

    return from_backend(image)


# ==========================================================
# Ensure Backend Tensor
# ==========================================================

def ensure_tensor(image):
    """
    Always returns backend tensor.
    """

    return to_backend(image)