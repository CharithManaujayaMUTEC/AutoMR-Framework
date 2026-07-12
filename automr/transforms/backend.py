"""
backend.py

Automatic backend selection for AutoMR.

Responsibilities
----------------
- Detect CUDA availability
- Select execution device
- Convert NumPy <-> Torch
- Helper utilities shared by GPU transforms
"""

import numpy as np
import torch

# ==========================================================
# Backend Detection
# ==========================================================

USE_CUDA = torch.cuda.is_available()

DEVICE = torch.device(
    "cuda" if USE_CUDA else "cpu"
)


def gpu_available():
    """
    Return True if CUDA is available.
    """
    return USE_CUDA


def get_device():
    """
    Return execution device.
    """
    return DEVICE


# ==========================================================
# Conversion Helpers
# ==========================================================

def numpy_to_tensor(image):
    """
    Convert HWC uint8/float NumPy image
    to BCHW float32 tensor.
    """

    if torch.is_tensor(image):

        if image.ndim == 3:
            image = image.unsqueeze(0)

        return image.to(DEVICE)

    tensor = (
        torch.from_numpy(image)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
    )

    return tensor.to(DEVICE)


def tensor_to_numpy(tensor):
    """
    Convert BCHW tensor
    back to HWC NumPy image.
    """

    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)

    image = (
        tensor
        .detach()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    return image


# ==========================================================
# Utility
# ==========================================================

def ensure_numpy(image):
    """
    Ensure output is NumPy.
    """

    if torch.is_tensor(image):
        return tensor_to_numpy(image)

    return image


def ensure_tensor(image):
    """
    Ensure output is Torch tensor.
    """

    if torch.is_tensor(image):
        return image.to(DEVICE)

    return numpy_to_tensor(image)


# ==========================================================
# Information
# ==========================================================

def backend_name():
    """
    Return backend name.
    """

    return "CUDA" if USE_CUDA else "CPU"


def print_backend():
    """
    Print active backend.
    """

    print("=" * 40)
    print("AutoMR Backend")
    print("=" * 40)
    print(f"Backend : {backend_name()}")
    print(f"Device  : {DEVICE}")

    if USE_CUDA:
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
        print(
            f"Memory  : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print("=" * 40)