import torch
import kornia
import kornia.filters as KF
import kornia.geometry.transform as KG

# ==========================================================
# Backend Selection
# ==========================================================

USE_CUDA = torch.cuda.is_available()

DEVICE = torch.device(
    "cuda" if USE_CUDA else "cpu"
)

# ==========================================================
# Information
# ==========================================================

def backend_name():
    return DEVICE.type


def is_cuda():
    return USE_CUDA


# ==========================================================
# Tensor Helpers
# ==========================================================

def to_device(x):

    if isinstance(x, torch.Tensor):
        return x.to(
            DEVICE,
            non_blocking=True,
        )

    return (
        torch.from_numpy(x)
        .to(
            DEVICE,
            non_blocking=True,
        )
    )


def synchronize():

    if USE_CUDA:
        torch.cuda.synchronize()


# ==========================================================
# GPU Memory
# ==========================================================

def clear_cache():

    if USE_CUDA:
        torch.cuda.empty_cache()


# ==========================================================
# Batch Helpers
# ==========================================================

def stack(images):

    tensors = []

    for img in images:

        if not isinstance(img, torch.Tensor):

            img = (
                torch.from_numpy(img)
                .permute(2, 0, 1)
                .float()
            )

        tensors.append(img)

    return torch.stack(
        tensors,
        dim=0,
    ).to(
        DEVICE,
        non_blocking=True,
    )


def unstack(batch):

    return [
        img
        .clamp(0, 255)
        .permute(1, 2, 0)
        .byte()
        .cpu()
        .numpy()
        for img in batch
    ]


# ==========================================================
# Exports
# ==========================================================

__all__ = [
    "DEVICE",
    "USE_CUDA",
    "KF",
    "KG",
    "backend_name",
    "is_cuda",
    "to_device",
    "stack",
    "unstack",
    "clear_cache",
    "synchronize",
]