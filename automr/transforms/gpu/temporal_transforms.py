import numpy as np
import torch

from ..backend import DEVICE
from .utils import create_rng


# ==========================================================
# Identity Sequence
# ==========================================================

def identity_sequence(sequence):
    """
    GPU-compatible identity sequence.
    """

    output = []

    for frame in sequence:

        if isinstance(frame, np.ndarray):

            frame = (
                torch.from_numpy(frame)
                .permute(2, 0, 1)
                .float()
                .to(DEVICE)
            )

        output.append(frame)

    return output


# ==========================================================
# Random Temporal Window
# ==========================================================

def sample_sequence(
    dataset,
    length=10,
    seed=None,
):
    """
    GPU-compatible temporal window.
    """

    rng = create_rng(seed)

    if len(dataset) <= length:

        seq = dataset

    else:

        start = rng.integers(
            0,
            len(dataset) - length + 1,
        )

        seq = dataset[start:start + length]

    output = []

    for frame in seq:

        if isinstance(frame, np.ndarray):

            frame = (
                torch.from_numpy(frame)
                .permute(2, 0, 1)
                .float()
                .to(DEVICE)
            )

        output.append(frame)

    return output


# ==========================================================
# Consecutive Frame Pair
# ==========================================================

def next_frame_pair(
    dataset,
    idx,
):

    idx = int(idx)

    if idx < 0:
        idx = 0

    if idx >= len(dataset) - 1:
        return None

    f1 = dataset[idx]
    f2 = dataset[idx + 1]

    if isinstance(f1, np.ndarray):
        f1 = (
            torch.from_numpy(f1)
            .permute(2, 0, 1)
            .float()
            .to(DEVICE)
        )

    if isinstance(f2, np.ndarray):
        f2 = (
            torch.from_numpy(f2)
            .permute(2, 0, 1)
            .float()
            .to(DEVICE)
        )

    return (
        f1,
        f2,
    )


# ==========================================================
# Random Temporal Pair
# ==========================================================

def temporal_pair(
    dataset,
    max_gap=5,
    seed=None,
):

    rng = create_rng(seed)

    if len(dataset) < 2:
        return None

    max_gap = max(1, int(max_gap))

    start = rng.integers(
        0,
        len(dataset) - 1,
    )

    gap = rng.integers(
        1,
        max_gap + 1,
    )

    end = min(
        start + gap,
        len(dataset) - 1,
    )

    f1 = dataset[start]
    f2 = dataset[end]

    if isinstance(f1, np.ndarray):
        f1 = (
            torch.from_numpy(f1)
            .permute(2, 0, 1)
            .float()
            .to(DEVICE)
        )

    if isinstance(f2, np.ndarray):
        f2 = (
            torch.from_numpy(f2)
            .permute(2, 0, 1)
            .float()
            .to(DEVICE)
        )

    return (
        f1,
        f2,
    )


# ==========================================================
# Temporal Skip Sequence
# ==========================================================

def skip_sequence(
    dataset,
    step=2,
):

    step = max(
        1,
        int(step),
    )

    output = []

    for frame in dataset[::step]:

        if isinstance(frame, np.ndarray):

            frame = (
                torch.from_numpy(frame)
                .permute(2, 0, 1)
                .float()
                .to(DEVICE)
            )

        output.append(frame)

    return output


# ==========================================================
# Temporal Jitter Sequence
# ==========================================================

def jitter_sequence(
    dataset,
    max_offset=2,
    seed=None,
):

    rng = create_rng(seed)

    if len(dataset) == 0:
        return []

    max_offset = max(
        0,
        int(max_offset),
    )

    output = []

    for i in range(len(dataset)):

        offset = rng.integers(
            -max_offset,
            max_offset + 1,
        )

        idx = min(
            max(
                i + offset,
                0,
            ),
            len(dataset) - 1,
        )

        frame = dataset[idx]

        if isinstance(frame, np.ndarray):

            frame = (
                torch.from_numpy(frame)
                .permute(2, 0, 1)
                .float()
                .to(DEVICE)
            )

        output.append(frame)

    return output