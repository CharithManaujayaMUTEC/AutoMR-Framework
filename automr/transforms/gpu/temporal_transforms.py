import numpy as np

from .utils import create_rng


# ==========================================================
# Identity Sequence
# ==========================================================

def identity_sequence(sequence):
    """
    Returns the original sequence unchanged.
    Used as the temporal baseline.
    """
    return list(sequence)


# ==========================================================
# Random Temporal Window
# ==========================================================

def sample_sequence(
    dataset,
    length=10,
    seed=None,
):
    """
    Randomly samples a contiguous temporal window.

    Controlled parameter
    --------------------
    length : sequence length

    Randomized
    ----------
    • starting frame
    """

    rng = create_rng(seed)

    if len(dataset) <= length:
        return list(dataset)

    start = rng.integers(
        0,
        len(dataset) - length + 1,
    )

    return list(
        dataset[start:start + length]
    )


# ==========================================================
# Consecutive Frame Pair
# ==========================================================

def next_frame_pair(
    dataset,
    idx,
):
    """
    Returns two consecutive frames.
    """

    idx = int(idx)

    if idx < 0:
        idx = 0

    if idx >= len(dataset) - 1:
        return None

    return (
        dataset[idx],
        dataset[idx + 1],
    )

# ==========================================================
# Random Temporal Pair
# ==========================================================

def temporal_pair(
    dataset,
    max_gap=5,
    seed=None,
):
    """
    Returns two temporally nearby frames.

    Controlled parameter
    --------------------
    max_gap : maximum temporal distance

    Randomized
    ----------
    • starting frame
    • temporal gap
    """

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

    return (
        dataset[start],
        dataset[end],
    )


# ==========================================================
# Temporal Skip Sequence
# ==========================================================

def skip_sequence(
    dataset,
    step=2,
):
    """
    Samples frames using a fixed interval.

    Controlled parameter
    --------------------
    step : frame interval
    """

    step = max(
        1,
        int(step),
    )

    return list(
        dataset[::step]
    )


# ==========================================================
# Temporal Jitter Sequence
# ==========================================================

def jitter_sequence(
    dataset,
    max_offset=2,
    seed=None,
):
    """
    Creates a sequence with small random temporal jitter.

    Controlled parameter
    --------------------
    max_offset : maximum frame displacement

    Randomized
    ----------
    • frame offsets
    """

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

        output.append(
            dataset[idx]
        )

    return output