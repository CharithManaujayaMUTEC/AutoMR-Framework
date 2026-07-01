import random


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

def sample_sequence(dataset, length=10):
    """
    Randomly samples a contiguous temporal window.

    Controlled parameter
    --------------------
    length : sequence length

    Randomized
    ----------
    • starting frame
    """

    if len(dataset) <= length:
        return list(dataset)

    start = random.randint(
        0,
        len(dataset) - length
    )

    return dataset[start:start + length]


# ==========================================================
# Consecutive Frame Pair
# ==========================================================

def next_frame_pair(dataset, idx):
    """
    Returns two consecutive frames.
    """

    idx = int(idx)

    if idx >= len(dataset) - 1:
        return None

    return (
        dataset[idx],
        dataset[idx + 1]
    )


# ==========================================================
# Random Temporal Pair
# ==========================================================

def temporal_pair(
    dataset,
    max_gap=5,
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

    if len(dataset) < 2:
        return None

    start = random.randint(
        0,
        len(dataset) - 2
    )

    gap = random.randint(
        1,
        max_gap
    )

    end = min(
        start + gap,
        len(dataset) - 1
    )

    return (
        dataset[start],
        dataset[end]
    )


# ==========================================================
# Temporal Skip Sequence
# ==========================================================

def skip_sequence(
    dataset,
    step=2,
):
    """
    Samples frames with a fixed temporal skip.

    Controlled parameter
    --------------------
    step : frame interval
    """

    step = max(1, int(step))

    return dataset[::step]


# ==========================================================
# Temporal Jitter Sequence
# ==========================================================

def jitter_sequence(
    dataset,
    max_offset=2,
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

    if len(dataset) == 0:
        return []

    output = []

    for i in range(len(dataset)):

        offset = random.randint(
            -max_offset,
            max_offset
        )

        idx = min(
            max(
                i + offset,
                0
            ),
            len(dataset) - 1
        )

        output.append(dataset[idx])

    return output