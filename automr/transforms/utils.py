import hashlib
import numpy as np


def create_rng(
    seed=None,
    sample_id=None,
    mr_name=None,
    intensity=None,
):
    """
    Creates a NumPy random number generator.

    Priority
    --------
    1. User-provided seed (fully reproducible)
    2. Deterministic hash(sample_id, MR, intensity)
    3. Fully random generator

    Returns
    -------
    numpy.random.Generator
    """

    # --------------------------------------------------
    # Explicit seed (highest priority)
    # --------------------------------------------------
    if seed is not None:
        return np.random.default_rng(int(seed))

    # --------------------------------------------------
    # Deterministic seed for AutoMR
    # --------------------------------------------------
    if (
        sample_id is not None
        and mr_name is not None
        and intensity is not None
    ):

        text = (
            f"{sample_id}|"
            f"{mr_name}|"
            f"{float(intensity):.6f}"
        )

        seed = int(
            hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()[:8],
            16,
        )

        return np.random.default_rng(seed)

    # --------------------------------------------------
    # Fully random
    # --------------------------------------------------
    return np.random.default_rng()