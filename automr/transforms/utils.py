import hashlib
import cv2
import numpy as np


# ==========================================================
# Random Number Generator
# ==========================================================

def create_rng(
    seed=None,
    sample_id=None,
    mr_name=None,
    intensity=None,
):
    """
    Creates a reproducible NumPy random generator.

    Priority
    --------
    1. Explicit seed
    2. Hash(sample_id, MR, intensity)
    3. Fully random
    """

    # Explicit user seed
    if seed is not None:
        return np.random.default_rng(int(seed))

    # Deterministic AutoMR seed
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

    # Fully random
    return np.random.default_rng()


# ==========================================================
# Random Patch Generator
# ==========================================================

def create_random_patch(
    image_shape,
    rng,
    min_scale=0.10,
    max_scale=0.35,
):
    """
    Returns a random rectangular patch.

    Returns
    -------
    x, y, width, height
    """

    h, w = image_shape

    patch_w = int(
        rng.uniform(min_scale, max_scale) * w
    )

    patch_h = int(
        rng.uniform(min_scale, max_scale) * h
    )

    patch_w = max(20, min(patch_w, w - 1))
    patch_h = max(20, min(patch_h, h - 1))

    x = int(
        rng.integers(
            0,
            max(1, w - patch_w + 1),
        )
    )

    y = int(
        rng.integers(
            0,
            max(1, h - patch_h + 1),
        )
    )

    return x, y, patch_w, patch_h


# ==========================================================
# Random Soft Mask
# ==========================================================

def create_random_mask(
    shape,
    rng,
    min_regions=1,
    max_regions=3,
    min_scale=0.30,
    max_scale=1.00,
    blur_choices=(21, 31, 41),
):
    """
    Creates a smooth random alpha mask.
    """

    h, w = shape

    alpha = np.zeros(
        (h, w),
        dtype=np.float32,
    )

    n_regions = int(
        rng.integers(
            min_regions,
            max_regions + 1,
        )
    )

    for _ in range(n_regions):

        axis_x = int(
            rng.uniform(min_scale, max_scale)
            * w
            / 2
        )

        axis_y = int(
            rng.uniform(min_scale, max_scale)
            * h
            / 2
        )

        center = (
            int(rng.integers(0, w)),
            int(rng.integers(0, h)),
        )

        angle = float(
            rng.uniform(0, 360)
        )

        mask = np.zeros(
            (h, w),
            dtype=np.uint8,
        )

        cv2.ellipse(
            mask,
            center,
            (
                max(5, axis_x),
                max(5, axis_y),
            ),
            angle,
            0,
            360,
            255,
            -1,
        )

        blur = int(
            rng.choice(blur_choices)
        )

        if blur % 2 == 0:
            blur += 1

        mask = cv2.GaussianBlur(
            mask,
            (blur, blur),
            0,
        )

        alpha += (
            mask.astype(np.float32)
            / 255.0
        )

    return np.clip(
        alpha,
        0,
        1,
    )


# ==========================================================
# Alpha Blending
# ==========================================================

def blend_images(
    original,
    transformed,
    alpha,
):
    """
    Alpha blends two images.

    Parameters
    ----------
    original : ndarray
    transformed : ndarray
    alpha : ndarray
        HxW or HxWx1 mask
    """

    original = original.astype(np.float32)
    transformed = transformed.astype(np.float32)

    if alpha.ndim == 2:
        alpha = alpha[:, :, None]

    return (
        original * (1.0 - alpha)
        + transformed * alpha
    )

# ==========================================================
# Random Motion Blur Kernel
# ==========================================================

def random_motion_kernel(
    ksize=15,
    angle=None,
    rng=None,
):
    """
    Generates a normalized random motion blur kernel.

    Parameters
    ----------
    ksize : int
        Kernel size (odd number)

    angle : float or None
        Motion direction in degrees.
        If None, a random angle is used.

    rng : numpy.random.Generator
        Random generator.
    """

    if rng is None:
        rng = np.random.default_rng()

    ksize = max(3, int(ksize))

    if ksize % 2 == 0:
        ksize += 1

    if angle is None:
        angle = float(rng.uniform(0, 180))

    kernel = np.zeros((ksize, ksize), dtype=np.float32)

    kernel[ksize // 2, :] = 1.0

    M = cv2.getRotationMatrix2D(
        (ksize / 2, ksize / 2),
        angle,
        1.0,
    )

    kernel = cv2.warpAffine(
        kernel,
        M,
        (ksize, ksize),
        flags=cv2.INTER_LINEAR,
    )

    s = kernel.sum()

    if s > 0:
        kernel /= s
    else:
        kernel[ksize // 2, :] = 1.0 / ksize

    return kernel