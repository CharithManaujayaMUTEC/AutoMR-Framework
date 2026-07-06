import cv2
import numpy as np


# ==========================================================
# Localized Visibility Reduction
# ==========================================================

def reduce_visibility(
    image,
    factor=0.5,
    min_patches=3,
    max_patches=8,
):
    """
    Localized visibility degradation.

    Controlled parameter
    --------------------
    factor : visibility reduction intensity

    Randomized
    ----------
    • number of degraded regions
    • locations
    • sizes
    • orientations
    • overlap
    • soft edges
    """

    img = image.astype(np.float32).copy()

    h, w = img.shape[:2]

    fog = np.full_like(img, 255)

    factor = np.clip(float(factor), 0.0, 1.5)

    num_regions = np.random.randint(
        min_patches,
        max_patches + 1
    )

    alpha = np.zeros((h, w), dtype=np.float32)

    for _ in range(num_regions):

        axis_x = np.random.randint(
            max(40, w // 12),
            max(100, w // 3)
        )

        axis_y = np.random.randint(
            max(40, h // 12),
            max(100, h // 3)
        )

        center = (
            np.random.randint(0, w),
            np.random.randint(0, h)
        )

        angle = np.random.uniform(0, 360)

        mask = np.zeros((h, w), dtype=np.uint8)

        cv2.ellipse(
            mask,
            center,
            (axis_x, axis_y),
            angle,
            0,
            360,
            255,
            -1
        )

        blur_size = np.random.choice(
            [41, 61, 81]
        )

        mask = cv2.GaussianBlur(
            mask,
            (blur_size, blur_size),
            0
        )

        alpha += mask.astype(np.float32) / 255.0

    alpha = np.clip(alpha, 0, 1)

    alpha *= factor

    result = (
        img * (1 - alpha[:, :, None])
        + fog * alpha[:, :, None]
    )

    return np.clip(result, 0, 255).astype(np.uint8)


# ==========================================================
# Localized Darkness
# ==========================================================

def darken(
    image,
    factor=0.5,
    min_patches=3,
    max_patches=8,
):
    """
    Localized illumination reduction.

    Controlled parameter
    --------------------
    factor : brightness scaling

    Randomized
    ----------
    • number of dark regions
    • locations
    • sizes
    • orientations
    • overlap
    • soft edges
    """

    img = image.astype(np.float32).copy()

    h, w = img.shape[:2]

    factor = np.clip(float(factor), 0.05, 1.0)

    num_regions = np.random.randint(
        min_patches,
        max_patches + 1
    )

    for _ in range(num_regions):

        axis_x = np.random.randint(
            max(40, w // 12),
            max(100, w // 3)
        )

        axis_y = np.random.randint(
            max(40, h // 12),
            max(100, h // 3)
        )

        center = (
            np.random.randint(0, w),
            np.random.randint(0, h)
        )

        angle = np.random.uniform(0, 360)

        mask = np.zeros((h, w), dtype=np.uint8)

        cv2.ellipse(
            mask,
            center,
            (axis_x, axis_y),
            angle,
            0,
            360,
            255,
            -1
        )

        blur_size = np.random.choice(
            [41, 61, 81]
        )

        mask = cv2.GaussianBlur(
            mask,
            (blur_size, blur_size),
            0
        )

        mask = mask.astype(np.float32) / 255.0

        darkened = img * factor

        img = (
            img * (1 - mask[:, :, None])
            + darkened * mask[:, :, None]
        )

    return np.clip(img, 0, 255).astype(np.uint8)