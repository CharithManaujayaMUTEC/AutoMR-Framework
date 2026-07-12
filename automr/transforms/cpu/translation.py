import cv2
import numpy as np

from ..utils import (
    create_rng,
    create_random_patch,
    create_random_mask,
    blend_images,
)


def shift_right(
    image,
    pixels=5,
    seed=None,
):
    """
    Localized Spatial Translation.

    Controlled parameter
    --------------------
    pixels : maximum translation magnitude

    Randomized
    ----------
    • number of translated regions
    • region locations
    • region sizes
    • translation direction
    • mask geometry
    • edge softness

    Reproducible when a seed is provided.
    """

    rng = create_rng(seed)

    img = image.astype(np.float32).copy()

    h, w = img.shape[:2]

    pixels = max(1, int(pixels))

    num_regions = rng.integers(3, 9)

    for _ in range(num_regions):

        # ---------------------------------
        # Random patch
        # ---------------------------------
        x, y, patch_w, patch_h = create_random_patch(
            (h, w),
            rng=rng,
            min_scale=0.10,
            max_scale=0.35,
        )

        patch = img[
            y:y + patch_h,
            x:x + patch_w,
        ].copy()

        # ---------------------------------
        # Random translation direction
        # ---------------------------------
        theta = rng.uniform(0.0, 2.0 * np.pi)

        distance = rng.uniform(
            pixels * 0.5,
            pixels,
        )

        dx = int(np.round(distance * np.cos(theta)))
        dy = int(np.round(distance * np.sin(theta)))

        M = np.float32(
            [
                [1, 0, dx],
                [0, 1, dy],
            ]
        )

        translated = cv2.warpAffine(
            patch,
            M,
            (patch_w, patch_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        ).astype(np.float32)

        # ---------------------------------
        # Random soft blending mask
        # ---------------------------------
        mask = create_random_mask(
            (patch_h, patch_w),
            rng=rng,
            min_regions=1,
            max_regions=2,
            min_scale=0.70,
            max_scale=1.00,
            blur_choices=(21, 31, 41, 51),
        )

        blended = blend_images(
            patch,
            translated,
            mask,
        )

        img[
            y:y + patch_h,
            x:x + patch_w,
        ] = blended

    return np.clip(
        img,
        0,
        255,
    ).astype(np.uint8)