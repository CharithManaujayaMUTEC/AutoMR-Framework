import cv2
import numpy as np


def shift_right(
    image,
    pixels=5,
    min_patches=3,
    max_patches=8,
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
    • ellipse orientation
    • edge softness
    """

    img = image.astype(np.float32).copy()

    h, w = img.shape[:2]

    pixels = max(1, int(pixels))

    num_regions = np.random.randint(
        min_patches,
        max_patches + 1
    )

    for _ in range(num_regions):

        # -------------------------------------
        # Random region size
        # -------------------------------------
        patch_w = np.random.randint(
            max(40, w // 10),
            max(120, w // 3)
        )

        patch_h = np.random.randint(
            max(40, h // 10),
            max(120, h // 3)
        )

        x = np.random.randint(
            0,
            w - patch_w
        )

        y = np.random.randint(
            0,
            h - patch_h
        )

        patch = img[
            y:y + patch_h,
            x:x + patch_w
        ].copy()

        # -------------------------------------
        # Random translation direction
        # -------------------------------------
        angle = np.random.uniform(
            0,
            360
        )

        dx = int(
            pixels * np.cos(
                np.deg2rad(angle)
            )
        )

        dy = int(
            pixels * np.sin(
                np.deg2rad(angle)
            )
        )

        M = np.float32([
            [1, 0, dx],
            [0, 1, dy]
        ])

        translated = cv2.warpAffine(
            patch,
            M,
            (patch_w, patch_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        # -------------------------------------
        # Soft elliptical mask
        # -------------------------------------
        mask = np.zeros(
            (patch_h, patch_w),
            dtype=np.uint8
        )

        cv2.ellipse(
            mask,
            (patch_w // 2, patch_h // 2),
            (patch_w // 2, patch_h // 2),
            np.random.uniform(0, 360),
            0,
            360,
            255,
            -1
        )

        blur_size = np.random.choice(
            [21, 31, 41, 51]
        )

        mask = cv2.GaussianBlur(
            mask,
            (blur_size, blur_size),
            0
        )

        mask = mask.astype(np.float32) / 255.0

        img[
            y:y + patch_h,
            x:x + patch_w
        ] = (
            patch * (1 - mask[:, :, None]) +
            translated * mask[:, :, None]
        )

    return np.clip(
        img,
        0,
        255
    ).astype(np.uint8)