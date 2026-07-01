import numpy as np
import cv2

def increase_brightness(
    image,
    factor=1.2,
    min_patches=3,
    max_patches=8,
):
    """
    Localized Random Brightness Transformation.

    Parameters
    ----------
    factor : float
        Brightness intensity multiplier.
        This is the ONLY deterministic parameter.

    Everything else (patch count, location, size,
    orientation and blur) is randomly generated.
    """

    img = image.astype(np.float32).copy()

    h, w = img.shape[:2]

    # Number of bright regions
    num_patches = np.random.randint(
        min_patches,
        max_patches + 1
    )

    for _ in range(num_patches):

        # Random ellipse size
        axis_x = np.random.randint(
            max(10, w // 20),
            max(20, w // 5)
        )

        axis_y = np.random.randint(
            max(10, h // 20),
            max(20, h // 5)
        )

        # Random center
        center = (
            np.random.randint(0, w),
            np.random.randint(0, h)
        )

        # Random orientation
        angle = np.random.uniform(0, 360)

        # Binary mask
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

        # Random edge softness
        blur_size = np.random.choice([21, 31, 41, 51])

        mask = cv2.GaussianBlur(
            mask,
            (blur_size, blur_size),
            0
        )

        mask = mask.astype(np.float32) / 255.0

        # Apply identical brightness intensity
        for c in range(img.shape[2]):
            img[:, :, c] *= (
                1 + (factor - 1) * mask
            )

    return np.clip(
        img,
        0,
        255
    ).astype(np.uint8)

def rotate_small(
    image,
    angle=5,
    min_patches=3,
    max_patches=7,
):
    """
    Spatially Variant Local Rotation.

    Controlled parameter
    --------------------
    angle : rotation angle (degrees)

    Randomized
    ----------
    • number of rotated regions
    • locations
    • region sizes
    • ellipse orientations
    • edge softness
    """

    img = image.astype(np.float32).copy()

    h, w = img.shape[:2]

    num_patches = np.random.randint(
        min_patches,
        max_patches + 1
    )

    for _ in range(num_patches):

        patch_w = np.random.randint(
            max(30, w // 8),
            max(60, w // 3)
        )

        patch_h = np.random.randint(
            max(30, h // 8),
            max(60, h // 3)
        )

        x = np.random.randint(0, w - patch_w)
        y = np.random.randint(0, h - patch_h)

        patch = img[
            y:y + patch_h,
            x:x + patch_w
        ].copy()

        center = (
            patch_w / 2,
            patch_h / 2
        )

        M = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )

        rotated = cv2.warpAffine(
            patch,
            M,
            (patch_w, patch_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

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
            [21, 31, 41]
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
            patch * (1 - mask[:, :, None])
            + rotated * mask[:, :, None]
        )

    return np.clip(
        img,
        0,
        255
    ).astype(np.uint8)

def shift_right(
    image,
    pixels=10,
    min_patches=3,
    max_patches=7,
):
    """
    Spatially Variant Local Translation.

    Controlled parameter
    --------------------
    pixels : translation magnitude

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

    pixels = int(max(1, pixels))

    num_patches = np.random.randint(
        min_patches,
        max_patches + 1
    )

    for _ in range(num_patches):

        patch_w = np.random.randint(
            max(30, w // 8),
            max(70, w // 3)
        )

        patch_h = np.random.randint(
            max(30, h // 8),
            max(70, h // 3)
        )

        x = np.random.randint(0, w - patch_w)
        y = np.random.randint(0, h - patch_h)

        patch = img[
            y:y + patch_h,
            x:x + patch_w
        ].copy()

        # Random translation direction
        dx = np.random.randint(-pixels, pixels + 1)
        dy = np.random.randint(-pixels, pixels + 1)

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
            [21, 31, 41]
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
            patch * (1 - mask[:, :, None])
            + translated * mask[:, :, None]
        )

    return np.clip(
        img,
        0,
        255
    ).astype(np.uint8)

def add_noise(
    image,
    level=15,
    min_patches=4,
    max_patches=10,
):
    """
    Spatially Variant Sensor Noise.

    Controlled parameter
    --------------------
    level : noise intensity

    Randomized
    ----------
    • number of noisy regions
    • locations
    • region sizes
    • ellipse orientations
    • edge softness
    • Gaussian vs Salt-and-Pepper noise
    """

    img = image.astype(np.float32).copy()

    h, w = img.shape[:2]

    level = max(1, float(level))

    num_patches = np.random.randint(
        min_patches,
        max_patches + 1
    )

    for _ in range(num_patches):

        # -----------------------------
        # Random region
        # -----------------------------

        axis_x = np.random.randint(
            max(20, w // 20),
            max(40, w // 4)
        )

        axis_y = np.random.randint(
            max(20, h // 20),
            max(40, h // 4)
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
            [21, 31, 41, 51]
        )

        mask = cv2.GaussianBlur(
            mask,
            (blur_size, blur_size),
            0
        )

        mask = mask.astype(np.float32) / 255.0

        # -----------------------------
        # Random noise model
        # -----------------------------

        if np.random.rand() < 0.7:

            # Gaussian sensor noise

            noise = np.random.normal(
                0,
                level,
                img.shape
            )

            noisy = img + noise

        else:

            # Salt & Pepper

            noisy = img.copy()

            amount = level / 255.0

            coords = (
                np.random.rand(h, w)
                < amount * 0.5
            )

            noisy[coords] = 255

            coords = (
                np.random.rand(h, w)
                < amount * 0.5
            )

            noisy[coords] = 0

        img = (
            img * (1 - mask[:, :, None])
            + noisy * mask[:, :, None]
        )

    return np.clip(
        img,
        0,
        255
    ).astype(np.uint8)

def mirror_image(image, _=None):
    return cv2.flip(image, 1)

def blur(
    image,
    k=11,
    min_patches=3,
    max_patches=8,
):
    """
    Spatially Variant Blur Transformation.

    Only the blur kernel size (k) is deterministic.
    Everything else is randomly generated.

    Randomized:
        • Number of blur regions
        • Region locations
        • Region sizes
        • Region orientations
        • Blur type (Gaussian / Motion)
        • Motion direction
        • Edge softness
    """

    img = image.astype(np.float32).copy()

    h, w = img.shape[:2]

    k = max(3, int(k))

    if k % 2 == 0:
        k += 1

    num_patches = np.random.randint(
        min_patches,
        max_patches + 1
    )

    for _ in range(num_patches):

        # ---------------------------------
        # Random ellipse
        # ---------------------------------

        axis_x = np.random.randint(
            max(15, w // 20),
            max(30, w // 4)
        )

        axis_y = np.random.randint(
            max(15, h // 20),
            max(30, h // 4)
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
            [21, 31, 41, 51]
        )

        mask = cv2.GaussianBlur(
            mask,
            (blur_size, blur_size),
            0
        )

        mask = mask.astype(np.float32) / 255.0

        # ---------------------------------
        # Random blur type
        # ---------------------------------

        if np.random.rand() < 0.5:

            # Gaussian blur
            blurred = cv2.GaussianBlur(
                img,
                (k, k),
                0
            )

        else:

            # Motion blur
            kernel = np.zeros((k, k))

            kernel[k // 2, :] = np.ones(k)

            rotation = cv2.getRotationMatrix2D(
                (k / 2, k / 2),
                np.random.uniform(0, 180),
                1
            )

            kernel = cv2.warpAffine(
                kernel,
                rotation,
                (k, k)
            )

            kernel /= np.sum(kernel)

            blurred = cv2.filter2D(
                img,
                -1,
                kernel
            )

        img = (
            img * (1 - mask[:, :, None])
            + blurred * mask[:, :, None]
        )

    return np.clip(
        img,
        0,
        255
    ).astype(np.uint8)

def adjust_contrast(
    image,
    factor=1.2,
    min_patches=3,
    max_patches=8,
):
    """
    Localized Adaptive Contrast Transformation.

    Only the contrast factor is fixed.
    Patch count, locations, sizes, orientations and edge
    smoothness are randomly generated for every execution.
    """

    img = image.astype(np.float32).copy()

    h, w = img.shape[:2]

    num_patches = np.random.randint(
        min_patches,
        max_patches + 1
    )

    for _ in range(num_patches):

        # -----------------------------
        # Random ellipse
        # -----------------------------
        axis_x = np.random.randint(
            max(10, w // 20),
            max(20, w // 5)
        )

        axis_y = np.random.randint(
            max(10, h // 20),
            max(20, h // 5)
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

        # -----------------------------
        # Soft edges
        # -----------------------------
        blur_size = np.random.choice([21, 31, 41, 51])

        mask = cv2.GaussianBlur(
            mask,
            (blur_size, blur_size),
            0
        )

        mask = mask.astype(np.float32) / 255.0

        # -----------------------------
        # Local mean
        # -----------------------------
        gray = cv2.cvtColor(
            img.astype(np.uint8),
            cv2.COLOR_RGB2GRAY
        ).astype(np.float32)

        local_mean = cv2.GaussianBlur(
            gray,
            (31, 31),
            0
        )

        local_mean = np.repeat(
            local_mean[:, :, np.newaxis],
            3,
            axis=2
        )

        # -----------------------------
        # Adaptive contrast
        # -----------------------------
        contrast_img = (
            local_mean +
            factor * (img - local_mean)
        )

        img = (
            img * (1 - mask[:, :, None]) +
            contrast_img * mask[:, :, None]
        )

    return np.clip(
        img,
        0,
        255
    ).astype(np.uint8)

def add_fog(
    image,
    intensity=0.3,
    min_patches=4,
    max_patches=10,
):
    """
    Spatially Variant Fog Transformation.

    Controlled parameter
    --------------------
    intensity : fog density

    Randomized
    ----------
    • number of fog regions
    • locations
    • sizes
    • orientations
    • edge softness
    • overlap
    """

    img = image.astype(np.float32).copy()

    h, w = img.shape[:2]

    intensity = np.clip(float(intensity), 0.0, 1.0)

    fog_layer = np.full_like(img, 255)

    num_patches = np.random.randint(
        min_patches,
        max_patches + 1
    )

    mask_total = np.zeros((h, w), dtype=np.float32)

    for _ in range(num_patches):

        axis_x = np.random.randint(
            max(40, w // 10),
            max(100, w // 3)
        )

        axis_y = np.random.randint(
            max(40, h // 10),
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
            [51, 71, 91, 111]
        )

        mask = cv2.GaussianBlur(
            mask,
            (blur_size, blur_size),
            0
        )

        mask_total += mask.astype(np.float32) / 255.0

    mask_total = np.clip(mask_total, 0, 1)

    mask_total *= intensity

    result = (
        img * (1 - mask_total[:, :, None])
        + fog_layer * mask_total[:, :, None]
    )

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)