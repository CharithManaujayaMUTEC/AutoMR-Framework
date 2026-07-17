import cv2
import numpy as np

from ..utils import (
    create_rng,
    create_random_patch,
    create_random_mask,
    blend_images,
    random_motion_kernel,
)


# ==========================================================
# Brightness
# ==========================================================

def increase_brightness(
    image,
    factor=1.2,
    seed=None,
):
    """
    Localized adaptive brightness.

    Deterministic:
        factor

    Random:
        • patch count
        • patch size
        • patch locations
        • edge softness
        • local intensity variation
    """

    rng = create_rng(seed)

    img = image.astype(np.float32)

    mask = create_random_mask(
        image.shape[:2],
        rng=rng,
        min_regions=3,
        max_regions=8,
        min_scale=0.08,
        max_scale=0.30,
        blur_choices=(31, 41, 51, 61),
    )

    local_factor = factor * rng.uniform(0.9, 1.15)

    bright = img * local_factor

    return blend_images(
        img,
        bright,
        mask,
    )


# ==========================================================
# Contrast
# ==========================================================

def adjust_contrast(
    image,
    factor=1.2,
    seed=None,
):
    """
    Localized adaptive contrast.

    Deterministic:
        factor

    Random:
        • regions
        • region geometry
        • mask softness
        • slight contrast variation
    """

    rng = create_rng(seed)

    img = image.astype(np.float32)

    mask = create_random_mask(
        image.shape[:2],
        rng=rng,
        min_regions=3,
        max_regions=8,
        min_scale=0.10,
        max_scale=0.30,
        blur_choices=(31, 41, 51),
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    ).astype(np.float32)

    mean = cv2.GaussianBlur(
        gray,
        (31, 31),
        0,
    )

    mean = np.repeat(
        mean[:, :, None],
        3,
        axis=2,
    )

    local_factor = factor * rng.uniform(0.9, 1.1)

    contrast = mean + local_factor * (img - mean)

    return blend_images(
        img,
        contrast,
        mask,
    )


# ==========================================================
# Blur
# ==========================================================

def blur(
    image,
    k=11,
    seed=None,
):
    """
    Localized blur.

    Deterministic:
        kernel size

    Random:
        • Gaussian or motion blur
        • motion direction
        • affected regions
    """

    rng = create_rng(seed)

    img = image.astype(np.float32)

    k = int(round(float(k)))

    if k < 3:
        k = 3

    if k % 2 == 0:
        k += 1

    mask = create_random_mask(
        image.shape[:2],
        rng=rng,
        min_regions=3,
        max_regions=8,
        min_scale=0.10,
        max_scale=0.30,
        blur_choices=(31, 41, 51),
    )

    if rng.random() < 0.5:

        blurred = cv2.GaussianBlur(
            image,
            (k, k),
            0,
        ).astype(np.float32)

    else:

        kernel = random_motion_kernel(
            ksize=k,
            rng=rng,
        )

        blurred = cv2.filter2D(
            image,
            -1,
            kernel,
        ).astype(np.float32)

    return blend_images(
        img,
        blurred,
        mask,
    )

# ==========================================================
# Noise
# ==========================================================

def add_noise(
    image,
    level=15,
    seed=None,
):
    """
    Localized stochastic sensor noise.

    Deterministic
    -------------
    level

    Random
    ------
    • Gaussian / Speckle / Salt & Pepper
    • Region geometry
    • Noise variance
    • Region placement
    """

    rng = create_rng(seed)

    img = image.astype(np.float32)

    mask = create_random_mask(
        image.shape[:2],
        rng=rng,
        min_regions=4,
        max_regions=10,
        min_scale=0.06,
        max_scale=0.25,
        blur_choices=(31, 41, 51, 61),
    )

    mode = rng.choice([
        "gaussian",
        "speckle",
        "saltpepper",
    ])

    if mode == "gaussian":

        sigma = level * rng.uniform(0.8, 1.4)

        noisy = img + rng.normal(
            0,
            sigma,
            img.shape,
        )

    elif mode == "speckle":

        sigma = (level / 255.0) * rng.uniform(
            0.5,
            1.5,
        )

        noisy = img + img * rng.normal(
            0,
            sigma,
            img.shape,
        )

    else:

        noisy = img.copy()

        amount = (level / 255.0) * rng.uniform(
            0.5,
            1.2,
        )

        salt = rng.random(image.shape[:2]) < amount / 2
        pepper = rng.random(image.shape[:2]) < amount / 2

        noisy[salt] = 255
        noisy[pepper] = 0

    return blend_images(
        img,
        noisy,
        mask,
    )


# ==========================================================
# Local Rotation
# ==========================================================

def rotate_small(
    image,
    angle=5,
    seed=None,
):
    """
    Localized rotation with optional scaling.
    """

    rng = create_rng(seed)

    img = image.astype(np.float32)

    h, w = image.shape[:2]

    patch = create_random_patch(
        (h, w),
        rng,
        min_scale=0.18,
        max_scale=0.40,
    )

    x, y, pw, ph = patch

    roi = img[
        y:y + ph,
        x:x + pw,
    ].copy()

    theta = angle * rng.uniform(
        -1.0,
        1.0,
    )

    scale = rng.uniform(
        0.97,
        1.03,
    )

    angle = float(angle)

    theta = float(angle) * rng.uniform(-1.0, 1.0)
    scale = float(rng.uniform(0.97, 1.03))

    M = cv2.getRotationMatrix2D(
        (pw / 2.0, ph / 2.0),
        theta,
        scale,
    )

    rotated = cv2.warpAffine(
        roi,
        M,
        (pw, ph),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )

    mask = create_random_mask(
        (ph, pw),
        rng=rng,
        min_regions=1,
        max_regions=2,
        min_scale=0.7,
        max_scale=1.0,
        blur_choices=(31, 41),
    )

    img[
        y:y + ph,
        x:x + pw,
    ] = blend_images(
        roi,
        rotated,
        mask,
    )

    return np.clip(
        img,
        0,
        255,
    ).astype(np.uint8)


# ==========================================================
# Local Translation
# ==========================================================

def shift_right(
    image,
    pixels=10,
    seed=None,
):
    """
    Localized translation.
    """

    rng = create_rng(seed)

    img = image.astype(np.float32)

    h, w = image.shape[:2]

    x, y, pw, ph = create_random_patch(
        (h, w),
        rng,
        min_scale=0.18,
        max_scale=0.40,
    )

    roi = img[
        y:y + ph,
        x:x + pw,
    ].copy()

    dx = rng.integers(
        -pixels,
        pixels + 1,
    )

    dy = rng.integers(
        -pixels,
        pixels + 1,
    )

    M = np.float32([
        [1, 0, dx],
        [0, 1, dy],
    ])

    translated = cv2.warpAffine(
        roi,
        M,
        (pw, ph),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )

    mask = create_random_mask(
        (ph, pw),
        rng=rng,
        min_regions=1,
        max_regions=2,
        min_scale=0.7,
        max_scale=1.0,
        blur_choices=(31, 41),
    )

    img[
        y:y + ph,
        x:x + pw,
    ] = blend_images(
        roi,
        translated,
        mask,
    )

    return np.clip(
        img,
        0,
        255,
    ).astype(np.uint8)


# ==========================================================
# Mirror
# ==========================================================

def mirror_image(
    image,
    *_,
):
    """
    Horizontal mirror.
    """

    return cv2.flip(
        image,
        1,
    )

# ==========================================================
# Global Brightness
# ==========================================================

def global_brightness(
    image,
    factor=1.2,
    seed=None,
):
    """
    Global adaptive brightness.

    Deterministic:
        factor

    Random:
        • slight intensity variation
    """

    rng = create_rng(seed)

    img = image.astype(np.float32)

    local_factor = factor * rng.uniform(
        0.95,
        1.05,
    )

    bright = img * local_factor

    return np.clip(
        bright,
        0,
        255,
    ).astype(np.uint8)

# ==========================================================
# Global Contrast
# ==========================================================

def global_contrast(
    image,
    factor=1.2,
    seed=None,
):
    """
    Global adaptive contrast.

    Deterministic:
        factor

    Random:
        • slight contrast variation
    """

    rng = create_rng(seed)

    img = image.astype(np.float32)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    ).astype(np.float32)

    mean = np.mean(gray)

    local_factor = factor * rng.uniform(
        0.95,
        1.05,
    )

    contrast = mean + local_factor * (
        img - mean
    )

    return np.clip(
        contrast,
        0,
        255,
    ).astype(np.uint8)


# ==========================================================
# Global Blur
# ==========================================================

def global_blur(
    image,
    k=11,
    seed=None,
):
    """
    Global blur.

    Deterministic:
        kernel size

    Random:
        • Gaussian or motion blur
        • motion direction
    """

    rng = create_rng(seed)

    k = int(round(float(k)))

    if k < 3:
        k = 3

    if k % 2 == 0:
        k += 1

    if rng.random() < 0.5:

        blurred = cv2.GaussianBlur(
            image,
            (k, k),
            0,
        )

    else:

        kernel = random_motion_kernel(
            ksize=k,
            rng=rng,
        )

        blurred = cv2.filter2D(
            image,
            -1,
            kernel,
        )

    return np.clip(
        blurred,
        0,
        255,
    ).astype(np.uint8)

# ==========================================================
# Global Noise
# ==========================================================

def global_noise(
    image,
    level=15,
    seed=None,
):
    """
    Global stochastic sensor noise.

    Deterministic
    -------------
    level

    Random
    ------
    • Gaussian / Speckle / Salt & Pepper
    • Noise variance
    """

    rng = create_rng(seed)

    img = image.astype(np.float32)

    mode = rng.choice([
        "gaussian",
        "speckle",
        "saltpepper",
    ])

    if mode == "gaussian":

        sigma = level * rng.uniform(
            0.8,
            1.4,
        )

        noisy = img + rng.normal(
            0,
            sigma,
            img.shape,
        )

    elif mode == "speckle":

        sigma = (level / 255.0) * rng.uniform(
            0.5,
            1.5,
        )

        noisy = img + img * rng.normal(
            0,
            sigma,
            img.shape,
        )

    else:

        noisy = img.copy()

        amount = (level / 255.0) * rng.uniform(
            0.5,
            1.2,
        )

        salt = rng.random(
            image.shape[:2],
        ) < amount / 2

        pepper = rng.random(
            image.shape[:2],
        ) < amount / 2

        noisy[salt] = 255
        noisy[pepper] = 0

    return np.clip(
        noisy,
        0,
        255,
    ).astype(np.uint8)


# ==========================================================
# Global Rotation
# ==========================================================

def global_rotation(
    image,
    angle=5,
    seed=None,
):
    """
    Global image rotation.

    Deterministic:
        angle

    Random:
        • rotation direction
        • slight scaling
    """

    rng = create_rng(seed)

    h, w = image.shape[:2]

    theta = float(angle) * rng.uniform(
        -1.0,
        1.0,
    )

    scale = rng.uniform(
        0.97,
        1.03,
    )

    M = cv2.getRotationMatrix2D(
        (
            w / 2.0,
            h / 2.0,
        ),
        theta,
        scale,
    )

    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )

    return rotated.astype(np.uint8)


# ==========================================================
# Global Translation
# ==========================================================

def global_translation(
    image,
    pixels=10,
    seed=None,
):
    """
    Global image translation.

    Deterministic:
        pixels

    Random:
        • translation direction
    """

    rng = create_rng(seed)

    h, w = image.shape[:2]

    dx = rng.integers(
        -pixels,
        pixels + 1,
    )

    dy = rng.integers(
        -pixels,
        pixels + 1,
    )

    M = np.float32([
        [1, 0, dx],
        [0, 1, dy],
    ])

    translated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )

    return translated.astype(np.uint8)