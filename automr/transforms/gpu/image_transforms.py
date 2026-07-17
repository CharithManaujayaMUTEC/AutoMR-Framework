import torch
import torch.nn.functional as F
import kornia
import kornia.geometry.transform as KGT
import kornia.filters as KF
import numpy as np


# ----------------------------------------------------------
# Device
# ----------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------

def _to_tensor(image):

    if isinstance(image, torch.Tensor):

        if image.ndim == 4:
            return image.float().to(DEVICE)

        if image.ndim == 3:

            # already CHW
            if image.shape[0] in (1, 3):
                return image.unsqueeze(0).float().to(DEVICE)

            # HWC
            return image.permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

    x = torch.from_numpy(np.asarray(image))

    if x.ndim == 3:
        x = x.permute(2, 0, 1).unsqueeze(0)

    return x.float().to(DEVICE)


def _to_numpy(t):

    if t.ndim == 4:
        t = t.squeeze(0)

    if t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0)

    return (
        t.clamp(0,255)
         .byte()
         .cpu()
         .numpy()
    )


def _blend(original, transformed, mask):
    return original * (1.0 - mask) + transformed * mask


def _random_mask(h, w):
    mask = torch.zeros((1, 1, h, w), device=DEVICE)

    # Handle tiny images
    if h < 4 or w < 4:
        return mask

    for _ in range(np.random.randint(3, 8)):

        min_ph = max(1, int(h * 0.08))
        max_ph = max(min_ph + 1, int(h * 0.30))

        min_pw = max(1, int(w * 0.08))
        max_pw = max(min_pw + 1, int(w * 0.30))

        ph = np.random.randint(min_ph, max_ph)
        pw = np.random.randint(min_pw, max_pw)

        # Never allow the patch to exceed the image
        ph = min(ph, h)
        pw = min(pw, w)

        max_y = max(1, h - ph + 1)
        max_x = max(1, w - pw + 1)

        y = np.random.randint(0, max_y)
        x = np.random.randint(0, max_x)

        mask[:, :, y:y + ph, x:x + pw] = 1.0

    mask = KF.gaussian_blur2d(
        mask,
        (31, 31),
        (10.0, 10.0),
    )

    return mask.clamp(0, 1)

# ==========================================================
# Brightness
# ==========================================================

def increase_brightness(
    image,
    factor=1.2,
    seed=None,
):

    img = _to_tensor(image)

    _, _, h, w = img.shape

    mask = _random_mask(h, w)

    bright = img * float(factor)

    out = _blend(
        img,
        bright,
        mask,
    )

    return _to_numpy(out)


# ==========================================================
# Contrast
# ==========================================================

def adjust_contrast(
    image,
    factor=1.2,
    seed=None,
):

    img = _to_tensor(image)

    _, _, h, w = img.shape

    gray = kornia.color.rgb_to_grayscale(
        img / 255.0
    )

    mean = KF.gaussian_blur2d(
        gray,
        (31, 31),
        (10.0, 10.0),
    )

    mean = mean.repeat(1, 3, 1, 1) * 255.0

    contrast = mean + float(factor) * (img - mean)

    mask = _random_mask(h, w)

    out = _blend(
        img,
        contrast,
        mask,
    )

    return _to_numpy(out)


# ==========================================================
# Blur
# ==========================================================

def blur(
    image,
    k=11,
    seed=None,
):

    img = _to_tensor(image)

    _, _, h, w = img.shape

    k = int(k)

    if k < 3:
        k = 3

    if k % 2 == 0:
        k += 1

    blurred = KF.gaussian_blur2d(
        img,
        (k, k),
        (2.0, 2.0),
    )

    mask = _random_mask(h, w)

    out = _blend(
        img,
        blurred,
        mask,
    )

    return _to_numpy(out)

# ==========================================================
# Noise
# ==========================================================

def add_noise(
    image,
    level=15,
    seed=None,
):
    """
    GPU Gaussian noise.
    """

    img = _to_tensor(image)

    _, _, h, w = img.shape

    sigma = float(level)

    noise = torch.randn_like(img) * sigma

    noisy = img + noise

    mask = _random_mask(h, w)

    out = _blend(
        img,
        noisy,
        mask,
    )

    return _to_numpy(out)


# ==========================================================
# Local Rotation
# ==========================================================

def rotate_small(
    image,
    angle=5,
    seed=None,
):
    """
    GPU rotation.
    """

    img = _to_tensor(image)

    theta = torch.tensor(
        [float(angle)],
        device=DEVICE,
    )

    rotated = KGT.rotate(
        img,
        theta,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    )

    _, _, h, w = img.shape

    mask = _random_mask(h, w)

    out = _blend(
        img,
        rotated,
        mask,
    )

    return _to_numpy(out)


# ==========================================================
# Local Translation
# ==========================================================

def shift_right(
    image,
    pixels=10,
    seed=None,
):
    """
    GPU translation.
    """

    img = _to_tensor(image)

    tx = float(np.random.randint(-pixels, pixels + 1))
    ty = float(np.random.randint(-pixels, pixels + 1))

    transform = torch.tensor(
        [[[1.0, 0.0, tx],
          [0.0, 1.0, ty]]],
        device=DEVICE,
    )

    translated = KGT.warp_affine(
        img,
        transform,
        dsize=(img.shape[2], img.shape[3]),
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    )

    _, _, h, w = img.shape

    mask = _random_mask(h, w)

    out = _blend(
        img,
        translated,
        mask,
    )

    return _to_numpy(out)


# ==========================================================
# Mirror
# ==========================================================

def mirror_image(
    image,
    *_,
):
    """
    GPU horizontal flip.
    """

    img = _to_tensor(image)

    flipped = torch.flip(
        img,
        dims=[3],
    )

    return _to_numpy(flipped)

# ==========================================================
# Batch Versions
# ==========================================================

def increase_brightness_batch(
    images,
    factor=1.2,
):
    """
    Batch GPU brightness.
    images : (N,C,H,W)
    """

    images = images.float().to(DEVICE)

    bright = images * float(factor)

    return bright.clamp(0, 255)


def adjust_contrast_batch(
    images,
    factor=1.2,
):
    """
    Batch GPU contrast.
    """

    images = images.float().to(DEVICE)

    gray = kornia.color.rgb_to_grayscale(
        images / 255.0
    )

    mean = KF.gaussian_blur2d(
        gray,
        (31, 31),
        (10.0, 10.0),
    )

    mean = mean.repeat(1, 3, 1, 1) * 255.0

    contrast = mean + float(factor) * (images - mean)

    return contrast.clamp(0, 255)


def blur_batch(
    images,
    k=11,
):
    """
    Batch GPU blur.
    """

    images = images.float().to(DEVICE)

    k = int(k)

    if k < 3:
        k = 3

    if k % 2 == 0:
        k += 1

    return KF.gaussian_blur2d(
        images,
        (k, k),
        (2.0, 2.0),
    ).clamp(0, 255)


def add_noise_batch(
    images,
    level=15,
):
    """
    Batch GPU noise.
    """

    images = images.float().to(DEVICE)

    noise = torch.randn_like(images) * float(level)

    return (images + noise).clamp(0, 255)


def rotate_batch(
    images,
    angle=5,
):
    """
    Batch GPU rotation.
    """

    images = images.float().to(DEVICE)

    angles = torch.full(
        (images.shape[0],),
        float(angle),
        device=DEVICE,
    )

    return KGT.rotate(
        images,
        angles,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    ).clamp(0, 255)


def translate_batch(
    images,
    pixels=10,
):
    """
    Batch GPU translation.
    """

    images = images.float().to(DEVICE)

    n = images.shape[0]

    transforms = torch.zeros(
        (n, 2, 3),
        device=DEVICE,
    )

    transforms[:, 0, 0] = 1
    transforms[:, 1, 1] = 1

    transforms[:, 0, 2] = float(pixels)
    transforms[:, 1, 2] = 0.0

    return KGT.warp_affine(
        images,
        transforms,
        dsize=(images.shape[2], images.shape[3]),
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    ).clamp(0, 255)


def mirror_batch(
    images,
):
    """
    Batch GPU mirror.
    """

    return torch.flip(
        images.to(DEVICE),
        dims=[3],
    )

# ==========================================================
# Batch Utilities
# ==========================================================

def numpy_to_batch(images):
    """
    List[np.ndarray] -> Torch Batch
    """

    tensors = []

    for img in images:
        x = torch.from_numpy(img)
        x = x.permute(2, 0, 1)
        tensors.append(x)

    return torch.stack(
        tensors,
        dim=0,
    ).float().to(DEVICE)


def batch_to_numpy(batch):
    """
    Torch Batch -> List[np.ndarray]
    """

    batch = batch.clamp(
        0,
        255,
    ).byte().cpu()

    outputs = []

    for img in batch:

        outputs.append(
            img.permute(
                1,
                2,
                0,
            ).numpy()
        )

    return outputs


# ==========================================================
# Generic GPU Dispatcher
# ==========================================================

GPU_TRANSFORMS = {
    "brightness": increase_brightness_batch,
    "contrast": adjust_contrast_batch,
    "blur": blur_batch,
    "noise": add_noise_batch,
    "rotation": rotate_batch,
    "translation": translate_batch,
    "mirror": mirror_batch,
}


def apply_batch(
    name,
    images,
    parameter,
):
    """
    Apply one transformation to an entire batch.
    """

    batch = numpy_to_batch(images)

    fn = GPU_TRANSFORMS[name]

    result = fn(
        batch,
        parameter,
    )

    return batch_to_numpy(result)


# ==========================================================
# GPU Availability
# ==========================================================

def gpu_available():
    return torch.cuda.is_available()


def current_device():
    return DEVICE


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==========================================================
# Global Brightness
# ==========================================================

def global_brightness(
    image,
    factor=1.2,
    seed=None,
):
    """
    GPU global brightness.
    """

    img = _to_tensor(image)

    bright = img * float(factor)

    return _to_numpy(
        bright.clamp(0, 255)
    )


# ==========================================================
# Global Darkness
# ==========================================================

def global_darkness(
    image,
    factor=0.75,
    seed=None,
):
    """
    GPU global darkness.
    """

    img = _to_tensor(image)

    dark = img * float(factor)

    return _to_numpy(
        dark.clamp(0, 255)
    )


# ==========================================================
# Global Contrast
# ==========================================================

def global_contrast(
    image,
    factor=1.2,
    seed=None,
):
    """
    GPU global contrast.
    """

    img = _to_tensor(image)

    gray = kornia.color.rgb_to_grayscale(
        img / 255.0
    )

    mean = gray.mean(
        dim=(2, 3),
        keepdim=True,
    ) * 255.0

    mean = mean.repeat(
        1,
        3,
        1,
        1,
    )

    contrast = mean + float(factor) * (
        img - mean
    )

    return _to_numpy(
        contrast.clamp(0, 255)
    )


# ==========================================================
# Global Blur
# ==========================================================

def global_blur(
    image,
    k=11,
    seed=None,
):
    """
    GPU global blur.
    """

    img = _to_tensor(image)

    k = int(k)

    if k < 3:
        k = 3

    if k % 2 == 0:
        k += 1

    blurred = KF.gaussian_blur2d(
        img,
        (k, k),
        (2.0, 2.0),
    )

    return _to_numpy(
        blurred.clamp(0, 255)
    )


# ==========================================================
# Global Noise
# ==========================================================

def global_noise(
    image,
    level=15,
    seed=None,
):
    """
    GPU global Gaussian noise.
    """

    img = _to_tensor(image)

    noise = (
        torch.randn_like(img)
        * float(level)
    )

    noisy = img + noise

    return _to_numpy(
        noisy.clamp(0, 255)
    )


# ==========================================================
# Global Rotation
# ==========================================================

def global_rotation(
    image,
    angle=5,
    seed=None,
):
    """
    GPU global rotation.
    """

    img = _to_tensor(image)

    theta = torch.tensor(
        [float(angle)],
        device=DEVICE,
    )

    rotated = KGT.rotate(
        img,
        theta,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    )

    return _to_numpy(
        rotated.clamp(0, 255)
    )


# ==========================================================
# Global Translation
# ==========================================================

def global_translation(
    image,
    pixels=10,
    seed=None,
):
    """
    GPU global translation.
    """

    img = _to_tensor(image)

    transform = torch.tensor(
        [[[1.0, 0.0, float(pixels)],
          [0.0, 1.0, 0.0]]],
        device=DEVICE,
    )

    translated = KGT.warp_affine(
        img,
        transform,
        dsize=(
            img.shape[2],
            img.shape[3],
        ),
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    )

    return _to_numpy(
        translated.clamp(0, 255)
    )