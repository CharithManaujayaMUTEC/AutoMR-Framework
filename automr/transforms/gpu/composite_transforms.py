import torch

# Image transforms
from automr.transforms.image_transforms import (
    increase_brightness,
    adjust_contrast,
    blur,
    add_noise,
    rotate_small,
    shift_right,
)

# Weather transforms
from automr.transforms.weather_transforms import (
    add_rain,
    add_snow,
    add_fog,
    add_dust,
    add_haze,
    add_smoke,
    add_sandstorm,
)

# Behavioral transforms
from automr.transforms.behavioral_transforms import (
    darken,
    reduce_visibility,
)

from automr.transforms.utils import create_rng
from automr.transforms.backend import DEVICE


def composite_transform(
    image,
    factor=0.5,
    seed=None,
):
    """
    GPU-compatible composite transform.
    """

    rng = create_rng(seed)

    factor = float(factor)

    # Move image to GPU if needed
    if not isinstance(image, torch.Tensor):
        img = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .float()
            .to(DEVICE)
        )
    else:
        img = image.to(DEVICE)

    available = [

        lambda x: increase_brightness(
            x,
            factor=max(1.0, 1.0 + factor),
            seed=rng.integers(1_000_000),
        ),

        lambda x: adjust_contrast(
            x,
            factor=max(1.0, 1.0 + factor),
            seed=rng.integers(1_000_000),
        ),

        lambda x: blur(
            x,
            k=max(3, int(3 + factor * 8)),
            seed=rng.integers(1_000_000),
        ),

        lambda x: add_noise(
            x,
            level=max(2, int(5 + factor * 25)),
            seed=rng.integers(1_000_000),
        ),

        lambda x: rotate_small(
            x,
            angle=max(1, factor * 15),
            seed=rng.integers(1_000_000),
        ),

        lambda x: shift_right(
            x,
            pixels=max(2, int(5 + factor * 20)),
            seed=rng.integers(1_000_000),
        ),

        lambda x: darken(
            x,
            factor=max(0.15, 1.0 - factor),
            seed=rng.integers(1_000_000),
        ),

        lambda x: reduce_visibility(
            x,
            factor=min(1.0, factor),
            seed=rng.integers(1_000_000),
        ),

        lambda x: add_rain(
            x,
            intensity=min(1.0, factor),
            seed=rng.integers(1_000_000),
        ),

        lambda x: add_snow(
            x,
            intensity=min(1.0, factor),
            seed=rng.integers(1_000_000),
        ),

        lambda x: add_fog(
            x,
            intensity=min(1.0, factor),
            seed=rng.integers(1_000_000),
        ),

        lambda x: add_dust(
            x,
            intensity=min(1.0, factor),
            seed=rng.integers(1_000_000),
        ),

        lambda x: add_haze(
            x,
            intensity=min(1.0, factor),
            seed=rng.integers(1_000_000),
        ),

        lambda x: add_smoke(
            x,
            intensity=min(1.0, factor),
            seed=rng.integers(1_000_000),
        ),

        lambda x: add_sandstorm(
            x,
            intensity=min(1.0, factor),
            seed=rng.integers(1_000_000),
        ),
    ]

    # Random number of transforms
    num_transforms = int(
        rng.integers(2, 6)
    )

    # Random subset
    chosen = list(
        rng.choice(
            available,
            size=num_transforms,
            replace=False,
        )
    )

    # Random order
    rng.shuffle(chosen)

    # Sequential GPU execution
    for transform in chosen:
        img = transform(img)

    return img