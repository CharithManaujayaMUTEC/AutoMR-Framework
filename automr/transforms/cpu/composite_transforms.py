import random

# Image transforms
from automr.transforms.cpu.image_transforms import (
    increase_brightness,
    adjust_contrast,
    blur,
    add_noise,
    rotate_small,
    shift_right,
)

# Weather transforms
from automr.transforms.cpu.weather_transforms import (
    add_rain,
    add_snow,
    add_fog,
    add_dust,
    add_haze,
    add_smoke,
    add_sandstorm,
)

# Behavioral transforms
from automr.transforms.cpu.behavioral_transforms import (
    darken,
    reduce_visibility,
)

from automr.transforms.utils import create_rng


def composite_transform(
    image,
    factor=0.5,
    seed=None,
):
    """
    Random Composite Transformation.

    Controlled parameter
    --------------------
    factor : transformation severity

    Randomized
    ----------
    • Number of transformations
    • Selected transformations
    • Order of execution

    Reproducible when a seed is supplied.
    """

    rng = create_rng(seed)

    img = image.copy()

    factor = float(factor)

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

    # Random number of transformations
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

    # Apply sequentially
    for transform in chosen:
        img = transform(img)

    return img