import random


def composite_transform(image, factor=0.5):
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

    Notes
    -----
    Every selected transformation receives the
    SAME factor supplied by AutoMR.
    """

    img = image.copy()

    available = [

        lambda x: increase_brightness(
            x,
            factor=max(1.0, factor)
        ),

        lambda x: adjust_contrast(
            x,
            factor=max(1.0, factor)
        ),

        lambda x: blur(
            x,
            k=max(3, int(factor))
        ),

        lambda x: add_noise(
            x,
            level=max(1, int(factor))
        ),

        lambda x: rotate_small(
            x,
            angle=factor
        ),

        lambda x: shift_right(
            x,
            pixels=max(1, int(factor))
        ),

        lambda x: darken(
            x,
            factor=max(0.05, factor)
        ),

        lambda x: reduce_visibility(
            x,
            factor=min(1.0, factor)
        ),

        lambda x: add_rain(
            x,
            intensity=min(1.0, factor)
        ),

        lambda x: add_snow(
            x,
            intensity=min(1.0, factor)
        ),

        lambda x: add_fog(
            x,
            intensity=min(1.0, factor)
        ),

        lambda x: add_dust(
            x,
            intensity=min(1.0, factor)
        ),

        lambda x: add_haze(
            x,
            intensity=min(1.0, factor)
        ),

        lambda x: add_smoke(
            x,
            intensity=min(1.0, factor)
        ),

        lambda x: add_sandstorm(
            x,
            intensity=min(1.0, factor)
        ),
    ]

    # Random number of transformations
    num_transforms = random.randint(2, 5)

    # Random subset
    chosen = random.sample(
        available,
        num_transforms
    )

    # Random order
    random.shuffle(chosen)

    # Apply sequentially
    for transform in chosen:
        img = transform(img)

    return img