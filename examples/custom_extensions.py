"""
AutoMR Custom Extension Example

Demonstrates how to add:

1. A custom transformation
2. A custom metamorphic relation
3. A complete custom MR

without modifying AutoMR's built-in transformations,
relations, or registries.
"""

import numpy as np

from automr import AutoMR


# ==========================================================
# Custom Transformation
# ==========================================================

def invert_colors(
    image,
    factor=1.0,
    seed=None,
):
    """
    Custom image inversion transformation.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.

    factor : float
        Transformation intensity.

        0.0 -> original image
        1.0 -> fully inverted image

    seed : optional
        Included for compatibility with AutoMR
        transformations.

    Returns
    -------
    numpy.ndarray
        Transformed image.
    """

    image = image.astype(np.float32)

    inverted = 255.0 - image

    transformed = (
        (1.0 - factor) * image
        + factor * inverted
    )

    return np.clip(
        transformed,
        0,
        255,
    ).astype(np.uint8)


# ==========================================================
# Custom Metamorphic Relation
# ==========================================================

class InvertColorRelation:
    """
    Custom metamorphic relation.

    The prediction is expected to remain reasonably stable
    under color inversion.
    """

    def __init__(
        self,
        epsilon=0.10,
    ):
        self.epsilon = epsilon

    def type(self):
        """
        Relation type.
        """

        return "equality"

    def expected(self):
        """
        Human-readable expected behavior.
        """

        return (
            "Prediction should remain approximately "
            "consistent under controlled color inversion."
        )

    def check(
        self,
        y1,
        y2,
    ):
        """
        Evaluate the metamorphic relation.

        PASS when the prediction difference is within
        the configured tolerance.
        """

        return abs(y1 - y2) <= self.epsilon


# ==========================================================
# Register Custom MR
# ==========================================================

def register_custom_extensions(
    automr,
):
    """
    Register the custom transformation and relation
    with an AutoMR instance.

    Existing AutoMR functionality is not modified.
    """

    automr.register_custom_mr(
        name="invert_colors",

        transform=invert_colors,

        relation=InvertColorRelation(
            epsilon=0.10,
        ),

        param_range={
            "start": 0.0,
            "end": 1.0,
            "samples": 5,
        },
    )

    return automr


# ==========================================================
# Example Usage
# ==========================================================

if __name__ == "__main__":

    print(
        "Custom AutoMR Extension Example"
    )

    print(
        "\nThis file demonstrates how to define "
        "and register a custom transformation and "
        "metamorphic relation."
    )

    print(
        "\nExample registration:"
    )

    print(
        """
automr.register_custom_mr(
    name="invert_colors",
    transform=invert_colors,
    relation=InvertColorRelation(
        epsilon=0.10
    ),
    param_range={
        "start": 0.0,
        "end": 1.0,
        "samples": 5,
    },
)
        """
    )