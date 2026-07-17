"""
Default relation registration.

This module registers the built-in metamorphic relations provided
by AutoMR into a relation registry instance.
"""

from automr.relations.image_relations import *
from automr.relations.weather_relations import *
from automr.relations.behavioral_relations import *
from automr.relations.temporal_relations import *


def register_default_relations(registry, epsilon):
    """
    Register all built-in metamorphic relations.

    Parameters
    ----------
    registry : RelationRegistry
        Registry used to store relation instances.
    epsilon : float
        Tolerance value passed to supported relations.
    """

    # Image relations.
    registry.register(
        "brightness",
        BrightnessRelation(epsilon)
    )

    registry.register(
        "rotation",
        RotationRelation(epsilon)
    )

    registry.register(
        "translation",
        TranslationRelation(epsilon)
    )

    registry.register(
        "noise",
        NoiseRelation(epsilon)
    )

    registry.register(
        "blur",
        BlurRelation(epsilon)
    )

    registry.register(
        "contrast",
        ContrastRelation(epsilon)
    )

    registry.register(
        "composite",
        CompositeRelation(epsilon)
    )

    registry.register(
        "global_brightness",
        GlobalBrightnessRelation(epsilon)
    )

    registry.register(
        "global_contrast",
        GlobalContrastRelation(epsilon)
    )

    registry.register(
        "global_blur",
        GlobalBlurRelation(epsilon)
    )

    registry.register(
        "global_noise",
        GlobalNoiseRelation(epsilon)
    )

    registry.register(
        "global_rotation",
        GlobalRotationRelation(epsilon)
    )

    registry.register(
        "global_translation",
        GlobalTranslationRelation(epsilon)
    )

    # Weather relations.
    registry.register(
        "rain",
        RainRelation(epsilon)
    )

    registry.register(
        "snow",
        SnowRelation(epsilon)
    )

    registry.register(
        "fog",
        FogRelation(epsilon)
    )

    registry.register(
        "sandstorm",
        SandstormRelation(epsilon)
    )

    registry.register(
        "dust",
        DustRelation(epsilon)
    )

    registry.register(
        "smoke",
        SmokeRelation(epsilon)
    )

    registry.register(
        "haze",
        HazeRelation(epsilon)
    )

    registry.register(
        "visibility",
        DarkVisibilityRelation(epsilon)
    )

    registry.register(
        "darkness",
        DarkVisibilityRelation(epsilon)
    )

    # Temporal relations.
    registry.register(
        "temporal",
        TemporalSmoothnessRelation(epsilon)
    )