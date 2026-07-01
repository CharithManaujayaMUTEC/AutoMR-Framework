from automr.relations.image_relations import *
from automr.relations.weather_relations import *
from automr.relations.behavioral_relations import *
from automr.relations.temporal_relations import *


def register_default_relations(registry, epsilon):

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

    registry.register(
        "temporal",
        TemporalSmoothnessRelation(epsilon)
    )