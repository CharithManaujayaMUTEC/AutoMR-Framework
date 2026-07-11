"""
AutoMR Weather & Atmospheric Effects Package

This package provides reusable generators for physically-inspired
weather and visibility transformations.

Modules
-------
depth.py
    Approximate monocular depth estimation and atmospheric blending.

weather_particles.py
    Particle generators (rain, snow, dust).

atmospheric.py
    Atmospheric scattering effects (fog, haze, smoke, dust, sandstorm).

weather_renderer.py
    High-level rendering functions used by the transformations.
"""

from .depth import (
    estimate_depth,
    transmission,
    blend_with_airlight,
)

from .weather_particles import (
    generate_rain_layer,
    generate_snow_layer,
    generate_dust_layer,
)

from .atmospheric import (
    apply_fog,
    apply_haze,
    apply_smoke,
    apply_dust,
    apply_sandstorm,
)

from .weather_renderer import (
    render_rain,
    render_snow,
    render_dust,
    render_sandstorm,
    render_fog,
    render_haze,
    render_smoke,
)

__all__ = [
    # Depth
    "estimate_depth",
    "transmission",
    "blend_with_airlight",

    # Particle generators
    "generate_rain_layer",
    "generate_snow_layer",
    "generate_dust_layer",

    # Atmospheric effects
    "apply_fog",
    "apply_haze",
    "apply_smoke",
    "apply_dust",
    "apply_sandstorm",

    # High-level renderers
    "render_rain",
    "render_snow",
    "render_dust",
    "render_sandstorm",
    "render_fog",
    "render_haze",
    "render_smoke",
]