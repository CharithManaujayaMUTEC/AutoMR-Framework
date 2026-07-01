from automr.transforms.image_transforms import *
from automr.transforms.weather_transforms import *
from automr.transforms.behavioral_transforms import *
from automr.transforms.temporal_transforms import *


def register_default_transforms(registry):

    registry.register("brightness", increase_brightness)
    registry.register("rotation", rotate_small)
    registry.register("translation", shift_right)
    registry.register("noise", add_noise)
    registry.register("blur", blur)
    registry.register("contrast", adjust_contrast)
    registry.register("composite", composite_transform)

    registry.register("rain", add_rain)
    registry.register("snow", add_snow)
    registry.register("fog", add_fog)
    registry.register("sandstorm", add_sandstorm)
    registry.register("dust", add_dust)
    registry.register("smoke", add_smoke)
    registry.register("haze", add_haze)

    registry.register("visibility", reduce_visibility)
    registry.register("darkness", darken)

    registry.register("temporal", next_frame_pair)