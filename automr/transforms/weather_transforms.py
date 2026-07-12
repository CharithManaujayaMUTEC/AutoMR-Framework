from .backend import USE_CUDA

if USE_CUDA:
    from .gpu.weather_transforms import *
else:
    from .cpu.weather_transforms import *