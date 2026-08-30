from .backend import USE_CUDA

if USE_CUDA:
    from .gpu.image_transforms import *
else:
    from .cpu.image_transforms import *