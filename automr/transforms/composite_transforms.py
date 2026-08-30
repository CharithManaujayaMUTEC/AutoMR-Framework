from .backend import USE_CUDA

if USE_CUDA:
    from .gpu.composite_transforms import *
else:
    from .cpu.composite_transforms import *