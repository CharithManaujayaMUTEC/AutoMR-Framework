from .backend import USE_CUDA

if USE_CUDA:
    from .gpu.temporal_transforms import *
else:
    from .cpu.temporal_transforms import *