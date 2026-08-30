from .backend import USE_CUDA

if USE_CUDA:
    from .gpu.behavioral_transforms import *
else:
    from .cpu.behavioral_transforms import *