from .backend import USE_CUDA

if USE_CUDA:
    from .gpu.translation import *
else:
    from .cpu.translation import *