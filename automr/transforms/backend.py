import torch

USE_GPU = torch.cuda.is_available()


def gpu_available():
    return USE_GPU