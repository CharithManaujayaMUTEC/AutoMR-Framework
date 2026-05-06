
import numpy as np

def increase_brightness(image, delta=30):
    return np.clip(image + delta, 0, 255).astype(image.dtype)
