
import numpy as np

def add_noise(image, level=10):
    noise = np.random.randint(-level, level, image.shape)
    return np.clip(image + noise, 0, 255).astype(image.dtype)
