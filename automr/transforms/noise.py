
import numpy as np

def add_noise(image, level=10):

    level = int(level)

    # 🔥 fix: avoid invalid range
    if level <= 0:
        return image.copy()

    noise = np.random.randint(-level, level + 1, image.shape)

    return np.clip(image + noise, 0, 255).astype(image.dtype)
