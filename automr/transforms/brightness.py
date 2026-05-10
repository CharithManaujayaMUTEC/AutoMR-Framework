
import numpy as np

def increase_brightness(image, factor=1.0):
    img = image.astype(np.float32) * factor
    return np.clip(img, 0, 255).astype(np.uint8)
