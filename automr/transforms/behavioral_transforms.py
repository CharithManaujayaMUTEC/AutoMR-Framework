import cv2
import numpy as np

def reduce_visibility(image, factor=0.5):
    image = np.asarray(image, dtype=np.float32)
    fog = np.full(image.shape, 255, dtype=np.float32)

    img = cv2.addWeighted(image, 1 - factor, fog, factor, 0)

    return np.clip(img, 0, 255).astype(np.uint8)

def darken(image, factor=0.5):
    img = image.astype(np.float32) * factor
    return np.clip(img, 0, 255).astype(np.uint8)