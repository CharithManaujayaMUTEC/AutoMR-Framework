import cv2
import numpy as np

def reduce_visibility(image, factor=0.5):
    fog = np.full_like(image, 255)
    return cv2.addWeighted(image, 1 - factor, fog, factor, 0)

def darken(image, factor=0.5):
    img = image.astype(np.float32) * factor
    return np.clip(img, 0, 255).astype(np.uint8)