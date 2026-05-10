
import cv2
import numpy as np

def shift_right(image, pixels=5):
    h, w = image.shape[:2]
    M = np.float32([[1, 0, pixels], [0, 1, 0]])
    return cv2.warpAffine(image, M, (w, h))
