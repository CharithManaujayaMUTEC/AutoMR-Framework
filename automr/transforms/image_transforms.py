import numpy as np
import cv2

def increase_brightness(image, factor=1.0):
    img = image.astype(np.float32) * factor
    return np.clip(img, 0, 255).astype(np.uint8)

def rotate_small(image, angle=5):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def shift_right(image, pixels=5):
    h, w = image.shape[:2]
    M = np.float32([[1, 0, pixels], [0, 1, 0]])
    return cv2.warpAffine(image, M, (w, h))

def add_noise(image, level=10):
    level = int(level)

    if level <= 0:
        return image.copy()

    noise = np.random.randint(-level, level + 1, image.shape)
    return np.clip(image + noise, 0, 255).astype(image.dtype)

# 🔥 RENAMED
def mirror_image(image, _=None):
    return cv2.flip(image, 1)

def blur(image, k=5):
    k = int(k)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(image, (k, k), 0)

def adjust_contrast(image, factor=1.0):
    img = image.astype(np.float32) * factor
    return np.clip(img, 0, 255).astype(np.uint8)

def add_fog(image, intensity=0.3):
    fog = np.full_like(image, 255)
    return cv2.addWeighted(image, 1 - intensity, fog, intensity, 0)