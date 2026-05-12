import numpy as np
import cv2

# 🌧️ Rain
def add_rain(image, intensity=0.5):
    img = image.copy()

    h, w = img.shape[:2]
    rain = np.zeros_like(img)

    num_drops = int(h * w * intensity * 0.002)

    for _ in range(num_drops):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        length = np.random.randint(5, 15)

        cv2.line(rain, (x, y), (x, y + length), (200, 200, 200), 1)

    return cv2.addWeighted(img, 0.8, rain, 0.2, 0)


# ❄️ Snow
def add_snow(image, intensity=0.5):
    img = image.astype(np.float32)

    noise = np.random.randn(*img.shape) * 255 * intensity
    snow = img + noise

    return np.clip(snow, 0, 255).astype(np.uint8)


# 🌫️ Fog (better version than before)
def add_fog(image, intensity=0.3):
    fog = np.full_like(image, 255)
    return cv2.addWeighted(image, 1 - intensity, fog, intensity, 0)