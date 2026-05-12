import numpy as np
import cv2

# 🌧️ Rain (STRONG + realistic + failure-inducing)
def add_rain(image, intensity=0.5):
    img = image.copy()
    h, w = img.shape[:2]

    rain = np.zeros_like(img)

    #  much denser rain
    num_drops = int(h * w * intensity * 0.01)

    for _ in range(num_drops):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)

        length = np.random.randint(10, 25)
        angle = np.random.randint(-3, 3)

        cv2.line(
            rain,
            (x, y),
            (x + angle, y + length),
            (220, 220, 220),
            1
        )

    #  stronger blending (this forces failures)
    return cv2.addWeighted(img, 0.6, rain, 0.7 * intensity, 0)


# ❄️ Snow (structured + bright particles)
def add_snow(image, intensity=0.5):
    img = image.copy().astype(np.float32)
    h, w = img.shape[:2]

    snow = img.copy()

    #  dense flakes
    num_flakes = int(h * w * intensity * 0.01)

    for _ in range(num_flakes):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)

        cv2.circle(snow, (x, y), radius=1, color=(255, 255, 255), thickness=-1)

    #  brightness washout
    snow = cv2.addWeighted(img, 0.7, snow, 0.6 * intensity, 0)

    return np.clip(snow, 0, 255).astype(np.uint8)


# 🌫️ Fog (VERY strong + destructive)
def add_fog(image, intensity=0.5):
    img = image.astype(np.float32)

    fog = np.full_like(img, 255)

    #  nonlinear amplification
    alpha = min(1.5, intensity * 1.8)

    foggy = cv2.addWeighted(img, 1 - alpha, fog, alpha, 0)

    #  simulate depth loss
    foggy = cv2.GaussianBlur(foggy, (15, 15), 0)

    return np.clip(foggy, 0, 255).astype(np.uint8)