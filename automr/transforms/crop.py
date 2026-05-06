
import cv2

def crop_top(image, fraction=0.2):
    h, w = image.shape[:2]
    cropped = image[int(h*fraction):, :]
    return cv2.resize(cropped, (w, h))
