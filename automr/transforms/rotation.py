import cv2

def rotate_small(image, angle=5):
    h, w = image.shape[:2]
    center = (w//2, h//2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))
