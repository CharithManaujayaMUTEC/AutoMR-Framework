# webcam_automr_demo.py

import sys
import cv2
import numpy as np

# --------------------------------------------------
# USER PATHS
# --------------------------------------------------
sys.path.append("D:/FYP 78SEm/Datasets")
sys.path.append("D:/FYP 78SEm/Modals")

from load_model import get_model
from automr.transforms.image_transforms import increase_brightness

# --------------------------------------------------
# MODEL WRAPPER
# --------------------------------------------------
class RealModel:

    def __init__(self):
        self.model = get_model()

    def preprocess(self, img):
        img = cv2.resize(img, (200, 66))
        img = img.astype(np.float32) / 255.0
        return img

    def predict(self, x):

        if x is None:
            return 0.0

        x = self.preprocess(x)
        x = np.expand_dims(x, axis=0)

        pred = self.model.predict(
            x,
            verbose=0
        )

        return float(pred.flatten()[0])


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
print("Loading model...")
model = RealModel()
print("✅ Model loaded")

# --------------------------------------------------
# WEBCAM
# --------------------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("✅ Webcam started")
print("Press ESC to quit")

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
BRIGHTNESS_FACTOR = 1.5
THRESHOLD = 0.01

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    try:

        # ------------------------------------------
        # ORIGINAL PREDICTION
        # ------------------------------------------
        original_pred = model.predict(frame)

        # ------------------------------------------
        # APPLY MR
        # ------------------------------------------
        transformed = increase_brightness(
            frame.copy(),
            BRIGHTNESS_FACTOR
        )

        transformed_pred = model.predict(
            transformed
        )

        # ------------------------------------------
        # CHECK
        # ------------------------------------------
        diff = abs(
            transformed_pred - original_pred
        )

        if diff > THRESHOLD:
            status = "FAIL"
            color = (0, 0, 255)
        else:
            status = "PASS"
            color = (0, 255, 0)

        # ------------------------------------------
        # DISPLAY LEFT IMAGE
        # ------------------------------------------
        display = frame.copy()

        cv2.putText(
            display,
            f"Original: {original_pred:.4f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            f"Brightness: {transformed_pred:.4f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.putText(
            display,
            f"Diff: {diff:.5f}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        cv2.putText(
            display,
            status,
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3
        )

        # ------------------------------------------
        # LABEL TRANSFORMED IMAGE
        # ------------------------------------------
        transformed_view = transformed.copy()

        cv2.putText(
            transformed_view,
            "Brightness MR",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        # ------------------------------------------
        # COMBINE
        # ------------------------------------------
        transformed_view = cv2.resize(
            transformed_view,
            (
                display.shape[1],
                display.shape[0]
            )
        )

        combined = cv2.hconcat([
            display,
            transformed_view
        ])

        cv2.imshow(
            "AutoMR Live Webcam Demo",
            combined
        )

    except Exception as e:

        cv2.putText(
            frame,
            f"ERROR: {str(e)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        cv2.imshow(
            "AutoMR Live Webcam Demo",
            frame
        )

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break

# --------------------------------------------------
# CLEANUP
# --------------------------------------------------
cap.release()
cv2.destroyAllWindows()

print("Demo closed")