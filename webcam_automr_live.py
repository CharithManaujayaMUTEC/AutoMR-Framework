import sys
import cv2
import numpy as np
import pandas as pd

# --------------------------------------------------
# USER PATHS
# --------------------------------------------------
sys.path.append("D:/FYP 78SEm/Modals")

from load_model import get_model
from automr.api import AutoMR


# --------------------------------------------------
# MODEL WRAPPER
# --------------------------------------------------
class RealModel:

    def __init__(self):
        self.model = get_model()

    def preprocess(self, img):
        img = cv2.resize(img, (200, 66))
        img = img / 255.0
        return img.astype(np.float32)

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

automr = AutoMR(
    model=model,
    task="regression",
    input_type="image",
    epsilon=0.05,
    strict=True
)

print("✅ Model loaded")

# --------------------------------------------------
# WEBCAM
# --------------------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("✅ Webcam started")

frame_count = 0
FRAME_SKIP = 30

latest_results = []

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    if frame_count % FRAME_SKIP == 0:

        latest_results = []

        original_pred = model.predict(frame)

        for mr_name in automr.list_transforms():

            if mr_name == "temporal":
                continue

            if mr_name not in automr.list_relations():
                continue

            try:

                transform = automr.transform_registry.get(
                    mr_name
                )

                relation = automr.relation_registry.get(
                    mr_name
                )

                start, end = automr.mr_ranges[mr_name]

                param = (start + end) / 2

                transformed = transform(
                    frame.copy(),
                    param
                )

                transformed_pred = model.predict(
                    transformed
                )

                diff = abs(
                    transformed_pred -
                    original_pred
                )

                latest_results.append({
                    "mr": mr_name,
                    "original": original_pred,
                    "transformed": transformed_pred,
                    "difference": diff
                })

            except Exception as e:

                latest_results.append({
                    "mr": mr_name,
                    "original": original_pred,
                    "transformed": original_pred,
                    "difference": 0.0
                })

        pd.DataFrame(
            latest_results
        ).to_csv(
            "webcam_results.csv",
            index=False
        )

    display = frame.copy()

    y = 30

    cv2.putText(
        display,
        f"Prediction: {model.predict(frame):.4f}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    y += 40

    for row in latest_results:

        txt = (
            f"{row['mr']} "
            f"delta={row['difference']:.5f}"
        )

        cv2.putText(
            display,
            txt,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255,255,0),
            2
        )

        y += 25

    cv2.imshow(
        "AutoMR Live Webcam Demo",
        display
    )

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()