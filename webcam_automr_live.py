# webcam_automr_demo.py

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
print("Press ESC to quit")

frame_count = 0
FRAME_SKIP = 30

latest_results = []

CELL_W = 320
CELL_H = 240

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    current_tiles = []

    # --------------------------------------------------
    # ORIGINAL
    # --------------------------------------------------
    original_pred = model.predict(frame)

    original_tile = frame.copy()

    cv2.putText(
        original_tile,
        f"Original",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        original_tile,
        f"{original_pred:.4f}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    current_tiles.append(original_tile)

    # --------------------------------------------------
    # RUN MRs + SAVE CSV
    # --------------------------------------------------
    if frame_count % FRAME_SKIP == 0:

        latest_results = []

        for mr_name in automr.list_transforms():

            if mr_name == "temporal":
                continue

            try:

                if mr_name not in automr.mr_ranges:
                    continue

                transform = automr.transform_registry.get(
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

            except Exception:
                pass

        if len(latest_results) > 0:

            pd.DataFrame(
                latest_results
            ).to_csv(
                "webcam_results.csv",
                index=False
            )

    # --------------------------------------------------
    # BUILD LIVE DASHBOARD
    # --------------------------------------------------
    for mr_name in automr.list_transforms():

        if mr_name == "temporal":
            continue

        try:

            transform = automr.transform_registry.get(
                mr_name
            )

            start, end = automr.mr_ranges[mr_name]

            param = (start + end) / 2

            transformed = transform(
                frame.copy(),
                param
            )

            pred = model.predict(
                transformed
            )

            cv2.putText(
                transformed,
                mr_name,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                transformed,
                f"{pred:.4f}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            current_tiles.append(
                transformed
            )

        except Exception:
            pass

    # --------------------------------------------------
    # RESIZE ALL TILES
    # --------------------------------------------------
    resized = []

    for img in current_tiles:

        resized.append(
            cv2.resize(
                img,
                (CELL_W, CELL_H)
            )
        )

    # --------------------------------------------------
    # GRID LAYOUT (3 COLUMNS)
    # --------------------------------------------------
    rows = []

    for i in range(0, len(resized), 3):

        row = resized[i:i+3]

        while len(row) < 3:

            row.append(
                np.zeros(
                    (CELL_H, CELL_W, 3),
                    dtype=np.uint8
                )
            )

        rows.append(
            np.hstack(row)
        )

    dashboard = np.vstack(rows)

    # --------------------------------------------------
    # SHOW
    # --------------------------------------------------
    cv2.imshow(
        "AutoMR Live Dashboard",
        dashboard
    )

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break

# --------------------------------------------------
# CLEANUP
# --------------------------------------------------
cap.release()
cv2.destroyAllWindows()

print("✅ Results saved to webcam_results.csv")