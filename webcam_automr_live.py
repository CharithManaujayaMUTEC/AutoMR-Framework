# run_dashboard.py

import sys
import cv2
import numpy as np

sys.path.append("D:/FYP 78SEm/Modals")

from load_model import get_model

from automr.api import AutoMR
from automr.dashboard import run_live_dashboard


class RealModel:

    def __init__(self):
        self.model = get_model()

    def preprocess(self, img):

        img = cv2.resize(
            img,
            (200, 66)
        )

        img = img / 255.0

        return img.astype(
            np.float32
        )

    def predict(self, x):

        if x is None:
            return 0.0

        x = self.preprocess(x)

        x = np.expand_dims(
            x,
            axis=0
        )

        pred = self.model.predict(
            x,
            verbose=0
        )

        return float(
            pred.flatten()[0]
        )

    def predict_batch(self, images):

        batch = []

        for img in images:

            if img is None:
                continue

            batch.append(
                self.preprocess(img)
            )

        if len(batch) == 0:
            return []

        batch = np.asarray(
            batch,
            dtype=np.float32
        )

        preds = self.model.predict(
            batch,
            verbose=0
        )

        return (
            preds.flatten()
            .tolist()
        )


print("Loading model...")

model = RealModel()

automr = AutoMR(
    model=model,
    task="regression",
    input_type="image",
    epsilon=0.05,
    strict=True
)

print("Model loaded")

run_live_dashboard(

    automr=automr,

    model=model,

    video_source=0,

    selected_mrs=[

        "brightness",
        "rotation",
        "translation",
        "noise",
        "blur",
        "contrast",
        "rain",
        "snow",
        "fog",
        "visibility",
        "darkness"
    ],

    custom_ranges={

        "brightness": 1.5,
        "rotation": 15,
        "translation": 20,
        "noise": 25,
        "blur": 5,
        "contrast": 1.8,
        "rain": 0.8,
        "snow": 0.8,
        "fog": 0.8,
        "visibility": 0.5,
        "darkness": 0.5
    },

    frame_skip=30,

    save_results=True,

    save_violations=True,

    output_dir="results/live_dashboard"
)