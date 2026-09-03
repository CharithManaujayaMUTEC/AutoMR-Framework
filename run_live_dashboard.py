"""
Run the AutoMR Live Dashboard against the real trained steering-angle model.

Model source: D:/7th semester/FYP/PROGRESS/Modals
    - Architecture: automr's load_model.py (NVIDIA-style CNN, 66x200x3 input)
    - Weights:      Modals/models/model.h5
    - Preprocessing matches Modals/car-behavioral-cloning/drive.py: crop out
      sky/hood, resize to 200x66, convert RGB->YUV. The model itself applies
      the x/127.5-1.0 normalization internally (Lambda layer), so no extra
      scaling is done here.

Run:
    python run_live_dashboard.py
"""

import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import Adam

from automr.api import AutoMR
from automr.dashboard import run_live_dashboard

MODEL_PATH = r"D:\7th semester\FYP\PROGRESS\Modals\models\model.h5"

# Original training images were 160px tall; the crop below removed the top
# 60px (sky) and bottom 25px (car hood). Applied proportionally here so it
# still makes sense against webcam frames of a different resolution.
_CROP_TOP_RATIO = 60 / 160
_CROP_BOTTOM_RATIO = 25 / 160


def _build_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="elu"))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="elu"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="elu"))
    model.add(Conv2D(64, (3, 3), activation="elu"))
    model.add(Conv2D(64, (3, 3), activation="elu"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation="elu"))
    model.add(Dense(50, activation="elu"))
    model.add(Dense(10, activation="elu"))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer=Adam(learning_rate=1e-4))
    model.load_weights(MODEL_PATH)
    return model


def _preprocess(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h = rgb.shape[0]
    top = int(round(h * _CROP_TOP_RATIO))
    bottom = int(round(h * _CROP_BOTTOM_RATIO))
    cropped = rgb[top: h - bottom, :, :]
    resized = cv2.resize(cropped, (200, 66), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_RGB2YUV)


class RealModel:
    """Steering-angle regression model, trained on Udacity sim frames."""

    def __init__(self):
        self.model = _build_model()

    def predict(self, x):
        batch = np.expand_dims(_preprocess(x), axis=0)
        return float(self.model.predict(batch, verbose=0)[0][0])

    def predict_batch(self, xs):
        batch = np.stack([_preprocess(x) for x in xs], axis=0)
        preds = self.model.predict(batch, verbose=0)
        return [float(p[0]) for p in preds]


model = RealModel()

automr = AutoMR(
    model=model,
    task="regression",
    input_type="image",
    range_threshold=0.5,  # steering angle output is roughly in [-1, 1]
)

video_source = 0  # USB webcam

run_live_dashboard(
    automr=automr,
    model=model,
    video_source=video_source,
    frame_skip=30,
    save_results=True,
    save_violations=True,
    output_dir="results/live_dashboard",
)
