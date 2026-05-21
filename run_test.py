import sys
import numpy as np
import cv2

# paths (keep for now)
sys.path.append("D:/FYP 78SEm/Datasets")
sys.path.append("D:/FYP 78SEm/Modals")

from load_data import load_images
from load_model import get_model
from automr.api import AutoMR

# -------- MODEL WRAPPER --------
class RealModel:
    def __init__(self):
        self.model = get_model()

    def predict(self, x):
        if x is None:
            return 0.0

        x = cv2.resize(x, (200, 66))
        x = x / 255.0
        x = np.expand_dims(x, axis=0)

        pred = self.model.predict(x, verbose=0)
        return float(pred.flatten()[0])


# -------- LOAD DATA --------
dataset = load_images("D:/FYP 78SEm/Datasets/archive/trafic_data/train/images")

# -------- INIT MODEL --------
model = RealModel()

print("Model loaded successfully")

test_img = dataset[0]
if test_img is not None:
    print("Sample prediction:", model.predict(test_img))


# -------- AUTOMR --------
automr = AutoMR(model)

# -------- RUN EVERYTHING --------
df, results = automr.run_full_test(
    dataset,
    max_samples=10,
    samples_per_mr=5,
    show_progress=True,
    output_dir="results"
)

print("✅ DONE: All results saved")