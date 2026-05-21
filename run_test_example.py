import sys
import numpy as np
import cv2

# paths (USER updates these)
sys.path.append("Path/to/your/dataset")
sys.path.append("Path/to/your/model")

from load_data import load_images
from load_model import get_model
from automr.api import AutoMR

# -------- MODEL WRAPPER(change according to your model) --------
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
dataset = load_images("Path/to/your/dataset")

# -------- INIT MODEL --------
model = RealModel()

print("Model loaded successfully")

# sanity check
test_img = dataset[0]
if test_img is not None:
    print("Sample prediction:", model.predict(test_img))
else:
    print("First image is None")


# -------- AUTOMR --------
automr = AutoMR(model)

# -------- RUN FULL TEST  --------
df, results = automr.run_full_test(
    dataset,
    max_samples=2047,       # user can change
    samples_per_mr=5,       # user can change
    show_progress=True,
    output_dir="results"    # saved automatically
)

print("✅ DONE: Results saved in /results folder")