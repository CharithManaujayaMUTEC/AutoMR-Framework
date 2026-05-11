import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# paths
sys.path.append("D:/FYP 78SEm/Datasets")
sys.path.append("D:/FYP 78SEm/Modals")

from load_data import load_images
from automr.api import AutoMR
from load_model import get_model

# ✅ expected behavior mapping
def get_expected(mr_name):
    if "Brightness" in mr_name:
        return "Output should remain approximately same under brightness change"
    elif "Rotation" in mr_name:
        return "Small rotation should not significantly change output"
    elif "Translation" in mr_name:
        return "Small translation should preserve prediction consistency"
    elif "Noise" in mr_name:
        return "Noise should not significantly affect prediction"
    return "Unknown MR"

# ✅ interpret result
def interpret(row):
    if row["status"] == "PASS":
        return "Consistent"
    return "Violation"

# ✅ model wrapper
class RealModel:
    def __init__(self):
        self.model = get_model()

    def predict(self, img):
        if img is None:
            return 0.0

        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        pred = self.model.predict(img, verbose=0)
        return float(pred.flatten()[0])

# ✅ load dataset
dataset = load_images("D:/FYP 78SEm/Datasets/archive/trafic_data/train/images")

model = RealModel()
automr = AutoMR(model)

all_results = []

# ✅ clean progress bar
for i, img in enumerate(tqdm(dataset, desc="Running AutoMR")):

    if img is None:
        continue

    res = automr.run_all_mrs(img, samples=5)

    # 🔥 add research-level details
    res["image_id"] = i
    res["expected_behavior"] = res["mr"].apply(get_expected)
    res["actual_behavior"] = res.apply(interpret, axis=1)

    all_results.append(res)

# ✅ save
final_df = pd.concat(all_results, ignore_index=True)
final_df.to_csv("automr_results_detailed.csv", index=False)

print("✅ DONE: automr_results_detailed.csv generated")