import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

# paths
sys.path.append("D:/FYP 78SEm/Datasets")
sys.path.append("D:/FYP 78SEm/Modals")

from load_data import load_images
from automr.api import AutoMR
from load_model import get_model
from automr.comparator import RegressionComparator

# ✅ comparator
comparator = RegressionComparator(epsilon=0.1)


# ✅ model wrapper (GENERIC)
class RealModel:
    def __init__(self):
        self.model = get_model()

    def predict(self, x):
        if x is None:
            return 0.0

        # 🔥 IMPORTANT: match DAVE-2 preprocessing
        x = cv2.resize(x, (200, 66))
        x = x / 255.0
        x = np.expand_dims(x, axis=0)

        pred = self.model.predict(x, verbose=0)
        return float(pred.flatten()[0])


# ✅ dataset
dataset = load_images("D:/FYP 78SEm/Datasets/archive/trafic_data/train/images")


# ✅ INIT MODEL
model = RealModel()

# 🔥 SANITY CHECK (VERY IMPORTANT)
print("✅ Model loaded successfully")

test_img = dataset[0]
if test_img is not None:
    test_pred = model.predict(test_img)
    print("🔍 Sample prediction:", test_pred)
else:
    print("⚠️ First image is None")


# ✅ AutoMR
automr = AutoMR(model, comparator)

all_results = []


# ✅ run with progress bar
for i, sample in enumerate(tqdm(dataset, desc="Running AutoMR")):

    if sample is None:
        continue

    df = automr.run_all_mrs(sample, samples=5)

    # metadata
    df["sample_id"] = i

    # expected behavior
    def get_expected(row):
        key = row["mr"].replace("Relation", "").lower()
        relation_obj = automr.mr_config[key]["relation"]
        return relation_obj.expected() if hasattr(relation_obj, "expected") else "N/A"

    df["expected_behavior"] = df.apply(get_expected, axis=1)

    # actual behavior
    df["actual_behavior"] = df["status"].apply(
        lambda x: "Consistent" if x == "PASS" else "Violation"
    )

    all_results.append(df)


# ✅ save results
final_df = pd.concat(all_results, ignore_index=True)
final_df.to_csv("automr_results_detailed.csv", index=False)

print("✅ DONE: automr_results_detailed.csv generated")