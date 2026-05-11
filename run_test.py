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
from automr.comparator import RegressionComparator

comparator = RegressionComparator(epsilon=0.1)

# ✅ model wrapper (GENERIC)
class RealModel:
    def __init__(self):
        self.model = get_model()

    def predict(self, x):
        if x is None:
            return 0.0

        # keep preprocessing OUTSIDE AutoMR (domain-specific)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)

        pred = self.model.predict(x, verbose=0)
        return float(pred.flatten()[0])


# ✅ dataset (can be anything later)
dataset = load_images("D:/FYP 78SEm/Datasets/archive/trafic_data/train/images")

model = RealModel()
automr = AutoMR(model, comparator)

all_results = []

# ✅ clean progress bar
for i, sample in enumerate(tqdm(dataset, desc="Running AutoMR")):

    if sample is None:
        continue

    df = automr.run_all_mrs(sample, samples=5)

    # ✅ attach metadata ONLY here (not in framework)
    df["sample_id"] = i

    # expected behavior from relation (NEW GENERIC WAY)
    def get_expected(row):
        relation_obj = automr.mr_config[row["mr"].replace("Relation", "").lower()]["relation"]
        return relation_obj.expected() if hasattr(relation_obj, "expected") else "N/A"

    df["expected_behavior"] = df.apply(get_expected, axis=1)

    # actual behavior
    df["actual_behavior"] = df["status"].apply(
        lambda x: "Consistent" if x == "PASS" else "Violation"
    )

    all_results.append(df)

# ✅ save
final_df = pd.concat(all_results, ignore_index=True)
final_df.to_csv("automr_results_detailed.csv", index=False)

print("✅ DONE: automr_results_detailed.csv generated")