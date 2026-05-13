import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

# paths
sys.path.append("D:/7th semester/FYP/PROGRESS/Datasets")
sys.path.append("D:/7th semester/FYP/PROGRESS/Models")

from load_data import load_images
from automr.api import AutoMR
from load_model import get_model
from automr.comparator import RegressionComparator
from automr.core.failure_analysis import FailureAnalyzer

# comparator
comparator = RegressionComparator(epsilon=0.002)


# model wrapper (GENERIC)
# model wrapper (GENERIC)
class RealModel:
    def __init__(self):
        self.model = get_model()

    def predict(self, x):
        if x is None:
            return 0.0

        #  IMPORTANT: match DAVE-2 preprocessing
        x = cv2.resize(x, (200, 66))
        x = x / 255.0
        x = np.expand_dims(x, axis=0)

        pred = self.model.predict(x, verbose=0)
        return float(pred.flatten()[0])


# dataset
dataset = load_images("D:/7th semester/FYP/PROGRESS/Datasets/train/images")


# INIT MODEL
# INIT MODEL
model = RealModel()

#  SANITY CHECK (VERY IMPORTANT)
print("Model loaded successfully")

test_img = dataset[0]
if test_img is not None:
    test_pred = model.predict(test_img)
    print(" Sample prediction:", test_pred)
else:
    print("First image is None")


# AutoMR
# AutoMR
automr = AutoMR(model, comparator)

all_results = []

# Temporal MR (use full dataset once OR reuse)
df_temp, _ = automr.run_mr(dataset, "temporal", samples=5)

# run with progress bar
for i, sample in enumerate(tqdm(dataset[:2047], desc="Running AutoMR")):

    if sample is None:
        continue

    # Image MRs (per sample)
    df_img = automr.run_all_mrs(sample, samples=5)

    # Combine
    df = pd.concat([df_img, df_temp], ignore_index=True)

    # metadata
    df["sample_id"] = i

    # expected behavior
    def get_expected(row):
        for k, v in automr.mr_config.items():
            if v["relation"].__class__.__name__ == row["mr"]:
                relation = v["relation"]
                return relation.expected() if hasattr(relation, "expected") else "Standard invariance (output should remain consistent)"
        return "N/A"

    df["expected_behavior"] = df.apply(get_expected, axis=1)

    # actual behavior
    df["actual_behavior"] = df["status"].apply(
        lambda x: "Consistent" if x == "PASS" else "Violation"
    )

    all_results.append(df)


# save results
final_df = pd.concat(all_results, ignore_index=True)
final_df.to_csv("automr_results_detailed.csv", index=False)

print("✅ DONE: automr_results_detailed.csv generated")

analyzer = FailureAnalyzer()

#  Failure rate per MR
failure_summary = analyzer.failure_rate_per_mr(final_df)
print("\n=== FAILURE RATE PER MR ===")
print(failure_summary)

#  Severity ranking
severity_summary = analyzer.severity_per_mr(final_df)
print("\n=== SEVERITY PER MR ===")
print(severity_summary)

#  Worst cases
worst = analyzer.worst_cases(final_df, top_k=10)
print("\n=== TOP 10 FAILURES ===")
print(worst)

#  Failure regions
regions = analyzer.failure_regions(final_df)
print("\n=== FAILURE REGIONS ===")
for k, v in regions.items():
    print(k, ":", v)

failure_summary.to_csv("failure_summary.csv", index=False)
severity_summary.to_csv("severity_summary.csv")
worst.to_csv("worst_cases.csv", index=False)