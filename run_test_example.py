import sys
import time
import numpy as np
import cv2
import multiprocessing
import pickle
import os


# --------------------------------------------------
# USER PATHS (update these)
# --------------------------------------------------
sys.path.append("Path/to/your/dataset")
sys.path.append("Path/to/your/model")


from load_data import load_images
from load_model import get_model
from automr.api import AutoMR


# --------------------------------------------------
# MODEL WRAPPER (customize for your model)
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

    # Optional: enables AutoMR batch inference
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

        return preds.flatten().tolist()


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":

    multiprocessing.freeze_support()

    # --------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------
    dataset = load_images(
        "Path/to/your/dataset"
    )

    print(f"Dataset size: {len(dataset)}")

    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------
    model = RealModel()

    print("✅ Model loaded successfully")

    # sanity check
    if len(dataset) > 0 and dataset[0] is not None:
        print(
            "Sample prediction:",
            model.predict(dataset[0])
        )
    else:
        print("⚠ Dataset is empty")
        sys.exit()

    # --------------------------------------------------
    # AUTOMR
    # --------------------------------------------------
    automr = AutoMR(
        model=model,
        task="regression",
        input_type="image",
        range_threshold=5.0
    )

    print("\nRegistered Transformations:")
    print(automr.list_transforms())

    print("\nRegistered Relations:")
    print(automr.list_relations())

    # --------------------------------------------------
    # VALIDATION RUN
    # --------------------------------------------------
    print("\n===================================")
    print("Running AutoMR Validation Suite")
    print("===================================")
    print("Original Samples : 10")
    print("Samples per MR   : 5")
    print("Epsilon Range    : 0.01 - 0.20")
    print("Epsilon Steps    : 20")
    print("Range Threshold  : 5%\n")

    start_time = time.time()

    df, results = automr.run_full_test(
        dataset=dataset,
        max_samples=10,          # Validation run
        samples_per_mr=5,
        show_progress=True,
        save=True,
        output_dir="results",
        verbose=True,
        epsilon_min=0.01,
        epsilon_max=0.20,
        epsilon_count=20,
    )

    end_time = time.time()

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------
    runtime = end_time - start_time

    print("\n✅ VALIDATION COMPLETE")
    print(f"Runtime: {runtime:.2f} seconds")

    print("\nGenerated Reports")
    print("--------------------------------")

    reports = [
        "baseline_metrics.json",
        "model_summary.txt",
        "dataset_info.json",
        "original_predictions.csv",
        "automr_results.csv",
        "prediction_trace.csv",
        "range_summary.csv",
        "range_analysis.csv",
        "failure_summary.csv",
        "severity_summary.csv",
        "worst_cases.csv",
        "failure_regions.txt",
        "epsilon_summary.csv",
        "epsilon_report.txt"
    ]

    for report in reports:

        path = f"results/{report}"

        if os.path.exists(path):
            print(f"✅ {path}")
        else:
            print(f"❌ {path}")

    print("\nGenerated Verification Artifacts")
    print("--------------------------------")

    verification_files = [
        "results/transformation_samples/metadata.csv",
        "results/transformation_samples/transformation_summary.csv"
    ]

    for file in verification_files:

        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")

    # --------------------------------------------------
    # QUICK RESULTS OVERVIEW
    # --------------------------------------------------
    print("\nResults Overview")
    print("--------------------------------")

    print("\nGenerated Epsilon Results")
    print("--------------------------------")

    for item in os.listdir("results"):

        if item.startswith("epsilon_"):
            print(f"✅ results/{item}")

    try:
        print(f"Total MT Results: {len(df)}")
        print(f"Unique MRs Tested: {df['mr'].nunique()}")
        print(f"Pass Rate: {(df['passed'].mean()*100):.2f}%")
    except Exception:
        pass

    # --------------------------------------------------
    # PICKLE CHECK
    # --------------------------------------------------
    try:

        pickle.dumps(automr)

        print("\n✅ PICKLE OK")

    except Exception as e:

        print("\n❌ PICKLE FAILED")
        print(e)

    print("\n===================================")
    print("AutoMR Validation Finished")
    print("===================================")