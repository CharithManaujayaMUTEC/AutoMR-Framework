import sys
import time
import numpy as np
import cv2
import multiprocessing

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

    # --------------------------------------------------
    # AUTOMR
    # --------------------------------------------------
    automr = AutoMR(
        model=model,
        task="regression",
        input_type="image",
        epsilon=0.05,
        strict=True
    )

    print("\nRegistered Transformations:")
    print(automr.list_transforms())

    print("\nRegistered Relations:")
    print(automr.list_relations())

    # --------------------------------------------------
    # RUN TEST
    # --------------------------------------------------
    start_time = time.time()

    df, results = automr.run_full_test(
        dataset=dataset,
        max_samples=2047,      # user can change
        samples_per_mr=5,      # user can change
        show_progress=True,
        save=True,
        output_dir="results",
        verbose=True
    )

    end_time = time.time()

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------
    runtime = end_time - start_time

    print("\n✅ DONE")
    print(f"Runtime: {runtime:.2f} seconds")

    print("\nResults saved to:")
    print("results/automr_results.csv")
    print("results/failure_summary.csv")
    print("results/severity_summary.csv")
    print("results/worst_cases.csv")
    print("results/failure_regions.txt")