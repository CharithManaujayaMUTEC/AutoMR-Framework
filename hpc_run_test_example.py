# examples/hpc_run_test.py

import sys
import os
import time
import pickle
import multiprocessing
import platform

import cv2
import numpy as np

# ==========================================================
# USER CONFIGURATION
# ==========================================================

# Add your project paths
sys.path.append("Path/to/your/dataset")
sys.path.append("Path/to/your/model")

# Dataset
DATASET_PATH = "Path/to/your/dataset"

# AutoMR Settings
TASK = "regression"
INPUT_TYPE = "image"

# Testing
MAX_SAMPLES = None          # None = Full dataset
SAMPLES_PER_MR = 5

# MR
EPSILON = 0.05
RANGE_THRESHOLD = 5.0

# Epsilon Sweep
ENABLE_EPSILON_ANALYSIS = True
EPSILON_MIN = 0.005
EPSILON_MAX = 0.05
EPSILON_COUNT = 3

# HPC Settings
WORKERS = os.cpu_count()
BATCH_SIZE = 64
CHUNK_SIZE = 64
PREFETCH = True
CACHE_PREDICTIONS = True

# Output
OUTPUT_DIR = "results_hpc"

SHOW_PROGRESS = True
SAVE_RESULTS = True
VERBOSE = True

# ==========================================================
# CPU SETTINGS
# ==========================================================

import torch

CPU_THREADS = os.cpu_count()

torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(max(1, CPU_THREADS // 2))
torch.backends.mkldnn.enabled = True

print(f"Using {CPU_THREADS} CPU threads")

# ==========================================================
# IMPORTS
# ==========================================================

from load_data import load_images
from load_model import get_model

from automr.hpc import HighPerformanceAutoMR

# ==========================================================
# MODEL WRAPPER
# ==========================================================

class RealModel:

    def __init__(self):
        self.model = get_model()

    def preprocess(self, img):

        img = cv2.resize(img, (200, 66))
        img = img.astype(np.float32) / 255.0

        return img

    def predict(self, image):

        if image is None:
            return 0.0

        x = self.preprocess(image)
        x = np.expand_dims(x, axis=0)

        prediction = self.model.predict(
            x,
            verbose=0,
        )

        return float(prediction.flatten()[0])

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
            dtype=np.float32,
        )

        predictions = self.model.predict(
            batch,
            verbose=0,
        )

        return predictions.flatten().tolist()

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    multiprocessing.freeze_support()

    print("=" * 60)
    print("HighPerformanceAutoMR Validation")
    print("=" * 60)

    print(f"Platform      : {platform.system()}")
    print(f"CPU Threads   : {CPU_THREADS}")

    dataset = load_images(DATASET_PATH)

    print(f"Dataset Size  : {len(dataset)}")

    model = RealModel()

    print("Model Loaded")

    sample = dataset[0]

    print("Sample Prediction:", model.predict(sample))

    automr = HighPerformanceAutoMR(

        model=model,

        task=TASK,

        input_type=INPUT_TYPE,

        epsilon=EPSILON,

        range_threshold=RANGE_THRESHOLD,

        workers=WORKERS,

        batch_size=BATCH_SIZE,

        chunk_size=CHUNK_SIZE,

        prefetch=PREFETCH,

        cache_predictions=CACHE_PREDICTIONS,

    )

    print("\nRegistered Transformations")
    print(automr.list_transforms())

    print("\nRegistered Relations")
    print(automr.list_relations())

    print("\nHPC Configuration")
    print("----------------------------")
    print(f"Workers            : {WORKERS}")
    print(f"Batch Size         : {BATCH_SIZE}")
    print(f"Chunk Size         : {CHUNK_SIZE}")
    print(f"Prefetch           : {PREFETCH}")
    print(f"Prediction Cache   : {CACHE_PREDICTIONS}")
    print(f"Max Samples        : {MAX_SAMPLES}")
    print(f"Samples Per MR     : {SAMPLES_PER_MR}")

    start = time.time()

    if ENABLE_EPSILON_ANALYSIS:

        df, results = automr.run_full_test(

            dataset=dataset,

            max_samples=MAX_SAMPLES,

            samples_per_mr=SAMPLES_PER_MR,

            show_progress=SHOW_PROGRESS,

            save=SAVE_RESULTS,

            output_dir=OUTPUT_DIR,

            verbose=VERBOSE,

            epsilon_min=EPSILON_MIN,

            epsilon_max=EPSILON_MAX,

            epsilon_count=EPSILON_COUNT,

        )

    else:

        df, results = automr.run_full_test(

            dataset=dataset,

            max_samples=MAX_SAMPLES,

            samples_per_mr=SAMPLES_PER_MR,

            show_progress=SHOW_PROGRESS,

            save=SAVE_RESULTS,

            output_dir=OUTPUT_DIR,

            verbose=VERBOSE,

        )

    runtime = time.time() - start

    print("\nValidation Complete")
    print(f"Runtime : {runtime:.2f} sec")

    print("\nGenerated Reports")

    reports = [

        "automr_results.csv",
        "failure_summary.csv",
        "severity_summary.csv",
        "prediction_trace.csv",
        "range_summary.csv",
        "range_analysis.csv",
        "worst_cases.csv",
        "failure_regions.txt",
        "baseline_metrics.json",
        "dataset_info.json",
        "model_summary.txt",
        "original_predictions.csv",

    ]

    if ENABLE_EPSILON_ANALYSIS:

        reports.extend([
            "epsilon_summary.csv",
            "epsilon_report.txt",
        ])

    for report in reports:

        path = os.path.join(
            OUTPUT_DIR,
            report,
        )

        if os.path.exists(path):
            print(f"[OK] {path}")
        else:
            print(f"[--] {path}")

    try:

        print(f"\nTotal Results : {len(df)}")
        print(f"Unique MRs    : {df['mr'].nunique()}")
        print(f"Pass Rate     : {(df['passed'].mean()*100):.2f}%")

    except Exception:
        pass

    try:

        pickle.dumps(automr)

        print("\nPickle : OK")

    except Exception as e:

        print("\nPickle Failed")
        print(e)

    print("\nFinished.")