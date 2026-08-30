import sys
import os
import time
import pickle
import multiprocessing
import platform

import cv2
import numpy as np
import torch

# ==========================================================
# USER CONFIGURATION
# ==========================================================

# Add your project paths
sys.path.append("Path/to/your/dataset")
sys.path.append("Path/to/your/model")

# Dataset
DATASET_PATH = "Path/to/your/dataset"

# AutoMR settings
TASK = "regression"
INPUT_TYPE = "image"

# Testing
MAX_SAMPLES = None
SAMPLES_PER_MR = 5

# ==========================================================
# OPTIONAL TRANSFORMATION RANGE OVERRIDES
# ==========================================================
# Leave as None to use AutoMR's built-in defaults.
#
# Each entry may override:
#   start   -> minimum parameter value
#   end     -> maximum parameter value
#   samples -> number of values tested
#
# Only specify the transformations you want to customize.
# All others continue using the framework defaults.
# ==========================================================

TRANSFORM_RANGES = {

    "brightness": {
        "start": 0.5,
        "end": 2.0,
        "samples": 8,
    },

    "rotation": {
        "start": -25,
        "end": 25,
        "samples": 11,
    },

    "translation": {
        "start": 0,
        "end": 40,
        "samples": 9,
    },

    "noise": {
        "start": 0,
        "end": 60,
        "samples": 7,
    },

    "fog": {
        "start": 0.0,
        "end": 0.8,
        "samples": 6,
    },

}

# To use AutoMR defaults instead:
# TRANSFORM_RANGES = None

# MR
EPSILON = 0.05
RANGE_THRESHOLD = 5.0

# Epsilon sensitivity
ENABLE_EPSILON_ANALYSIS = True
EPSILON_MIN = 0.005
EPSILON_MAX = 0.05
EPSILON_COUNT = 3

# Backend
BACKEND = "auto"      # auto | cpu | gpu
DEVICE = "auto"       # auto | cpu | cuda

# Output
OUTPUT_DIR = "results"

SHOW_PROGRESS = True
SAVE_RESULTS = True
VERBOSE = True

# ==========================================================
# DEVICE CONFIGURATION
# ==========================================================

if DEVICE == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cpu":
    CPU_THREADS = os.cpu_count()
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(max(1, CPU_THREADS // 2))
    torch.backends.mkldnn.enabled = True
else:
    torch.backends.cudnn.benchmark = True

print(f"Backend : {BACKEND}")
print(f"Device  : {DEVICE}")

# ==========================================================
# IMPORTS
# ==========================================================

from load_data import load_images
from load_model import get_model

from automr.api import AutoMR
from automr.transforms.backend import set_backend

set_backend(BACKEND)

# ==========================================================
# MODEL WRAPPER
# ==========================================================

class RealModel:

    def __init__(self):

        self.model = get_model()

        if hasattr(self.model, "to"):
            self.model = self.model.to(DEVICE)

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
    print("AutoMR Validation")
    print("=" * 60)

    print(f"Platform         : {platform.system()}")
    print(f"Backend          : {BACKEND}")
    print(f"Device           : {DEVICE}")

    if DEVICE == "cpu":
        print(f"CPU Threads      : {os.cpu_count()}")
    else:
        print(f"CUDA Available   : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU              : {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------

    dataset = load_images(DATASET_PATH)

    print(f"Dataset Size     : {len(dataset)}")

    # ------------------------------------------------------

    model = RealModel()

    print("Model Loaded")

    sample = dataset[0]

    print("Sample Prediction:", model.predict(sample))

    # ------------------------------------------------------

    automr = AutoMR(

        model=model,

        task=TASK,

        input_type=INPUT_TYPE,

        epsilon=EPSILON,

        range_threshold=RANGE_THRESHOLD,

        transform_ranges=TRANSFORM_RANGES,

    )

    print("\nRegistered Transformations")
    print(automr.list_transforms())

    print("\nRegistered Relations")
    print(automr.list_relations())

    print("\nConfiguration")
    print("----------------------------")
    print(f"Task               : {TASK}")
    print(f"Input Type         : {INPUT_TYPE}")
    print(f"Backend            : {BACKEND}")
    print(f"Device             : {DEVICE}")
    print(f"Max Samples        : {MAX_SAMPLES}")
    print(f"Samples Per MR     : {SAMPLES_PER_MR}")
    print(f"Epsilon            : {EPSILON}")
    print(f"Range Threshold    : {RANGE_THRESHOLD}")

    if TRANSFORM_RANGES is None:
        print("Custom Ranges      : Disabled (using AutoMR defaults)")
    else:
        print("Custom Ranges      : Enabled")

        for name, cfg in TRANSFORM_RANGES.items():

            print(
                f"  {name:<15}"
                f"start={cfg.get('start','default')}, "
                f"end={cfg.get('end','default')}, "
                f"samples={cfg.get('samples','default')}"
            )

    if ENABLE_EPSILON_ANALYSIS:

        print(f"Epsilon Min        : {EPSILON_MIN}")
        print(f"Epsilon Max        : {EPSILON_MAX}")
        print(f"Epsilon Count      : {EPSILON_COUNT}")

    print()

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

    print()

    try:

        print(f"Total Results : {len(df)}")
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