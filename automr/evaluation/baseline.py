"""
Baseline evaluation module.

This module manages baseline artifacts generated before metamorphic
testing, including original predictions, dataset metadata, model
summaries, and prediction statistics.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path


class BaselineEvaluator:
    """
    Handles creation, storage, and retrieval of baseline evaluation
    data for AutoMR experiments.
    """

    def __init__(self, output_dir="results"):
        """
        Initialize the baseline evaluator.

        Parameters
        ----------
        output_dir : str, default="results"
            Directory where baseline files are stored.
        """
        self.output_dir = Path(output_dir)

        # Create the output directory if it does not exist.
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # --------------------------------------------------
    # Check whether cached predictions already exist
    # --------------------------------------------------
    def baseline_exists(self):
        """
        Check whether cached baseline predictions are available.

        Returns
        -------
        bool
            True if the cached prediction file exists.
        """
        return (
            self.output_dir /
            "original_predictions.csv"
        ).exists()

    # --------------------------------------------------
    # Load cached predictions
    # --------------------------------------------------
    def load_predictions(self):
        """
        Load cached baseline predictions.

        Returns
        -------
        list
            List of original model predictions.
        """
        df = pd.read_csv(
            self.output_dir /
            "original_predictions.csv"
        )

        return df["prediction"].tolist()

    # --------------------------------------------------
    # Save predictions
    # --------------------------------------------------
    def save_predictions(self, predictions):
        """
        Save baseline predictions to disk.

        Parameters
        ----------
        predictions : list
            Original model predictions.
        """

        pd.DataFrame(
            predictions
        ).to_csv(
            self.output_dir /
            "original_predictions.csv",
            index=False
        )

    # --------------------------------------------------
    # Save dataset information
    # --------------------------------------------------
    def save_dataset_info(self, dataset):
        """
        Save basic dataset metadata.

        Parameters
        ----------
        dataset : object
            Dataset used during evaluation.
        """

        info = {
            "dataset_size": len(dataset)
        }

        with open(
            self.output_dir /
            "dataset_info.json",
            "w"
        ) as f:

            json.dump(
                info,
                f,
                indent=4
            )

    # --------------------------------------------------
    # Save regression metrics
    # --------------------------------------------------
    def save_metrics(self, metrics):
        """
        Save baseline evaluation metrics.

        Parameters
        ----------
        metrics : dict
            Dictionary containing evaluation metrics.
        """

        with open(
            self.output_dir /
            "baseline_metrics.json",
            "w"
        ) as f:

            json.dump(
                metrics,
                f,
                indent=4
            )

    # --------------------------------------------------
    # Save model summary
    # --------------------------------------------------
    def save_model_summary(
        self,
        summary_text
    ):
        """
        Save a textual model summary.

        Parameters
        ----------
        summary_text : str
            Model description or architecture summary.
        """

        with open(
            self.output_dir /
            "model_summary.txt",
            "w",
            encoding="utf-8"
        ) as f:

            f.write(summary_text)

    # --------------------------------------------------
    # Save basic prediction statistics
    # --------------------------------------------------
    def save_basic_metrics(
        self,
        predictions
    ):
        """
        Compute and save basic prediction statistics.

        Parameters
        ----------
        predictions : list
            Collection of baseline predictions.
        """

        # Convert predictions to a NumPy array.
        preds = np.array(predictions)

        # Compute summary statistics.
        metrics = {
            "num_samples": int(len(preds)),
            "mean_prediction": float(np.mean(preds)),
            "std_prediction": float(np.std(preds)),
            "min_prediction": float(np.min(preds)),
            "max_prediction": float(np.max(preds))
        }

        # Save statistics to disk.
        with open(
            self.output_dir /
            "baseline_metrics.json",
            "w"
        ) as f:

            json.dump(
                metrics,
                f,
                indent=4
            )