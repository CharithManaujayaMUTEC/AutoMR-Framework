import json
import pandas as pd
import numpy as np
from pathlib import Path


class BaselineEvaluator:

    def __init__(self, output_dir="results"):
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # --------------------------------------------------
    # Check whether cached predictions already exist
    # --------------------------------------------------
    def baseline_exists(self):
        return (
            self.output_dir /
            "original_predictions.csv"
        ).exists()

    # --------------------------------------------------
    # Load cached predictions
    # --------------------------------------------------
    def load_predictions(self):
        df = pd.read_csv(
            self.output_dir /
            "original_predictions.csv"
        )

        return df["prediction"].tolist()

    # --------------------------------------------------
    # Save predictions
    # --------------------------------------------------
    def save_predictions(self, predictions):

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

        preds = np.array(predictions)

        metrics = {
            "num_samples": int(len(preds)),
            "mean_prediction": float(np.mean(preds)),
            "std_prediction": float(np.std(preds)),
            "min_prediction": float(np.min(preds)),
            "max_prediction": float(np.max(preds))
        }

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