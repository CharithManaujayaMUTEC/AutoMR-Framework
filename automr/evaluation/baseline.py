import json
import pandas as pd
import numpy as np
from pathlib import Path
from pathlib import Path


class BaselineEvaluator:

    def __init__(self, output_dir="results"):
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def save_predictions(self, predictions):

        pd.DataFrame(
            predictions
        ).to_csv(
            self.output_dir /
            "original_predictions.csv",
            index=False
        )

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