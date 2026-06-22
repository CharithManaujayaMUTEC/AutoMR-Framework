import json
import pandas as pd
from pathlib import Path

class BaselineEvaluator:

    def __init__(self, output_dir="results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def save_predictions(
        self,
        predictions
    ):

        pd.DataFrame(
            predictions
        ).to_csv(
            self.output_dir /
            "original_predictions.csv",
            index=False
        )

    def save_dataset_info(
        self,
        dataset_size
    ):

        info = {
            "dataset_size": dataset_size
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