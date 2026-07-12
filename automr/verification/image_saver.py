import os
import cv2
import pandas as pd
from threading import Lock


class TransformationSaver:

    def __init__(
        self,
        output_dir="results/transformation_samples",
        max_examples=10,
    ):

        self.output_dir = output_dir
        self.max_examples = max_examples

        self.counts = {}
        self.metadata = []
        self.lock = Lock()

        self.metadata_file = os.path.join(
            self.output_dir,
            "metadata.csv",
        )

        self.summary_file = os.path.join(
            self.output_dir,
            "transformation_summary.csv",
        )

        os.makedirs(
            self.output_dir,
            exist_ok=True,
        )

    def save(
        self,
        mr_name,
        param,
        original,
        transformed,
        prediction_original=None,
        prediction_transformed=None,
        difference=None,
    ):

        with self.lock:

            current = self.counts.get(mr_name, 0)

            if current >= self.max_examples:
                return

            self.counts[mr_name] = current + 1

        mr_dir = os.path.join(
            self.output_dir,
            mr_name,
        )

        os.makedirs(
            mr_dir,
            exist_ok=True,
        )

        original_filename = (
            f"{mr_name}_{current:03d}_{param:.2f}_original.jpg"
        )

        transformed_filename = (
            f"{mr_name}_{current:03d}_{param:.2f}_transformed.jpg"
        )

        original_file = os.path.join(
            mr_dir,
            original_filename,
        )

        transformed_file = os.path.join(
            mr_dir,
            transformed_filename,
        )

        cv2.imwrite(original_file, original)
        cv2.imwrite(transformed_file, transformed)

        with self.lock:

            self.metadata.append(
                {
                    "mr": mr_name,
                    "parameter": float(param),
                    "original_file": original_file,
                    "transformed_file": transformed_file,
                    "prediction_original": prediction_original,
                    "prediction_transformed": prediction_transformed,
                    "difference": difference,
                }
            )

    def flush(self):

        with self.lock:

            if not self.metadata:
                return

            df = pd.DataFrame(self.metadata)

        df.to_csv(
            self.metadata_file,
            index=False,
        )

        summary = (
            df.groupby("mr")
            .agg(
                {
                    "parameter": ["min", "max", "count"],
                    "difference": ["mean", "max"],
                }
            )
            .round(6)
        )

        summary.columns = [
            "_".join(col)
            for col in summary.columns
        ]

        summary.reset_index().to_csv(
            self.summary_file,
            index=False,
        )

    def get_metadata(self):

        with self.lock:
            return pd.DataFrame(self.metadata)

    def clear_metadata(self):

        with self.lock:
            self.metadata.clear()
            self.counts.clear()

        if os.path.exists(self.metadata_file):
            os.remove(self.metadata_file)

        if os.path.exists(self.summary_file):
            os.remove(self.summary_file)