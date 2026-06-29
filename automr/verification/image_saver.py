import os
import cv2
import pandas as pd


class TransformationSaver:

    def __init__(
        self,
        output_dir="results/transformation_samples",
        max_examples=10
    ):

        self.output_dir = output_dir
        self.max_examples = max_examples

        self.counts = {}

        self.metadata = []

        self.metadata_file = os.path.join(
            self.output_dir,
            "metadata.csv"
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

    def save(
        self,
        mr_name,
        param,
        original,
        transformed,
        prediction_original=None,
        prediction_transformed=None,
        difference=None
    ):

        current = self.counts.get(
            mr_name,
            0
        )

        if current >= self.max_examples:
            return

        mr_dir = os.path.join(
            self.output_dir,
            mr_name
        )

        os.makedirs(
            mr_dir,
            exist_ok=True
        )

        original_filename = (
            f"{mr_name}_{param:.2f}_original.jpg"
        )

        transformed_filename = (
            f"{mr_name}_{param:.2f}_transformed.jpg"
        )

        original_file = os.path.join(
            mr_dir,
            original_filename
        )

        transformed_file = os.path.join(
            mr_dir,
            transformed_filename
        )

        cv2.imwrite(
            original_file,
            original
        )

        cv2.imwrite(
            transformed_file,
            transformed
        )

        self.metadata.append({
            "mr": mr_name,
            "parameter": float(param),
            "original_file": original_file,
            "transformed_file": transformed_file,
            "prediction_original": prediction_original,
            "prediction_transformed": prediction_transformed,
            "difference": difference
        })

        pd.DataFrame(
            self.metadata
        ).to_csv(
            self.metadata_file,
            index=False
        )

        self.counts[mr_name] = current + 1
        self.export_summary()

    def get_metadata(self):

        return pd.DataFrame(
            self.metadata
        )

    def clear_metadata(self):

        self.metadata = []

        if os.path.exists(
            self.metadata_file
        ):
            os.remove(
                self.metadata_file
            )

    def export_summary(self):

        if not self.metadata:
            return

        df = pd.DataFrame(self.metadata)

        summary = (
            df.groupby("mr")
            .agg({
                "parameter": ["min", "max", "count"],
                "difference": ["mean", "max"]
            })
            .round(6)
        )

        summary.columns = [
            "_".join(col)
            for col in summary.columns
        ]

        summary.reset_index().to_csv(
            os.path.join(
                self.output_dir,
                "transformation_summary.csv"
            ),
            index=False
        )