"""
Transformation sample saver.

This module provides utilities for saving example transformed images
generated during metamorphic testing. It also records metadata and
summary statistics for later inspection.
"""

import os
import cv2
import pandas as pd
from threading import Lock


class TransformationSaver:
    """
    Saves representative transformation examples and their metadata.
    """

    def __init__(
        self,
        output_dir="results/transformation_samples",
        max_examples=10,
    ):
        """
        Initialize the transformation saver.

        Parameters
        ----------
        output_dir : str, default="results/transformation_samples"
            Directory where images and metadata are stored.
        max_examples : int, default=10
            Maximum number of examples saved per metamorphic relation.
        """

        self.output_dir = output_dir
        self.max_examples = max_examples

        # Track saved examples and metadata.
        self.counts = {}
        self.metadata = []
        self.lock = Lock()

        # Output file paths.
        self.metadata_file = os.path.join(
            self.output_dir,
            "metadata.csv",
        )

        self.summary_file = os.path.join(
            self.output_dir,
            "transformation_summary.csv",
        )

        # Create the output directory if necessary.
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
        """
        Save a transformation example and its metadata.

        Parameters
        ----------
        mr_name : str
            Name of the metamorphic relation.
        param : float
            Transformation parameter value.
        original : ndarray
            Original input image.
        transformed : ndarray
            Transformed image.
        prediction_original : float, optional
            Prediction for the original image.
        prediction_transformed : float, optional
            Prediction for the transformed image.
        difference : float, optional
            Difference between predictions.
        """

        # Ensure thread-safe counting.
        with self.lock:

            current = self.counts.get(mr_name, 0)

            # Limit the number of stored examples.
            if current >= self.max_examples:
                return

            self.counts[mr_name] = current + 1

        # Create a directory for this MR.
        mr_dir = os.path.join(
            self.output_dir,
            mr_name,
        )

        os.makedirs(
            mr_dir,
            exist_ok=True,
        )

        # Generate output filenames.
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

        # Save image files.
        cv2.imwrite(original_file, original)
        cv2.imwrite(transformed_file, transformed)

        # Store metadata for later export.
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
        """
        Write metadata and summary statistics to disk.
        """

        with self.lock:

            if not self.metadata:
                return

            df = pd.DataFrame(self.metadata)

        # Save detailed metadata.
        df.to_csv(
            self.metadata_file,
            index=False,
        )

        # Generate summary statistics for each MR.
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

        # Flatten multi-level column names.
        summary.columns = [
            "_".join(col)
            for col in summary.columns
        ]

        # Save summary report.
        summary.reset_index().to_csv(
            self.summary_file,
            index=False,
        )

    def get_metadata(self):
        """
        Return the collected metadata as a DataFrame.
        """

        with self.lock:
            return pd.DataFrame(self.metadata)

    def clear_metadata(self):
        """
        Clear stored metadata and remove generated CSV files.
        """

        with self.lock:
            self.metadata.clear()
            self.counts.clear()

        # Remove metadata files if they exist.
        if os.path.exists(self.metadata_file):
            os.remove(self.metadata_file)

        if os.path.exists(self.summary_file):
            os.remove(self.summary_file)