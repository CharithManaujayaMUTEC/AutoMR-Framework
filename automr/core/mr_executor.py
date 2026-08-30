"""
Metamorphic Relation (MR) execution engine.

This module executes multiple metamorphic relation jobs for a
single input sample by generating transformed inputs in parallel
and performing batched model inference to improve execution
efficiency.
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor


class MRExecutor:
    """
    Executes all metamorphic relations (MRs) for a single sample.

    Optimizations:
    ----------------
    1. Parallel transformation generation
    2. One large batch prediction
    3. Reuse original prediction
    4. Prediction cache for epsilon runs
    """

    def __init__(self, model):
        """
        Initialize the MR executor.

        Parameters
        ----------
        model : object
            Wrapped model providing prediction methods.
        """
        self.model = model

    def execute(
        self,
        input_data,
        mr_jobs,
        original_prediction,
    ):
        """
        Execute a collection of metamorphic relation jobs.

        Parameters
        ----------
        input_data : Any
            Original input sample.
        mr_jobs : list
            Collection of MR execution jobs.
        original_prediction : float
            Baseline prediction computed for the original input.

        Returns
        -------
        list
            Generated MR execution results containing transformed
            inputs and corresponding predictions.
        """

        # ---------------------------------------
        # Generate transformed images in parallel
        # ---------------------------------------
        def build(job):
            """
            Generate a transformed sample for a single MR job.
            """

            # Apply the transformation using the specified parameter.
            transformed = job["transform"](
                input_data,
                job["param"],
            )

            # Store metadata required for later verification.
            return {
                "mr": job["mr"],
                "relation": job["relation"],
                "param": job["param"],
                "image": transformed,
            }

        # Generate all transformed samples concurrently.
        with ThreadPoolExecutor() as executor:
            generated = list(
                executor.map(
                    build,
                    mr_jobs,
                )
            )

        # ---------------------------------------
        # Single batch inference
        # ---------------------------------------

        # Collect transformed inputs into a single batch.
        images = [
            g["image"]
            for g in generated
        ]

        # Perform batched prediction for improved efficiency.
        predictions = self.model.predict_batch(
            images
        )

        # ---------------------------------------
        # Attach predictions
        # ---------------------------------------

        # Associate predictions with their corresponding MR results.
        for g, pred in zip(
            generated,
            predictions,
        ):
            g["prediction"] = float(pred)

            # Reuse the previously computed baseline prediction.
            g["original"] = float(
                original_prediction
            )

        return generated