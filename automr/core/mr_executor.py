import numpy as np
from concurrent.futures import ThreadPoolExecutor


class MRExecutor:
    """
    Executes all MRs for a single sample.

    Optimizations:
    ----------------
    1. Parallel transformation generation
    2. One large batch prediction
    3. Reuse original prediction
    4. Prediction cache for epsilon runs
    """

    def __init__(self, model):
        self.model = model

    def execute(
        self,
        input_data,
        mr_jobs,
        original_prediction,
    ):

        # ---------------------------------------
        # Generate transformed images in parallel
        # ---------------------------------------
        def build(job):

            transformed = job["transform"](
                input_data,
                job["param"],
            )

            return {
                "mr": job["mr"],
                "relation": job["relation"],
                "param": job["param"],
                "image": transformed,
            }

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
        images = [
            g["image"]
            for g in generated
        ]

        predictions = self.model.predict_batch(
            images
        )

        # ---------------------------------------
        # Attach predictions
        # ---------------------------------------
        for g, pred in zip(
            generated,
            predictions,
        ):
            g["prediction"] = float(pred)
            g["original"] = float(
                original_prediction
            )

        return generated