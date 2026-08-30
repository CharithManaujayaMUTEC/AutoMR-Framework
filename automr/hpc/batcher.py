"""
Batch Manager

Responsible for creating and executing inference batches.

Responsibilities
----------------
- Collect transformed samples
- Execute model inference in batches
- Automatically split very large batches
- Preserve output order

This module is independent of AutoMR and can be reused by
other execution engines.
"""

import math
import numpy as np


class BatchManager:
    """
    High-performance inference batch manager.

    Parameters
    ----------
    model : object
        Wrapped AutoMR model.

    batch_size : int
        Maximum inference batch size.
    """

    def __init__(
        self,
        model,
        batch_size=64,
    ):
        self.model = model
        self.batch_size = batch_size

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------

    def set_batch_size(self, batch_size):
        """
        Update batch size.
        """

        self.batch_size = int(batch_size)

    def num_batches(self, n_samples):
        """
        Number of batches required.
        """

        return math.ceil(
            n_samples / self.batch_size
        )

    # -------------------------------------------------
    # Batch iterator
    # -------------------------------------------------

    def batches(self, samples):
        """
        Yield batches while preserving order.
        """

        total = len(samples)

        for start in range(
            0,
            total,
            self.batch_size,
        ):

            end = min(
                start + self.batch_size,
                total,
            )

            yield samples[start:end]

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------

    def predict(self, samples):
        """
        Predict an arbitrary number of samples.

        Automatically splits into batches when needed.
        """

        if len(samples) == 0:
            return np.asarray([])

        outputs = []

        for batch in self.batches(samples):

            preds = self.model.predict_batch(batch)

            outputs.extend(preds)

        return np.asarray(
            outputs,
            dtype=np.float32,
        )

    # -------------------------------------------------
    # Cached prediction
    # -------------------------------------------------

    def predict_cached(
        self,
        samples,
        cache,
        cache_keys,
    ):
        """
        Predict only missing samples.

        Parameters
        ----------
        samples : list
            Samples requiring prediction.

        cache : PredictionCache

        cache_keys : list
            Cache key for every sample.

        Returns
        -------
        np.ndarray
        """

        outputs = [None] * len(samples)

        missing_samples = []
        missing_indices = []

        for i, key in enumerate(cache_keys):

            if cache.exists(key):

                outputs[i] = cache[key]

            else:

                missing_samples.append(
                    samples[i]
                )

                missing_indices.append(i)

        if missing_samples:

            preds = self.predict(
                missing_samples
            )

            for idx, pred in zip(
                missing_indices,
                preds,
            ):

                pred = float(pred)

                outputs[idx] = pred

                cache[cache_keys[idx]] = pred

        return np.asarray(
            outputs,
            dtype=np.float32,
        )

    # -------------------------------------------------
    # Information
    # -------------------------------------------------

    def __repr__(self):

        return (
            f"BatchManager("
            f"batch_size={self.batch_size})"
        )