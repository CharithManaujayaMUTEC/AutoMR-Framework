"""
Tabular input handler.

This handler provides validation, preprocessing,
and batching for tabular datasets.
"""

from .base_handler import BaseInputHandler


class TabularHandler(BaseInputHandler):
    """
    Handles tabular input data.
    """

    def validate(self, data):
        """
        Validate tabular data.
        """
        return data is not None

    def preprocess(self, data):
        """
        Preprocess tabular data before inference.
        """
        return data

    def batch(self, data, batch_size):
        """
        Generate batches of tabular samples.
        """
        n = len(data)
        for i in range(0, n, batch_size):
            yield data[i:min(i + batch_size, n)]