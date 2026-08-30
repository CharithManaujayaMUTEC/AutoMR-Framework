"""
Text input handler.

This handler provides validation, preprocessing,
and batching functionality for text datasets.
"""

from .base_handler import BaseInputHandler


class TextHandler(BaseInputHandler):
    """
    Handles text input data.
    """

    def validate(self, data):
        """
        Validate text data.
        """
        return data is not None

    def preprocess(self, data):
        """
        Preprocess text data before inference.
        """
        return data

    def batch(self, data, batch_size):
        """
        Generate batches of text samples.
        """
        n = len(data)
        for i in range(0, n, batch_size):
            yield data[i:min(i + batch_size, n)]