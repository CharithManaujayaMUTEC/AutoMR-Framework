"""
Image input handler.

This handler provides basic validation, preprocessing,
and batching functionality for image datasets.
"""

from .base_handler import BaseInputHandler


class ImageHandler(BaseInputHandler):
    """
    Handles image input data.
    """

    def validate(self, data):
        """
        Validate image data.
        """
        return data is not None

    def preprocess(self, data):
        """
        Preprocess image data before inference.
        """
        return data

    def batch(self, data, batch_size):
        """
        Generate batches of image samples.
        """
        n = len(data)
        for i in range(0, n, batch_size):
            yield data[i:min(i + batch_size, n)]