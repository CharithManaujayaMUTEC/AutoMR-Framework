"""
Base input handler interface.

This module defines the abstract interface that all input handlers
must implement. Input handlers are responsible for validating,
preprocessing, and batching input data before it is passed to
the model.
"""

from abc import ABC, abstractmethod


class BaseInputHandler(ABC):
    """
    Abstract base class for all AutoMR input handlers.
    """

    @abstractmethod
    def validate(self, data):
        """
        Validate the input data before processing.
        """
        pass

    @abstractmethod
    def preprocess(self, data):
        """
        Preprocess the input data into the required format.
        """
        pass

    @abstractmethod
    def batch(self, data, batch_size):
        """
        Split input data into batches.
        """
        pass