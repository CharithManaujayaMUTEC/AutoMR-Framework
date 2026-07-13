"""
Input handler factory.

This module provides a factory function for creating the
appropriate input handler based on the specified input type.
"""

from .image_handler import ImageHandler
from .tabular_handler import TabularHandler
from .text_handler import TextHandler


def get_handler(input_type):
    """
    Return an input handler for the requested input type.

    Parameters
    ----------
    input_type : str
        Supported input type.

    Returns
    -------
    BaseInputHandler
        Appropriate handler instance.

    Raises
    ------
    ValueError
        If the input type is unsupported.
    """

    # Image input handler.
    if input_type == "image":
        return ImageHandler()

    # Tabular input handler.
    if input_type == "tabular":
        return TabularHandler()

    # Text input handler.
    if input_type == "text":
        return TextHandler()

    raise ValueError(
        f"Unsupported input type: {input_type}"
    )