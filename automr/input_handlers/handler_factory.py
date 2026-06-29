from .image_handler import ImageHandler
from .tabular_handler import TabularHandler
from .text_handler import TextHandler


def get_handler(input_type):

    if input_type == "image":
        return ImageHandler()

    if input_type == "tabular":
        return TabularHandler()

    if input_type == "text":
        return TextHandler()

    raise ValueError(
        f"Unsupported input type: {input_type}"
    )