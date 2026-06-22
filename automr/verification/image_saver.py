import os
import cv2


class TransformationSaver:

    def __init__(
        self,
        output_dir="results/transformation_samples",
        max_examples=10
    ):

        self.output_dir = output_dir
        self.max_examples = max_examples

        self.counts = {}

        os.makedirs(
            output_dir,
            exist_ok=True
        )

    def save(
        self,
        mr_name,
        original,
        transformed
    ):

        current = self.counts.get(
            mr_name,
            0
        )

        if current >= self.max_examples:
            return

        mr_dir = os.path.join(
            self.output_dir,
            mr_name
        )

        os.makedirs(
            mr_dir,
            exist_ok=True
        )

        cv2.imwrite(
            os.path.join(
                mr_dir,
                f"{current}_original.jpg"
            ),
            original
        )

        cv2.imwrite(
            os.path.join(
                mr_dir,
                f"{current}_transformed.jpg"
            ),
            transformed
        )

        self.counts[mr_name] = current + 1