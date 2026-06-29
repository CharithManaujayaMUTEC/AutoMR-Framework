import logging
import os


class AutoMRLogger:

    def __init__(self, log_dir="logs"):

        os.makedirs(
            log_dir,
            exist_ok=True
        )

        self.logger = logging.getLogger(
            "AutoMR"
        )

        self.logger.setLevel(
            logging.INFO
        )

        if not self.logger.handlers:

            fh = logging.FileHandler(
                f"{log_dir}/automr.log"
            )

            formatter = logging.Formatter(
                "%(asctime)s | %(message)s"
            )

            fh.setFormatter(
                formatter
            )

            self.logger.addHandler(
                fh
            )

    def log(self, msg):
        self.logger.info(msg)