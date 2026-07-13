"""
Logging module for AutoMR.

This module provides a simple logger used throughout the AutoMR
framework to record execution events and runtime information.
"""

import logging
import os


class AutoMRLogger:
    """
    Wrapper around Python's logging module for AutoMR.
    """

    def __init__(self, log_dir="logs"):
        """
        Initialize the AutoMR logger.

        Parameters
        ----------
        log_dir : str, default="logs"
            Directory where log files are stored.
        """

        # Create the log directory if it does not exist.
        os.makedirs(
            log_dir,
            exist_ok=True
        )

        # Create or retrieve the framework logger.
        self.logger = logging.getLogger(
            "AutoMR"
        )

        # Configure the logging level.
        self.logger.setLevel(
            logging.INFO
        )

        # Prevent duplicate handlers if the logger
        # has already been initialized.
        if not self.logger.handlers:

            # Write log messages to a file.
            fh = logging.FileHandler(
                f"{log_dir}/automr.log"
            )

            # Define the log message format.
            formatter = logging.Formatter(
                "%(asctime)s | %(message)s"
            )

            fh.setFormatter(
                formatter
            )

            # Register the file handler.
            self.logger.addHandler(
                fh
            )

    def log(self, msg):
        """
        Write an informational message to the log.

        Parameters
        ----------
        msg : str
            Message to be recorded.
        """
        self.logger.info(msg)