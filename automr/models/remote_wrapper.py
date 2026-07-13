"""
Remote model wrapper.

This wrapper enables AutoMR to communicate with machine learning
models hosted behind a REST API. It supports both single-sample and
batch prediction using HTTP requests.
"""

import numpy as np
import requests

from automr.interfaces import BaseModel


class RemoteWrapper(BaseModel):
    """
    Wrapper for remotely hosted machine learning models.
    """

    def __init__(self, endpoint):
        """
        Initialize the remote model wrapper.

        Parameters
        ----------
        endpoint : str
            URL of the remote inference endpoint.
        """
        self.endpoint = endpoint

    def predict(self, x):
        """
        Generate a prediction for a single input sample.
        """

        # Send the input to the remote prediction service.
        response = requests.post(
            self.endpoint,
            json={
                "input": np.asarray(x).tolist()
            },
            timeout=30
        )

        # Raise an exception for unsuccessful requests.
        response.raise_for_status()

        # Return the prediction from the server.
        return float(
            response.json()["prediction"]
        )

    def predict_batch(self, xs):
        """
        Generate predictions for multiple input samples.
        """

        # Send a batch prediction request.
        response = requests.post(
            self.endpoint,
            json={
                "inputs": np.asarray(xs).tolist()
            },
            timeout=60
        )

        # Return batch predictions if supported.
        if response.ok:
            return response.json()["predictions"]

        # Fall back to sequential requests if the server
        # only supports single-sample inference.
        return [self.predict(x) for x in xs]