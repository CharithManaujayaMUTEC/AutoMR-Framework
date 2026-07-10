import numpy as np
import requests

from automr.interfaces import BaseModel


class RemoteWrapper(BaseModel):

    def __init__(self, endpoint):
        self.endpoint = endpoint

    def predict(self, x):

        response = requests.post(
            self.endpoint,
            json={
                "input": np.asarray(x).tolist()
            },
            timeout=30
        )

        response.raise_for_status()

        return float(
            response.json()["prediction"]
        )

    def predict_batch(self, xs):

        response = requests.post(
            self.endpoint,
            json={
                "inputs": np.asarray(xs).tolist()
            },
            timeout=60
        )

        if response.ok:
            return response.json()["predictions"]

        # fallback if server only supports single inference
        return [self.predict(x) for x in xs]