import requests
from automr.interfaces import BaseModel


class RemoteWrapper(BaseModel):

    def __init__(self, endpoint):
        self.endpoint = endpoint

    def predict(self, x):

        response = requests.post(
            self.endpoint,
            json={
                "input": x.tolist()
            },
            timeout=30
        )

        response.raise_for_status()

        return float(
            response.json()["prediction"]
        )

    def predict_batch(self, xs):

        preds = []

        for x in xs:
            preds.append(
                self.predict(x)
            )

        return preds