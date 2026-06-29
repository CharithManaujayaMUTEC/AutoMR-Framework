import numpy as np
import onnxruntime as ort

from automr.interfaces import BaseModel


class ONNXWrapper(BaseModel):

    def __init__(self, model_path):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

    def predict(self, x):
        x = np.asarray(x, dtype=np.float32)

        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)

        pred = self.session.run(
            None,
            {self.input_name: x}
        )[0]

        return float(np.asarray(pred).flatten()[0])

    def predict_batch(self, xs):
        batch = np.asarray(xs, dtype=np.float32)

        preds = self.session.run(
            None,
            {self.input_name: batch}
        )[0]

        return np.asarray(preds).flatten().tolist()