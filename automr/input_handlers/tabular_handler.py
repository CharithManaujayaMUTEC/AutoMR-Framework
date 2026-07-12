from .base_handler import BaseInputHandler


class TabularHandler(BaseInputHandler):

    def validate(self, data):
        return data is not None

    def preprocess(self, data):
        return data

    def batch(self, data, batch_size):
        n = len(data)
        for i in range(0, n, batch_size):
            yield data[i:min(i + batch_size, n)]