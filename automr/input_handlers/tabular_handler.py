from .base_handler import BaseInputHandler


class TabularHandler(BaseInputHandler):

    def validate(self, data):
        return data is not None

    def preprocess(self, data):
        return data

    def batch(self, data, batch_size):

        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]