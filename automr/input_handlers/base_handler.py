from abc import ABC, abstractmethod


class BaseInputHandler(ABC):

    @abstractmethod
    def validate(self, data):
        pass

    @abstractmethod
    def preprocess(self, data):
        pass

    @abstractmethod
    def batch(self, data, batch_size):
        pass