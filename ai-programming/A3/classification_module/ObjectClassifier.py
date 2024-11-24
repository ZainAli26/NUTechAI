from abc import ABC, abstractmethod
from enum import Enum
from classification_module.DataPreprocessor import DataType
from classification_module.DataPreprocessor import DataPreprocessor


class ClassifierType(Enum):
    CNN = 1
    SVM = 2
    LogisticRegression = 3


class ObjectClassifier:

    def __init__(self, data_preprocessor: DataPreprocessor):
        self.data_preprocessor = data_preprocessor
    @abstractmethod
    def preprocess_data(self):
        pass
    @abstractmethod
    def build_model(self):
        pass
    @abstractmethod
    def train_model(self):
        pass
    @abstractmethod
    def evaluate_model(self):
        pass
    @abstractmethod
    def predict(self, samples: list) -> list:
        pass
