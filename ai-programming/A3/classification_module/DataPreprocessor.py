from abc import ABC, abstractmethod
from enum import Enum

class DataType(Enum):
    TABULAR = 1
    IMAGES = 2
    TEXT = 3
    TEXT_IMAGES = 4


class DataPreprocessor:
    def __init__(self, dataset_name: str, data_path: str):
        self.dataset_name = dataset_name
        self.data_path = data_path
    @abstractmethod
    def preprocess_data(self):
        pass
    @abstractmethod
    def preprocess_sample(self, sample):
        pass