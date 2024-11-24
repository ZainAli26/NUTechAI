from classification_module.DataPreprocessor import DataType, DataPreprocessor
from classification_module.ObjectClassifier import ClassifierType, ObjectClassifier
from classification_module.ImagePreprocessor import ImagePreprocessor
from classification_module.CNNClassifier import CNNClassifier
from classification_module.TabularDataPreprocessor import TabularDataPreprocessor
from classification_module.LogisticRegressionClassifier import LogisticRegressionClassifier
from classification_module.SVMClassifier import SVMClassifier

def getPreprocessor(data_type: DataType, dataset_name: str, data_path: str) -> DataPreprocessor:
    data_preprocessor: Optional[DataPreprocessor]
    if data_type == DataType.IMAGES:
        data_preprocessor = ImagePreprocessor(dataset_name, data_path)
    elif data_type == DataType.TABULAR:
        data_preprocessor = TabularDataPreprocessor(dataset_name, data_path)
    else:
        pass
    return data_preprocessor

def getClassifier(classifier: ClassifierType, data_preprocessor: DataPreprocessor) -> ObjectClassifier:
    classifier: Optional[ObjectClassifier]
    if classifier == ClassifierType.CNN:
        classifier = CNNClassifier(data_preprocessor)
    elif classifier == ClassifierType.LogisticRegression:
        classifier = LogisticRegressionClassifier(data_preprocessor)
    elif classifier == ClassifierType.SVM:
        classifier = SVMClassifier(data_preprocessor)
    else:
        pass
    return classifier 