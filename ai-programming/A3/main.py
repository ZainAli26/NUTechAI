import argparse
from classification_module.ObjectClassifier import ClassifierType
from classification_module.DataPreprocessor import DataType
from classification_module.builder import getClassifier, getPreprocessor

def map_to_enum_classifier(value: str) -> ClassifierType:
    try:
        return ClassifierType[value]
    except KeyError:
        raise ValueError(f"Invalid classifier type: {value}")

def map_to_enum_data_type(value: str) -> DataType:
    try:
        return DataType[value]
    except KeyError:
        raise ValueError(f"Invalid classifier type: {value}")

def main():
    parser = argparse.ArgumentParser(description="Select classifier type")
    
    # Define the argument for classifier
    parser.add_argument(
        'classifier',  # Command-line argument name
        choices=[e.name for e in ClassifierType],  # Enum names as valid choices
        help="Classifier type to use. Options: CNN, SVM, LogisticRegression"
    )

    parser.add_argument(
        'data_type',  # Command-line argument name
        choices=[e.name for e in DataType],  # Enum names as valid choices
        help="Data type to use. Options: TABULAR, IMAGES, TEXT, TEXT_IMAGES"
    )

    parser.add_argument(
        'dataset_name',
        help="Data set to be used Options: cifar, obesity"
    )

    parser.add_argument(
        'data_path',
        help="Data set to be used it should be directory path for images and csv path for the tabular data"
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Map the string argument to the corresponding Enum
    classifier = map_to_enum_classifier(args.classifier)
    
    # Print the selected classifier
    print(f"Selected classifier: {classifier.name} ({classifier.value})")

    data_type = map_to_enum_data_type(args.data_type)

    print(f"Selected data type: {data_type.name} ({data_type.value})")

    dataset_name = args.dataset_name
    data_path = args.data_path

    preprocessor = getPreprocessor(data_type, dataset_name, data_path)
    classifier = getClassifier(classifier, preprocessor)
    classifier.preprocess_data()
    classifier.build_model()
    classifier.train_model()
    classifier.evaluate_model()

if __name__ == '__main__':
    main()

