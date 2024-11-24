import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder

from classification_module.DataPreprocessor import DataPreprocessor

class TabularDataPreprocessor(DataPreprocessor):

    def __init__(self, dataset_name: str, data_path: str):
        super().__init__(dataset_name, data_path)

    def preprocess_data(self):
        dataset, labels, categorical_features = self.get_dataset()
        label_encoder = LabelEncoder()
        for feature in categorical_features:
            dataset[feature] = label_encoder.fit_transform(dataset[feature])
        labels = label_encoder.fit_transform(labels)
        label_mappings = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
        scaler = MinMaxScaler()

        columns = dataset.columns

        # Apply the scaler to the data (except the target if it's included)
        dataset = scaler.fit_transform(dataset)

        # Convert back to DataFrame for visualization
        dataset = pd.DataFrame(dataset, columns=columns)
        X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, label_mappings

    def preprocess_sample(image):
        pass

    def get_dataset(self):
        if self.dataset_name == 'obesity':
            return ObesityProcessor().get_dataset(self.data_path)

class ObesityProcessor():
    
    def __init__(self):
        super().__init__()
    
    def get_dataset(self, data_path):
        # Define data transformations
        obesity_df = pd.read_csv(data_path)
        obesity_df['BMI'] = obesity_df['Weight'] / (obesity_df['Height'] ** 2)
        categorical_features = ['Gender', 'family_history_with_overweight', 'CAEC', 'CALC', 'SCC', 'MTRANS', 'SMOKE', 'FAVC']
        y=obesity_df['NObeyesdad']
        obesity_df.drop(columns=["NObeyesdad"], inplace=True)
        return obesity_df, y, categorical_features