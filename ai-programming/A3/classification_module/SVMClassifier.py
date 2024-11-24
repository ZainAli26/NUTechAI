import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC

from classification_module.ObjectClassifier import ObjectClassifier
from classification_module.DataPreprocessor import DataPreprocessor

class SVMClassifier(ObjectClassifier):
    
    def __init__(self, data_preprocessor: DataPreprocessor):
        super().__init__(data_preprocessor)
    
    def preprocess_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test, self.label_mappings = self.data_preprocessor.preprocess_data()
        n_classes = len(np.unique(self.y_train))
        self.y_test_bin = label_binarize(self.y_test, classes=np.arange(n_classes))
    
    def build_model(self):
        self.svm_model = SVC(decision_function_shape='ovr',kernel='poly', probability=True, degree=5, random_state=42)

    def train_model(self):
        self.svm_model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_score = self.svm_model.predict_proba(self.X_test)

        # Make predictions on the test set
        y_pred = self.svm_model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")

        # Calculate F1 score
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        print(f'F1 Score: {f1:.2f}')

        conf_matrix = confusion_matrix(self.y_test, y_pred)
        print(f'Confusion Matrix: {conf_matrix}')

        xticketlabelvalues = []
        yticketlabelvalues = []
        for label in self.label_mappings.keys():
            xticketlabelvalues.append("Predicted " + label)
            yticketlabelvalues.append("Actual " + label)

        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    xticklabels=xticketlabelvalues, 
                    yticklabels=yticketlabelvalues)

        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.savefig("./data/svm-conf-matrix.png")

        # Compute ROC curve and ROC AUC for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        n_classes = len(np.unique(self.y_train))

        # Binarize the output labels (One-vs-Rest style for ROC curve)
        y_test_bin = np.zeros((self.y_test.shape[0], n_classes))
        for i in range(n_classes):
            y_test_bin[:, i] = (self.y_test == i).astype(int)

        # Loop over each class to calculate ROC curves
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])  # Use the probabilities for the current class
            roc_auc[i] = auc(fpr[i], tpr[i])  # Calculate the AUC for the current class

        # Plot ROC curves for each class
        plt.figure()
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', label="Random guessing")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic for Multinomial Logistic Regression')
        plt.legend(loc="lower right")
        plt.grid()
        plt.savefig("./data/svm-roc.png")

    
    def predict(self, samples: list) -> list:
        pass
