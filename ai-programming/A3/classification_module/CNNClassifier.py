import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt


from classification_module.ObjectClassifier import ObjectClassifier
from classification_module.DataPreprocessor import DataPreprocessor
from classification_module.config import ModelConfigsImages

class CNNClassifier(ObjectClassifier):
    
    def __init__(self, data_preprocessor: DataPreprocessor):
        super().__init__(data_preprocessor)
        self.metrics = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    
    def preprocess_data(self):
        self.train_loader, self.test_loader, self.classes = self.data_preprocessor.preprocess_data()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def build_model(self):
        self.model = ResnetClassifier().getModel()
        # Modify the fully connected layer for 10 output classes (CIFAR-10)
        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_features, ModelConfigsImages["num_classes"])

        # Move the model to the chosen device
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
        self.optimizer = optim.SGD(self.model.parameters(), lr=ModelConfigsImages["lr"], momentum=ModelConfigsImages["momentum"], weight_decay=ModelConfigsImages["weight_decay"])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=ModelConfigsImages["step_size"], gamma=ModelConfigsImages["gamma"])

    
    def train(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Track accuracy
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if (batch_idx + 1) % 100 == 0:
                print(f"Batch {batch_idx + 1}/{len(self.train_loader)} - Loss: {running_loss / (batch_idx + 1):.4f}")

        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = 100. * correct / total
        self.metrics["train_loss"].append(epoch_loss)
        self.metrics["train_accuracy"].append(epoch_accuracy)

        print(f"Train Epoch {epoch}: Loss: {running_loss / len(self.train_loader):.4f}, Accuracy: {100. * correct / total:.2f}%")

    def train_model(self):
        num_epochs = ModelConfigsImages["num_epochs"]
        best_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}/{num_epochs}")
            print('-' * 20)

            self.train(epoch)
            acc = self.validate()

            # Save the best model
            if acc > best_acc:
                best_acc = acc
                torch.save(self.model.state_dict(), os.path.join(ModelConfigsImages["model_path"] ,"best_cifar10_model.pth"))

            self.scheduler.step()  # Adjust learning rate

        print(f"Training complete. Best Accuracy: {best_acc:.2f}%")

    def evaluate_model(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics["train_loss"], label="Train Loss")
        plt.plot(self.metrics["val_loss"], label="Validation Loss")
        plt.title("Loss Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig("./data/cnn_loss_over_epochs.png")  # Save the figure

        # Plot accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics["train_accuracy"], label="Train Accuracy")
        plt.plot(self.metrics["val_accuracy"], label="Validation Accuracy")
        plt.title("Accuracy Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid()
        plt.savefig("./data/cnn_accuracy_over_epochs.png")  # Save the figure
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        epoch_loss = running_loss / len(self.test_loader)
        epoch_accuracy = 100. * correct / total
        self.metrics["val_loss"].append(epoch_loss)
        self.metrics["val_accuracy"].append(epoch_accuracy)

        print(f"Validation Loss: {running_loss / len(self.test_loader):.4f}, Accuracy: {100. * correct / total:.2f}%")
        return 100. * correct / total

    
    def predict(self, samples: list) -> list:
        predicted_labels = []
        for sample in samples:
            image_tensor = self.data_preprocessor.preprocess_sample(sample)
            image_tensor.to(self.device)

            self.model.eval()
            with torch.no_grad():
                outputs = self.model(image_tensor)
                _, predicted = outputs.max(1)

            predicted_labels.append(classes[predicted[0]])
        return predicted_labels
    

class ResnetClassifier():
    def getModel(self):
        # Load pretrained ResNet-18 model
        model = models.resnet18(pretrained=True)
        return model
