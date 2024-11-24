from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

from classification_module.config import DataConfigsImages
from classification_module.DataPreprocessor import DataPreprocessor

class ImagePreprocessor(DataPreprocessor):

    def __init__(self, dataset_name: str, data_path: str):
        super().__init__(dataset_name, data_path)

    def preprocess_data(self):
        train_dataset, test_dataset = self.get_dataset()

        train_loader = DataLoader(train_dataset, batch_size=DataConfigsImages["batch_size"], shuffle=True, num_workers=DataConfigsImages["num_workers"])
        test_loader = DataLoader(test_dataset, batch_size=DataConfigsImages["batch_size"], shuffle=False, num_workers=DataConfigsImages["num_workers"])

        # Get class names
        classes = train_dataset.classes
        return train_loader, test_loader, classes

    def preprocess_sample(image):
        if self.dataset_name == 'cifar':
            transform = CIFARImageProcessor().getTransform()
        image = Image.open(image_path).convert('RGB').resize((32, 32))  # Resize to CIFAR-10 size
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor


    def get_dataset(self):
        if self.dataset_name == 'cifar':
            train_dataset, test_dataset = CIFARImageProcessor().get_dataset(self.data_path)
            return train_dataset, test_dataset

class CIFARImageProcessor():
    
    def __init__(self):
        super().__init__()
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(DataConfigsImages["crop_size"], padding=4),  # Data augmentation
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(DataConfigsImages["normailzation_metrics"][0], DataConfigsImages["normailzation_metrics"][1]),  # CIFAR-10 mean and std
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(DataConfigsImages["normailzation_metrics"][0], DataConfigsImages["normailzation_metrics"][1]),
        ])
    
    def get_dataset(self, data_path):
        # Define data transformations
        train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=self.transform_train)
        test_dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=self.transform_test)
        return train_dataset, test_dataset

    def getTransform(self):
        return self.transform_test