ModelConfigsImages = {
    "num_classes" : 10,
    "lr": 0.1,
    "step_size": 30,
    "gamma": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-9,
    "num_epochs": 75,
    "model_path": "./data"
}

DataConfigsImages = {
    "crop_size": 32,
    "normailzation_metrics": [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)],
    "batch_size" : 128,
    "num_workers": 4,
    "data_path": "./data"
}