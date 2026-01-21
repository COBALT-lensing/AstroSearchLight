import json
import io

from tqdm.auto import tqdm

from pathlib import Path

from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot
import numpy

from lightcurves import plot, load


def load_fits(path):
    lc = load(path)
    f = plot(lc)

    img_buf = io.BytesIO()
    f.savefig(img_buf, format="png")
    pyplot.close(f)
    img_buf.seek(0)
    im = Image.open(img_buf)
    return im.convert("RGB")


class LightcurveFolderDataSet(torchvision.datasets.DatasetFolder):
    def __init__(self, root, *args, **kwargs):
        super().__init__(
            root, load_fits, *args, extensions=(".fits", ".fits.gz"), **kwargs
        )


class LightcurvePathsDataSet(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = load_fits(self.paths[idx])

        if self.transform:
            image = self.transform(image)

        return image, self.paths[idx]


class ResNetFilter(object):
    MODEL_SAVE_PATH = Path("data/cnn/model_weights.pth")
    STOPPING_PATIENCE = 5
    STOPPING_SIGMA = 5
    LOADER_BATCH_SIZE = 32
    LOADER_WORKERS = 8

    def __init__(
        self, device=None, lr=0.001, scheduler_step_size=5, scheduler_gamma=0.5
    ):
        self.MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.CLASS_SAVE_PATH = self.MODEL_SAVE_PATH.parent / "model_classes.json"
        if device is not None:
            self.device = device
        else:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.mps.is_available() else "cpu"
            )

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224)
                ),  # Resizing the images to fit ResNet input
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.model = models.resnet34(weights="ResNet34_Weights.IMAGENET1K_V1")
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(self.model.fc.in_features, 2)
        )  # Change the output layer to have two classes with dropout
        self.model = torch.nn.DataParallel(self.model.to(self.device))

        self.model.to(self.device)

        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        self.scheduler = StepLR(
            self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma
        )

    def train(self, path):
        train_dataset = LightcurveFolderDataSet(root=path, transform=self.transform)

        self.classes = train_dataset.classes
        print("Training classes:", self.classes)

        # Split the dataset into training and testing sets (80% train, 20% test)
        train_size = int(0.8 * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, test_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.LOADER_BATCH_SIZE,
            shuffle=True,
            num_workers=self.LOADER_WORKERS,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.LOADER_BATCH_SIZE,
            shuffle=False,
            num_workers=self.LOADER_WORKERS,
        )

        self.model.train()  # Set the model to training mode

        accuracies = []
        validation_losses = []
        training_losses = []

        final_weights = None

        # Training loop
        epoch = 0
        while True:
            training_loss = 0.0
            validation_loss = 0.0
            correct = 0
            total = 0

            epoch += 1

            for inputs, labels in tqdm(train_loader, desc="Train", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                training_loss += loss.item()

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

            for inputs, labels in tqdm(test_loader, desc="Test", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)
                validation_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            self.scheduler.step()

            training_losses.append(training_loss)
            validation_losses.append(validation_loss)

            accuracies.append(100 * correct / total)
            # accuracies = accuracies[-self.STOPPING_PATIENCE :]
            validation_sigma = numpy.std(validation_losses[-self.STOPPING_PATIENCE :])

            print(
                f"Epoch [{epoch}], Training Loss: {training_loss:.4f}, "
                f"Validation Loss: {validation_loss:.4f}, "
                f"Accuracy: {accuracies[-1]:.2f}%, Validation std: {validation_sigma:.2f}"
            )
            if (
                len(validation_losses) >= self.STOPPING_PATIENCE
                and validation_sigma <= self.STOPPING_SIGMA
                and validation_losses[-1] > validation_losses[-2]
            ):
                break
            else:
                final_weights = self.model.state_dict()

        torch.save(final_weights, self.MODEL_SAVE_PATH)
        with open(self.CLASS_SAVE_PATH, "w") as class_f:
            json.dump(self.classes, class_f)

    def classify(self, dataset):
        self.model.load_state_dict(torch.load(self.MODEL_SAVE_PATH))
        with open(self.CLASS_SAVE_PATH, "r") as class_f:
            self.classes = json.load(class_f)
        self.model.eval()

        loader = DataLoader(
            dataset, batch_size=self.LOADER_BATCH_SIZE, num_workers=self.LOADER_WORKERS
        )

        out_labels = {}

        for images, paths in tqdm(loader, desc="Classify batches"):
            images = images.to(self.device)
            with torch.no_grad():
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)

            for i in range(len(images)):
                out_labels[paths[i]] = [self.classes[preds[i].item()]] + outputs[
                    i
                ].tolist()

        return out_labels

    def classify_folder(self, path):
        return self.classify(
            LightcurveFolderDataSet(root=path, transform=self.transform)
        )

    def classify_paths(self, paths):
        return self.classify(
            LightcurvePathsDataSet(paths=paths, transform=self.transform)
        )
