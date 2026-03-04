import io
import json
from pathlib import Path

from PIL import Image
from matplotlib import pyplot
from tqdm.auto import tqdm

import numpy
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from lightcurves import plot, load
from utils import cache

IMAGE_DIMS = (224, 224)
PLOT_CACHE_PATH = "data/plots"

DEFAULT_SPLIT_SIZES = [2, 7, numpy.inf]


def plot_to_rgb(f):
    img_buf = io.BytesIO()
    f.savefig(img_buf, format="png")
    pyplot.close(f)
    img_buf.seek(0)
    im = Image.open(img_buf)
    im = im.convert("RGB").resize(IMAGE_DIMS)
    return im


@cache
def load_fits(path):
    return plot_to_rgb(plot(load(path)))


@cache
def load_fits_segment(path, relative_start_days, relative_end_days):
    lc = load(path)
    segment = _segment_lightcurve(lc, relative_start_days, relative_end_days)

    if len(segment) == 0:
        raise ValueError(
            f"No points found in segment ({relative_start_days}, {relative_end_days}) "
            f"for lightcurve {path}"
        )

    return plot_to_rgb(plot(segment))


def _relative_time_to_days(relative_time):
    if hasattr(relative_time, "to_value"):
        return float(relative_time.to_value("day"))
    return float(relative_time)


def _normalize_split_sizes(split_sizes):
    if not split_sizes:
        raise ValueError("split_sizes must not be empty")

    normalized_sizes = []
    for split_size in split_sizes:
        split_size_days = _relative_time_to_days(split_size)
        if split_size_days <= 0:
            raise ValueError(f"Each split size must be > 0. Got {split_size}.")
        normalized_sizes.append(split_size_days)

    return normalized_sizes


def _resolve_split_sizes(split_sizes):
    if split_sizes is None:
        return [numpy.inf]
    return _normalize_split_sizes(split_sizes)


def _segment_lightcurve(lc, relative_start_days, relative_end_days):
    time_jd = lc["time"].jd
    start_jd = time_jd[0] + relative_start_days
    if numpy.isinf(relative_end_days):
        mask = time_jd >= start_jd
    else:
        end_jd = time_jd[0] + relative_end_days
        mask = (time_jd >= start_jd) & (time_jd < end_jd)
    return lc[mask]


def _segment_windows_for_split_size(lc, split_size_days):
    if numpy.isinf(split_size_days):
        windows = [(0.0, numpy.inf, split_size_days)]
    else:
        time_jd = lc["time"].jd
        duration_days = float(numpy.max(time_jd) - time_jd[0])

        # Use ceil here to cover the full duration without adding an extra
        # end-aligned segment that can contain only a single point.
        n_segments = max(1, int(numpy.ceil(duration_days / split_size_days)))
        windows = [
            (
                segment_idx * split_size_days,
                (segment_idx + 1) * split_size_days,
                split_size_days,
            )
            for segment_idx in range(n_segments)
        ]

    return windows


def _segment_windows_for_path(path, split_sizes):
    lc = load(path)
    segment_windows = []
    for split_size_days in split_sizes:
        segment_windows.extend(_segment_windows_for_split_size(lc, split_size_days))
    return segment_windows


def _is_full_curve_segment(relative_start_days, relative_end_days):
    return relative_start_days <= 0 and numpy.isinf(relative_end_days)


def _load_fits_for_segment(path, relative_start_days, relative_end_days):
    if _is_full_curve_segment(relative_start_days, relative_end_days):
        return load_fits(path)
    return load_fits_segment(path, relative_start_days, relative_end_days)


def _is_lightcurve_file(path):
    path_s = str(path)
    return path_s.endswith(".fits") or path_s.endswith(".fits.gz")


def _find_folder_samples(root):
    root = Path(root)
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    classes = [d.name for d in class_dirs]
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    samples = []

    for class_dir in class_dirs:
        class_idx = class_to_idx[class_dir.name]
        for file_path in sorted(class_dir.rglob("*")):
            if file_path.is_file() and _is_lightcurve_file(file_path):
                samples.append((str(file_path), class_idx))

    return classes, class_to_idx, samples


class GroupedLightcurveFolderDataSet(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        split_sizes=None,
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split_sizes = _resolve_split_sizes(split_sizes)
        self.classes, self.class_to_idx, self.samples = _find_folder_samples(root)
        self.targets = [sample[1] for sample in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, class_idx = self.samples[idx]
        segment_windows = _segment_windows_for_path(path, self.split_sizes)
        if not segment_windows:
            segment_windows = [(0.0, numpy.inf, numpy.inf)]
        images = [
            _load_fits_for_segment(path, relative_start_days, relative_end_days)
            for relative_start_days, relative_end_days, _ in segment_windows
        ]
        if self.transform:
            images = [self.transform(image) for image in images]

        if self.target_transform is not None:
            class_idx = self.target_transform(class_idx)

        return images, class_idx


class GroupedLightcurvePathsDataSet(Dataset):
    def __init__(self, paths, transform=None, split_sizes=None):
        self.paths = list(paths)
        self.transform = transform
        self.split_sizes = _resolve_split_sizes(split_sizes)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        segment_windows = _segment_windows_for_path(path, self.split_sizes)
        if not segment_windows:
            segment_windows = [(0.0, numpy.inf, numpy.inf)]
        images = [
            _load_fits_for_segment(path, relative_start_days, relative_end_days)
            for relative_start_days, relative_end_days, _ in segment_windows
        ]
        if self.transform:
            images = [self.transform(image) for image in images]
        return images, path


def collate_segmented_batch(batch):
    segmented_items, targets = zip(*batch)
    batch_size = len(segmented_items)
    max_segments = max(len(item) for item in segmented_items)
    c, h, w = segmented_items[0][0].shape

    images = torch.zeros(
        batch_size, max_segments, c, h, w, dtype=segmented_items[0][0].dtype
    )
    mask = torch.zeros(batch_size, max_segments, dtype=torch.bool)

    for i, segments in enumerate(segmented_items):
        seg_count = len(segments)
        images[i, :seg_count] = torch.stack(segments, dim=0)
        mask[i, :seg_count] = True

    if isinstance(targets[0], int):
        targets = torch.tensor(targets, dtype=torch.long)
    else:
        targets = list(targets)

    return images, mask, targets


class SegmentAwareResNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        backbone = models.resnet34(weights="ResNet34_Weights.IMAGENET1K_V1")
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.segment_attention = nn.Linear(in_features, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x, mask):
        batch_size, n_segments, c, h, w = x.shape
        features = self.backbone(x.reshape(batch_size * n_segments, c, h, w))
        features = features.reshape(batch_size, n_segments, -1)

        attention_logits = self.segment_attention(features).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~mask, float("-inf"))
        attention = torch.softmax(attention_logits, dim=1)
        pooled = torch.sum(features * attention.unsqueeze(-1), dim=1)

        return self.classifier(self.dropout(pooled))


class LightcurveFolderDataSet(torchvision.datasets.DatasetFolder):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        split_sizes=None,
        is_valid_file=None,
    ):
        super().__init__(
            root,
            load_fits,
            extensions=(".fits", ".fits.gz"),
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.split_sizes = _resolve_split_sizes(split_sizes)

        base_samples = self.samples
        self.samples = [
            (path, class_index, relative_start_days, relative_end_days, split_size_days)
            for path, class_index in base_samples
            for relative_start_days, relative_end_days, split_size_days in _segment_windows_for_path(
                path, self.split_sizes
            )
        ]
        self.imgs = self.samples
        self.targets = [sample[1] for sample in self.samples]

    def __getitem__(self, idx):
        path, class_index, relative_start_days, relative_end_days, _ = self.samples[idx]
        image = _load_fits_for_segment(path, relative_start_days, relative_end_days)

        if self.transform:
            image = self.transform(image)
        if self.target_transform is not None:
            class_index = self.target_transform(class_index)

        return image, class_index


class LightcurvePathsDataSet(Dataset):
    def __init__(self, paths, transform=None, split_sizes=None):
        self.transform = transform
        self.split_sizes = _resolve_split_sizes(split_sizes)
        self.samples = [
            (path, relative_start_days, relative_end_days, split_size_days)
            for path in paths
            for relative_start_days, relative_end_days, split_size_days in _segment_windows_for_path(
                path, self.split_sizes
            )
        ]
        self.include_segment_in_id = not (
            len(self.split_sizes) == 1 and numpy.isinf(self.split_sizes[0])
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, relative_start_days, relative_end_days, split_size_days = self.samples[
            idx
        ]
        image = _load_fits_for_segment(path, relative_start_days, relative_end_days)

        if self.transform:
            image = self.transform(image)

        if self.include_segment_in_id:
            return image, (
                path,
                split_size_days,
                relative_start_days,
                relative_end_days,
            )

        return image, path


class ResNetFilter(object):
    MODEL_SAVE_PATH = Path("data/cnn/model_weights.pth")
    STOPPING_PATIENCE = 5
    STOPPING_SIGMA = 5
    LOADER_BATCH_SIZE = 32
    LOADER_WORKERS = 16

    def __init__(
        self,
        device=None,
        lr=0.001,
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        split_sizes=DEFAULT_SPLIT_SIZES,
    ):
        self.MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.CLASS_SAVE_PATH = self.MODEL_SAVE_PATH.parent / "model_classes.json"
        self.split_sizes = split_sizes
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
                    IMAGE_DIMS
                ),  # Resizing the images to fit ResNet input
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.model = SegmentAwareResNet(num_classes=2)
        self.model = torch.nn.DataParallel(self.model.to(self.device))

        self.model.to(self.device)

        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        self.scheduler = StepLR(
            self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma
        )

    def train(self, path):
        self.train_dataset = GroupedLightcurveFolderDataSet(
            root=path,
            transform=self.transform,
            split_sizes=self.split_sizes,
        )

        self.classes = self.train_dataset.classes

        # Split the dataset into training and testing sets (80% train, 20% test)
        train_size = int(0.8 * len(self.train_dataset))
        test_size = len(self.train_dataset) - train_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(
            self.train_dataset, [train_size, test_size]
        )

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.LOADER_BATCH_SIZE,
            shuffle=True,
            num_workers=self.LOADER_WORKERS,
            collate_fn=collate_segmented_batch,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.LOADER_BATCH_SIZE,
            shuffle=False,
            num_workers=self.LOADER_WORKERS,
            collate_fn=collate_segmented_batch,
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

            for inputs, mask, labels in tqdm(train_loader, desc="Train", leave=False):
                inputs = inputs.to(self.device)
                mask = mask.to(self.device)
                labels = labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs, mask)
                loss = self.criterion(outputs, labels)
                training_loss += loss.item()

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

            for inputs, mask, labels in tqdm(test_loader, desc="Test", leave=False):
                inputs = inputs.to(self.device)
                mask = mask.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs, mask)

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
        self.model.load_state_dict(
            torch.load(self.MODEL_SAVE_PATH, map_location=self.device)
        )
        with open(self.CLASS_SAVE_PATH, "r") as class_f:
            self.classes = json.load(class_f)
        self.model.eval()

        loader = DataLoader(
            dataset,
            batch_size=self.LOADER_BATCH_SIZE,
            num_workers=self.LOADER_WORKERS,
            collate_fn=collate_segmented_batch,
        )

        out_labels = {}

        for images, mask, paths in tqdm(loader, desc="Classify batches"):
            images = images.to(self.device)
            mask = mask.to(self.device)
            with torch.no_grad():
                outputs = self.model(images, mask)

            for i, path in enumerate(paths):
                logits = outputs[i].detach().cpu()
                pred = int(torch.argmax(logits).item())
                out_labels[path] = [self.classes[pred]] + logits.tolist()

        return out_labels

    def classify_folder(self, path):
        return self.classify_paths([str(p) for p in Path(path).glob("*.fits.gz")])

    def classify_paths(self, paths):
        return self.classify(
            GroupedLightcurvePathsDataSet(
                paths=paths, transform=self.transform, split_sizes=self.split_sizes
            )
        )
