import io
import json
import os
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


def _limit_segment_windows(segment_windows, max_segments_per_item):
    if max_segments_per_item is None:
        return segment_windows
    if max_segments_per_item <= 0:
        raise ValueError("max_segments_per_item must be > 0 or None")
    if len(segment_windows) <= max_segments_per_item:
        return segment_windows

    # Keep a representative spread across the full lightcurve.
    indices = numpy.linspace(
        0, len(segment_windows) - 1, num=max_segments_per_item, dtype=int
    )
    return [segment_windows[i] for i in indices]


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
        max_segments_per_item=None,
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split_sizes = _resolve_split_sizes(split_sizes)
        self.max_segments_per_item = max_segments_per_item
        self.classes, self.class_to_idx, self.samples = _find_folder_samples(root)
        self.targets = [sample[1] for sample in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, class_idx = self.samples[idx]
        segment_windows = _segment_windows_for_path(path, self.split_sizes)
        if not segment_windows:
            segment_windows = [(0.0, numpy.inf, numpy.inf)]
        segment_windows = _limit_segment_windows(
            segment_windows, self.max_segments_per_item
        )
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
    def __init__(
        self, paths, transform=None, split_sizes=None, max_segments_per_item=None
    ):
        self.paths = list(paths)
        self.transform = transform
        self.split_sizes = _resolve_split_sizes(split_sizes)
        self.max_segments_per_item = max_segments_per_item

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        segment_windows = _segment_windows_for_path(path, self.split_sizes)
        if not segment_windows:
            segment_windows = [(0.0, numpy.inf, numpy.inf)]
        segment_windows = _limit_segment_windows(
            segment_windows, self.max_segments_per_item
        )
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

    # Standard ResNet training expects one image per example. Here each example
    # is a variable-length list of segment images from the same light curve, so
    # we pad to the largest segment count in the batch and carry a mask that
    # marks which segment slots are real. The extra "padded slots" are dummy
    # zero-filled entries added only so every item in the batch has the same
    # tensor shape.
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


def _init_loader_worker(_worker_id):
    # Keep worker-side CPU parallelism bounded so multiple workers do not
    # multiply thread count and memory use.
    torch.set_num_threads(1)


class SegmentAwareResNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        backbone = models.resnet34(weights="ResNet34_Weights.IMAGENET1K_V1")
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        # A standard ResNet would send one image embedding directly into its
        # final classifier. This variant keeps the ResNet as a shared feature
        # extractor, then learns how to combine multiple segment embeddings from
        # the same light curve before classification.
        self.backbone = backbone
        # This small linear layer maps each segment embedding to a single score.
        # After softmax, those scores become attention weights that tell the
        # model how much each segment should contribute to the final pooled
        # light-curve representation.
        self.segment_attention = nn.Linear(in_features, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x, mask):
        batch_size, n_segments, c, h, w = x.shape

        # Flatten the batch of segment sets so every segment image goes through
        # the same ResNet backbone independently.
        features = self.backbone(x.reshape(batch_size * n_segments, c, h, w))
        features = features.reshape(batch_size, n_segments, -1)

        # self.segment_attention does not classify segments by itself. It learns
        # one scalar "importance" score per segment embedding. Softmax
        # normalizes those scores across the segments of the same light curve,
        # so segments with larger scores receive more weight in the pooled
        # representation. Padded slots are the fake zero-filled segment entries
        # added during batching for shorter light curves; they must be forced to
        # -inf first so softmax gives them zero attention instead of letting the
        # model treat missing segments as real observations.
        attention_logits = self.segment_attention(features).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~mask, float("-inf"))
        attention = torch.softmax(attention_logits, dim=1)

        # This is a weighted average over segment embeddings, producing one
        # feature vector for the full light curve before the final classifier.
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
    LOADER_BATCH_SIZE = 8
    LOADER_WORKERS = min(4, max(1, os.cpu_count() or 1))
    LOADER_PREFETCH_FACTOR = 1
    MAX_SEGMENTS_PER_ITEM = 24
    DEFAULT_NUM_CLASSES = 2

    def __init__(
        self,
        device=None,
        lr=0.001,
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        split_sizes=DEFAULT_SPLIT_SIZES,
        loader_batch_size=None,
        loader_workers=None,
        loader_prefetch_factor=None,
        max_segments_per_item=None,
        use_amp=True,
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
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

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

        self.loader_batch_size = (
            self.LOADER_BATCH_SIZE
            if loader_batch_size is None
            else int(loader_batch_size)
        )
        self.loader_workers = (
            self.LOADER_WORKERS if loader_workers is None else int(loader_workers)
        )
        self.loader_prefetch_factor = (
            self.LOADER_PREFETCH_FACTOR
            if loader_prefetch_factor is None
            else int(loader_prefetch_factor)
        )
        self.max_segments_per_item = (
            self.MAX_SEGMENTS_PER_ITEM
            if max_segments_per_item is None
            else int(max_segments_per_item)
        )
        self.pin_memory = self.device.type == "cuda"
        self.use_amp = bool(use_amp) and self.device.type == "cuda"
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.lr = lr
        self.weight_decay = 1e-4
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.num_classes = None
        self._set_model(self.DEFAULT_NUM_CLASSES)

        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self._reset_optimizer_and_scheduler()

    def _unwrap_model(self):
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module
        return self.model

    def _set_model(self, num_classes):
        num_classes = int(num_classes)
        if self.num_classes == num_classes and hasattr(self, "model"):
            return

        base_model = SegmentAwareResNet(num_classes=num_classes).to(self.device)
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(base_model)
        else:
            self.model = base_model
        self.num_classes = num_classes

    def _reset_optimizer_and_scheduler(self):
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = StepLR(
            self.optimizer,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma,
        )

    def _build_dataloader(self, dataset, shuffle):
        kwargs = {
            "dataset": dataset,
            "batch_size": self.loader_batch_size,
            "shuffle": shuffle,
            "num_workers": self.loader_workers,
            "collate_fn": collate_segmented_batch,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.loader_workers > 0,
        }
        if self.loader_workers > 0:
            kwargs["prefetch_factor"] = self.loader_prefetch_factor
            kwargs["worker_init_fn"] = _init_loader_worker
        return DataLoader(**kwargs)

    def _load_model_weights(self):
        state_dict = torch.load(self.MODEL_SAVE_PATH, map_location=self.device)
        target_model = self._unwrap_model()
        try:
            target_model.load_state_dict(state_dict)
            return
        except RuntimeError:
            pass

        # Backward compatibility with checkpoints saved from DataParallel.
        if all(key.startswith("module.") for key in state_dict.keys()):
            stripped = {k[len("module.") :]: v for k, v in state_dict.items()}
            target_model.load_state_dict(stripped)
            return

        raise RuntimeError("Could not load model weights into SegmentAwareResNet")

    def train(self, path):
        self.train_dataset = GroupedLightcurveFolderDataSet(
            root=path,
            transform=self.transform,
            split_sizes=self.split_sizes,
            max_segments_per_item=self.max_segments_per_item,
        )

        self.classes = self.train_dataset.classes
        self._set_model(len(self.classes))
        self._reset_optimizer_and_scheduler()

        # Split the dataset into training and testing sets (80% train, 20% test)
        train_size = int(0.8 * len(self.train_dataset))
        test_size = len(self.train_dataset) - train_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(
            self.train_dataset, [train_size, test_size]
        )

        train_loader = self._build_dataloader(self.train_dataset, shuffle=True)
        test_loader = self._build_dataloader(self.test_dataset, shuffle=False)

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
                inputs = inputs.to(self.device, non_blocking=self.pin_memory)
                mask = mask.to(self.device, non_blocking=self.pin_memory)
                labels = labels.to(self.device, non_blocking=self.pin_memory)

                # Zero the parameter gradients
                self.optimizer.zero_grad(set_to_none=True)

                # Forward pass
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    outputs = self.model(inputs, mask)
                    loss = self.criterion(outputs, labels)
                training_loss += loss.item()

                # Backward pass and optimization
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()

            for inputs, mask, labels in tqdm(test_loader, desc="Test", leave=False):
                inputs = inputs.to(self.device, non_blocking=self.pin_memory)
                mask = mask.to(self.device, non_blocking=self.pin_memory)
                labels = labels.to(self.device, non_blocking=self.pin_memory)
                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=self.use_amp):
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
                final_weights = self._unwrap_model().state_dict()

        torch.save(final_weights, self.MODEL_SAVE_PATH)
        with open(self.CLASS_SAVE_PATH, "w") as class_f:
            json.dump(self.classes, class_f)

    def classify(self, dataset, label=None):
        with open(self.CLASS_SAVE_PATH, "r") as class_f:
            self.classes = json.load(class_f)
        self._set_model(len(self.classes))
        self._load_model_weights()
        self.model.eval()

        loader = self._build_dataloader(dataset, shuffle=False)

        out_labels = {}

        if label is None:
            label = ""
        else:
            label = f" {label}"

        for images, mask, paths in tqdm(loader, desc=f"Classify{label}"):
            images = images.to(self.device)
            mask = mask.to(self.device)
            with torch.no_grad():
                outputs = self.model(images, mask)

            for i, path in enumerate(paths):
                logits = outputs[i].detach().cpu()
                pred = int(torch.argmax(logits).item())
                out_labels[path] = [self.classes[pred]] + logits.tolist()

        return out_labels

    def classify_folder(self, path, label=None):
        return self.classify_paths(
            [str(p) for p in Path(path).glob("*.fits.gz")], label=label
        )

    def classify_paths(self, paths, label=None):
        return self.classify(
            GroupedLightcurvePathsDataSet(
                paths=paths,
                transform=self.transform,
                split_sizes=self.split_sizes,
                max_segments_per_item=self.max_segments_per_item,
            )
        )
