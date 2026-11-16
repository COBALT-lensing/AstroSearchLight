"""Read-only checkpoint inference on the same preprocessed curves as the filter."""

import json
from pathlib import Path

import numpy as np

from .scoring import cnn_score
from .. import config


class EvaluationCNN:
    def __init__(self, checkpoint=None, classes=None, *, version=None, device="cpu", split_sizes=None,
                 max_segments_per_item=None, seed=42, workers=1):
        import torch
        if checkpoint is not None and version is not None:
            raise ValueError("Pass a version or an archived checkpoint, not both")
        self.MODEL_SAVE_PATH = Path(checkpoint) if checkpoint is not None else config.simulens_directory(version) / "model_weights.pth"
        self.CLASS_SAVE_PATH = Path(classes) if classes else self.MODEL_SAVE_PATH.with_name("model_classes.json")
        self.classes = json.loads(self.CLASS_SAVE_PATH.read_text())
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise ValueError("Checkpoint class mapping must contain unique class names")
        self.device = torch.device(device)
        self.rendering = {"figure.figsize": [6.4, 4.8], "figure.dpi": 100,
                          "savefig.dpi": 100, "figure.facecolor": "white", "savefig.facecolor": "white",
                          "axes.facecolor": "white", "axes.xmargin": .05, "axes.ymargin": .05,
                          "figure.subplot.left": .125, "figure.subplot.right": .9,
                          "figure.subplot.bottom": .11, "figure.subplot.top": .88}
        self.split_sizes, self.max_segments_per_item = config.cnn_inference_settings(
            self.MODEL_SAVE_PATH.parent, split_sizes=split_sizes,
            max_segments_per_item=max_segments_per_item)
        from .parallel import worker_count
        self.workers = worker_count(workers) if self.device.type == 'cpu' else 1
        self.seed = seed
        self.model = None
        if self.workers == 1:
            self._initialize_model()

    def _initialize_model(self):
        import torch
        from torchvision import transforms
        from ..models.simulens.v1 import SegmentAwareResNet, IMAGE_DIMS
        torch.manual_seed(self.seed)
        # Full trained weights replace initialization; no ImageNet download needed.
        self.model = SegmentAwareResNet(num_classes=len(self.classes), weights=None).to(self.device)
        state = torch.load(self.MODEL_SAVE_PATH, map_location=self.device, weights_only=True)
        if all(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        self.model.load_state_dict(state)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_DIMS), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def classify_curves(self, samples, curves, *, positive_class):
        from .parallel import process_map, worker_count
        if worker_count(self.workers, len(samples)) > 1:
            options = dict(checkpoint=str(self.MODEL_SAVE_PATH), classes=str(self.CLASS_SAVE_PATH),
                           device='cpu', split_sizes=self.split_sizes,
                           max_segments_per_item=self.max_segments_per_item, seed=self.seed, workers=1)
            return process_map(_predict_curve, [(s, curves[s.sample_id], positive_class) for s in samples],
                workers=self.workers, desc='CNN inference', progress=getattr(self, 'progress', False),
                initializer=_initialize_cnn_worker, initargs=(options,))
        if self.model is None:
            self._initialize_model()
        import torch
        import matplotlib
        from astropy.time import Time
        from astropy.timeseries import TimeSeries
        from ..lightcurves import plot
        from ..models.simulens.v1 import (
            plot_to_rgb, _segment_windows_for_split_size, _limit_segment_windows,
            _segment_lightcurve, collate_segmented_batch,
        )
        results = []
        from tqdm.auto import tqdm
        for sample in tqdm(samples, desc="CNN inference", disable=not getattr(self, "progress", False)):
            curve = curves[sample.sample_id]
            series = TimeSeries(time=Time(curve.time, format="jd", scale="tdb"))
            series["flux"] = curve.flux
            windows = [window for size in self.split_sizes for window in _segment_windows_for_split_size(series, size)]
            windows = _limit_segment_windows(windows, self.max_segments_per_item)
            images = []
            for start, end, _ in windows:
                segment = _segment_lightcurve(series, start, end)
                if len(segment):
                    with matplotlib.rc_context(self.rendering):
                        images.append(self.transform(plot_to_rgb(plot(segment))))
            if not images:
                raise ValueError(f"No CNN segments for {sample.sample_id}")
            batch, mask, _ = collate_segmented_batch([(images, sample.sample_id)])
            with torch.inference_mode():
                logits = self.model(batch.to(self.device), mask.to(self.device))[0].cpu().numpy()
            results.append(cnn_score(sample.sample_id, self.classes, logits, positive_class=positive_class))
        return results


_WORKER_CNN = None


def _initialize_cnn_worker(options):
    from .parallel import initialize_worker
    initialize_worker()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    global _WORKER_CNN
    _WORKER_CNN = EvaluationCNN(**options)


def _predict_curve(job):
    sample, curve, positive_class = job
    return _WORKER_CNN.classify_curves([sample], {sample.sample_id: curve}, positive_class=positive_class)[0]
