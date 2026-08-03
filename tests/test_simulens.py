import numpy as np
import pytest
import torch
from astropy import units
from astropy.time import Time
from astropy.timeseries import TimeSeries
from PIL import Image

from searchlight.models import simulens
from searchlight.models.simulens import v1


def _lightcurve():
    ts = TimeSeries(time=Time([1.0, 1.5, 2.0, 2.5, 3.0], format="jd"))
    ts["flux"] = np.arange(5.0) * units.dimensionless_unscaled
    return ts


def test_package_exports_simulens_class():
    assert simulens.SimuLens is v1.SimuLens


def test_plot_to_rgb_closes_figure_and_resizes():
    fig = v1.pyplot.figure()
    v1.pyplot.plot([0, 1], [0, 1])

    image = v1.plot_to_rgb(fig)

    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size == v1.IMAGE_DIMS
    assert not v1.pyplot.fignum_exists(fig.number)


def test_load_fits_loads_and_plots_lightcurve(monkeypatch):
    calls = []

    monkeypatch.setattr(v1, "load", lambda path: calls.append(("load", path)) or "lc")
    monkeypatch.setattr(v1, "plot", lambda lc: calls.append(("plot", lc)) or "fig")
    monkeypatch.setattr(v1, "plot_to_rgb", lambda fig: calls.append(("rgb", fig)) or "image")

    assert v1.load_fits.__wrapped__("path.fits") == "image"
    assert calls == [("load", "path.fits"), ("plot", "lc"), ("rgb", "fig")]


def test_split_size_helpers_validate_and_resolve():
    assert v1._relative_time_to_days(2 * units.day) == 2.0
    assert v1._normalize_split_sizes([1, np.inf]) == [1.0, np.inf]
    assert v1._resolve_split_sizes(None) == [np.inf]

    with pytest.raises(ValueError, match="must not be empty"):
        v1._normalize_split_sizes([])
    with pytest.raises(ValueError, match="must be > 0"):
        v1._normalize_split_sizes([0])


def test_segment_lightcurve_and_windows():
    lc = _lightcurve()

    segment = v1._segment_lightcurve(lc, 0.5, 1.5)
    assert list(segment["flux"]) == [1.0, 2.0]
    assert list(v1._segment_lightcurve(lc, 1.0, np.inf)["flux"]) == [2.0, 3.0, 4.0]
    assert v1._segment_windows_for_split_size(lc, np.inf) == [(0.0, np.inf, np.inf)]
    assert v1._segment_windows_for_split_size(lc, 0.75) == [
        (0.0, 0.75, 0.75),
        (0.75, 1.5, 0.75),
        (1.5, 2.25, 0.75),
    ]


def test_segment_windows_for_path_loads_lightcurve(monkeypatch):
    monkeypatch.setattr(v1, "load", lambda path: _lightcurve())

    assert v1._segment_windows_for_path("path.fits", [1.0]) == [
        (0.0, 1.0, 1.0),
        (1.0, 2.0, 1.0),
    ]


def test_limit_segment_windows_validates_and_samples_evenly():
    windows = [(float(i), float(i + 1), 1.0) for i in range(6)]

    assert v1._limit_segment_windows(windows, None) == windows
    assert v1._limit_segment_windows(windows, 3) == [windows[0], windows[2], windows[5]]
    with pytest.raises(ValueError, match="must be > 0"):
        v1._limit_segment_windows(windows, 0)


def test_load_fits_segment_returns_none_for_empty_segment(monkeypatch):
    monkeypatch.setattr(v1, "load", lambda path: _lightcurve())
    assert v1.load_fits_segment.__wrapped__("path.fits", 10.0, 11.0) is None


def test_load_fits_for_segment_uses_full_or_segment_loader(monkeypatch):
    calls = []
    monkeypatch.setattr(v1, "load_fits", lambda path: calls.append(("full", path)) or "full")
    monkeypatch.setattr(
        v1,
        "load_fits_segment",
        lambda path, start, end: calls.append(("segment", path, start, end)) or "segment",
    )

    assert v1._load_fits_for_segment("a", 0.0, np.inf) == "full"
    assert v1._load_fits_for_segment("a", 0.5, 1.0) == "segment"
    assert calls == [("full", "a"), ("segment", "a", 0.5, 1.0)]


def test_file_discovery_helpers(tmp_path):
    root = tmp_path / "root"
    class_b = root / "b"
    class_a = root / "a"
    nested = class_a / "nested"
    nested.mkdir(parents=True)
    class_b.mkdir(parents=True)
    (class_a / "one.fits").write_text("")
    (nested / "two.fits.gz").write_text("")
    (class_b / "ignore.txt").write_text("")

    assert v1._is_lightcurve_file("a.fits")
    assert v1._is_lightcurve_file("a.fits.gz")
    assert not v1._is_lightcurve_file("a.txt")
    assert v1._find_folder_samples(root) == (
        ["a", "b"],
        {"a": 0, "b": 1},
        [(str(nested / "two.fits.gz"), 0), (str(class_a / "one.fits"), 0)],
    )


def test_grouped_folder_dataset_returns_transformed_segments(monkeypatch, tmp_path):
    root = tmp_path / "root"
    (root / "class-a").mkdir(parents=True)
    path = root / "class-a" / "one.fits.gz"
    path.write_text("")
    monkeypatch.setattr(v1, "_segment_windows_for_path", lambda path, sizes: [(0, np.inf, np.inf), (1, 2, 1)])
    monkeypatch.setattr(v1, "_load_fits_for_segment", lambda path, start, end: f"{start}-{end}")

    dataset = v1.GroupedLightcurveFolderDataSet(
        root,
        transform=lambda image: image.upper(),
        target_transform=lambda target: target + 10,
        split_sizes=[1],
    )

    assert len(dataset) == 1
    assert dataset[0] == (["0-INF", "1-2"], 10)


def test_lightcurve_folder_dataset_expands_samples_into_segments(monkeypatch, tmp_path):
    root = tmp_path / "root"
    class_dir = root / "class-a"
    class_dir.mkdir(parents=True)
    path = class_dir / "one.fits"
    path.write_text("")
    monkeypatch.setattr(v1, "_segment_windows_for_path", lambda path, sizes: [(0, 1, 1), (1, 2, 1)])
    monkeypatch.setattr(v1, "_load_fits_for_segment", lambda path, start, end: f"{start}-{end}")

    dataset = v1.LightcurveFolderDataSet(
        root,
        transform=str.upper,
        target_transform=lambda target: target + 10,
        split_sizes=[1],
    )

    assert len(dataset) == 2
    assert dataset[0] == ("0-1", 10)
    assert dataset[1] == ("1-2", 10)


def test_grouped_paths_dataset_returns_path_and_limits_segments(monkeypatch):
    monkeypatch.setattr(
        v1,
        "_segment_windows_for_path",
        lambda path, sizes: [(0, 1, 1), (1, 2, 1), (2, 3, 1)],
    )
    monkeypatch.setattr(v1, "_load_fits_for_segment", lambda path, start, end: start)

    dataset = v1.GroupedLightcurvePathsDataSet(
        ["a"],
        transform=lambda image: image + 100,
        split_sizes=[1],
        max_segments_per_item=2,
    )

    assert len(dataset) == 1
    assert dataset[0] == ([100, 102], "a")


def test_collate_segmented_batch_pads_images_and_handles_path_targets():
    one = torch.ones(1, 2, 2)
    two = torch.full((1, 2, 2), 2.0)

    images, mask, targets = v1.collate_segmented_batch([([one, two], 3), ([one], 4)])
    assert images.shape == (2, 2, 1, 2, 2)
    assert mask.tolist() == [[True, True], [True, False]]
    assert targets.tolist() == [3, 4]
    assert torch.equal(images[1, 1], torch.zeros(1, 2, 2))

    _, _, path_targets = v1.collate_segmented_batch([([one], "a"), ([two], "b")])
    assert path_targets == ["a", "b"]


def test_lightcurve_paths_dataset_ids_segments(monkeypatch):
    monkeypatch.setattr(
        v1,
        "_segment_windows_for_path",
        lambda path, sizes: [(0.0, 1.0, 1.0), (1.0, 2.0, 1.0)],
    )
    monkeypatch.setattr(v1, "_load_fits_for_segment", lambda path, start, end: f"{path}:{start}:{end}")

    dataset = v1.LightcurvePathsDataSet(["a"], transform=str.upper, split_sizes=[1])

    assert len(dataset) == 2
    assert dataset[0] == ("A:0.0:1.0", ("a", 1.0, 0.0, 1.0))


def test_lightcurve_paths_dataset_uses_path_id_for_unsplit_lightcurve(monkeypatch):
    monkeypatch.setattr(
        v1,
        "_segment_windows_for_path",
        lambda path, sizes: [(0.0, np.inf, np.inf)],
    )
    monkeypatch.setattr(v1, "_load_fits_for_segment", lambda path, start, end: "image")

    dataset = v1.LightcurvePathsDataSet(["a"], split_sizes=None)

    assert dataset[0] == ("image", "a")


def test_segment_aware_resnet_forward_masks_padded_segments(monkeypatch):
    class FakeBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2, 2)

        def forward(self, x):
            return x[:, :, 0, 0]

    monkeypatch.setattr(v1.models, "resnet34", lambda weights: FakeBackbone())
    model = v1.SegmentAwareResNet(num_classes=2)
    model.dropout = torch.nn.Identity()
    with torch.no_grad():
        model.segment_attention.weight.fill_(1.0)
        model.segment_attention.bias.zero_()
        model.classifier.weight.copy_(torch.eye(2))
        model.classifier.bias.zero_()

    x = torch.zeros(1, 2, 2, 1, 1)
    x[0, 0, :, 0, 0] = torch.tensor([1.0, 0.0])
    x[0, 1, :, 0, 0] = torch.tensor([100.0, 100.0])
    mask = torch.tensor([[True, False]])

    assert torch.allclose(model(x, mask), torch.tensor([[1.0, 0.0]]))


def test_simulens_initializes_cpu_loader_and_builds_path_dataset(monkeypatch, tmp_path):
    class FakeSegmentAwareResNet(torch.nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.num_classes = num_classes
            self.linear = torch.nn.Linear(1, num_classes)

        def forward(self, x, mask):
            return self.linear(torch.ones(x.shape[0], 1))

    monkeypatch.setattr(v1.SimuLens, "MODEL_SAVE_PATH", tmp_path / "model.pth")
    monkeypatch.setattr(v1, "SegmentAwareResNet", FakeSegmentAwareResNet)
    monkeypatch.setattr(v1.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(v1.torch.mps, "is_available", lambda: False)

    model = v1.SimuLens(loader_batch_size=2, loader_workers=0, use_amp=True)

    assert model.device.type == "cpu"
    assert model.pin_memory is False
    assert model.use_amp is False

    dataset = torch.utils.data.TensorDataset(torch.ones(3, 1))
    loader = model._build_dataloader(dataset, shuffle=False)
    assert loader.batch_size == 2
    assert loader.num_workers == 0

    captured = {}
    def fake_classify(dataset, label=None):
        captured["dataset"] = dataset
        captured["label"] = label
        return {"ok": True}

    monkeypatch.setattr(model, "classify", fake_classify)
    assert model.classify_paths(["a"], label="demo") == {"ok": True}
    assert isinstance(captured["dataset"], v1.GroupedLightcurvePathsDataSet)
    assert captured["label"] is None
