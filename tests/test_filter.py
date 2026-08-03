import json
import builtins
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from searchlight import filter as feature_filter


class DummyFilter(feature_filter.FeatureFilter):
    def classify(self, dataset):
        return np.array([1, 0])

    def train(self, path):
        return path


def test_feature_filter_requires_model_save_path():
    class MissingPathFilter(feature_filter.FeatureFilter):
        pass

    with pytest.raises(RuntimeError, match="Model save path is not set"):
        MissingPathFilter()


def test_feature_filter_loads_and_saves_classes(tmp_path):
    DummyFilter.MODEL_SAVE_PATH = tmp_path
    (tmp_path / "model_classes.json").write_text(json.dumps(["old"]))

    filt = DummyFilter()
    assert filt.classes == ["old"]

    filt._save_classes([Path("first"), Path("second")])
    assert filt.classes == ["first", "second"]
    assert json.loads((tmp_path / "model_classes.json").read_text()) == [
        "first",
        "second",
    ]


def test_feature_filter_load_dir_only_uses_fits_gz(tmp_path, monkeypatch):
    DummyFilter.MODEL_SAVE_PATH = tmp_path / "model"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "a.fits.gz").write_text("a")
    (data_dir / "b.fits").write_text("b")
    seen = []

    def fake_load_paths(self, paths, desc=""):
        seen.append((paths, desc))
        return pd.DataFrame({"x": [1]})

    monkeypatch.setattr(DummyFilter, "_load_paths", fake_load_paths)

    paths, dataset = DummyFilter()._load_dir(data_dir)

    assert paths == [data_dir / "a.fits.gz"]
    assert seen == [([data_dir / "a.fits.gz"], str(data_dir))]
    assert list(dataset["x"]) == [1]


def test_feature_filter_load_paths_builds_dataframe(monkeypatch, tmp_path):
    DummyFilter.MODEL_SAVE_PATH = tmp_path

    def fake_process_map(func, paths, desc, chunksize, max_workers):
        assert desc == "Loading ...y/very/long/path.txt"
        assert chunksize == 1
        assert max_workers == 16
        return [{"a": 1}, {"a": 2}]

    monkeypatch.setattr(feature_filter, "process_map", fake_process_map)

    frame = DummyFilter()._load_paths(
        ["first", "second"],
        desc="/a/very/very/long/path.txt",
    )

    assert list(frame["a"]) == [1, 2]


def test_features_builds_upsilon_feature_dict(monkeypatch):
    captured = {}

    class FakeLightCurve:
        time = type("Time", (), {"jd": [1, 2]})()

        def __getitem__(self, key):
            assert key == "flux"
            return [3, 4]

    class FakeVariabilityFeatures:
        def __init__(self, time, flux, n_threads):
            captured["time"] = time
            captured["flux"] = flux
            captured["n_threads"] = n_threads

        def get_features(self):
            return {"amplitude": 0.5}

    monkeypatch.setattr(feature_filter, "load", lambda path: FakeLightCurve())
    monkeypatch.setattr(feature_filter, "VariabilityFeatures", FakeVariabilityFeatures)

    assert feature_filter.features.__wrapped__("path.fits") == {"amplitude": 0.5}
    assert captured == {"time": [1, 2], "flux": [3, 4], "n_threads": 1}


def test_features_explains_missing_optional_dependency(monkeypatch):
    class FakeLightCurve:
        time = type("Time", (), {"jd": [1, 2]})()

        def __getitem__(self, key):
            return [3, 4]

    monkeypatch.setattr(feature_filter, "load", lambda path: FakeLightCurve())
    monkeypatch.setattr(feature_filter, "VariabilityFeatures", None)
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("upsilont"):
            raise ImportError("missing upsilont")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="optional deprecated dependencies"):
        feature_filter.features.__wrapped__("path.fits")


def test_feature_filter_load_datasets_and_classify_folder(monkeypatch, tmp_path):
    DummyFilter.MODEL_SAVE_PATH = tmp_path / "model"
    data_dir = tmp_path / "data"
    class_a = data_dir / "a"
    class_b = data_dir / "b"
    class_a.mkdir(parents=True)
    class_b.mkdir()

    def fake_load_dir(self, path):
        return [path / "sample.fits.gz"], pd.DataFrame({"feature": [path.name]})

    monkeypatch.setattr(DummyFilter, "_load_dir", fake_load_dir)

    xs, ys, paths = DummyFilter()._load_datasets(data_dir)

    assert list(xs["feature"]) == ["a", "b"]
    assert ys.tolist() == [0, 1]
    assert paths == [class_a, class_b]


def test_feature_filter_classify_paths_and_folder_delegate(monkeypatch, tmp_path):
    DummyFilter.MODEL_SAVE_PATH = tmp_path / "model"
    filt = DummyFilter()
    filt.classes = ["not-lensing", "lensing"]
    loaded = pd.DataFrame({"feature": [1, 2]})
    monkeypatch.setattr(filt, "_load_paths", lambda paths, desc="": loaded)

    assert filt.classify_paths(["a", "b"]) == {
        "a": ("lensing", None, None),
        "b": ("not-lensing", None, None),
    }

    monkeypatch.setattr(filt, "_load_dir", lambda path: (["c", "d"], loaded))
    assert filt.classify_folder("folder") == {
        "c": ("lensing", None, None),
        "d": ("not-lensing", None, None),
    }
