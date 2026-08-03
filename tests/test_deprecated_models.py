import numpy as np
import pandas as pd

from searchlight.models.deprecated import tree, upsilon


def test_xgb_filter_constructs_loads_trains_and_predicts(monkeypatch, tmp_path):
    calls = []

    class FakeXGBClassifier:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def load_model(self, path):
            calls.append(("load", path))

        def fit(self, xs, ys):
            calls.append(("fit", list(xs["feature"]), ys.tolist()))

        def predict(self, dataset):
            calls.append(("predict", dataset))
            return np.array([0, 1])

        def save_model(self, path):
            calls.append(("save", path))

    monkeypatch.setattr(tree, "XGBClassifier", FakeXGBClassifier)
    monkeypatch.setattr(tree.XGBFilter, "MODEL_SAVE_PATH", tmp_path)
    monkeypatch.setattr(tree.XGBFilter, "MODEL_PATH", tmp_path / "model.ubj")
    monkeypatch.setattr(
        tree.XGBFilter,
        "_load_datasets",
        lambda self, path: (
            pd.DataFrame({"feature": [1, 2]}),
            np.array([0, 1]),
            [tmp_path / "a", tmp_path / "b"],
        ),
    )
    monkeypatch.setattr(
        tree,
        "train_test_split",
        lambda xs, ys, test_size: (xs, xs, ys, ys),
    )
    (tmp_path / "model.ubj").write_text("model")

    filt = tree.XGBFilter(n_estimators=3)
    assert filt.classify("dataset").tolist() == [0, 1]
    filt.train("data")

    assert calls[0][0] == "init"
    assert calls[0][1]["n_estimators"] == 3
    assert ("load", tmp_path / "model.ubj") in calls
    assert ("save", tmp_path / "model.ubj") in calls
    assert filt.classes == ["a", "b"]


def test_upsilon_filter_constructs_trains_and_predicts(monkeypatch, tmp_path):
    calls = []

    class FakeUPSILoNT:
        def transfer(self, xs, ys, path):
            calls.append(("transfer", list(xs["feature"]), ys.tolist(), path))

        def load(self, path):
            calls.append(("load", path))

        def predict(self, dataset):
            calls.append(("predict", dataset))
            return np.array([1])

    monkeypatch.setattr(upsilon, "UPSILoNT", FakeUPSILoNT)
    monkeypatch.setattr(upsilon.UpsilonFilter, "MODEL_SAVE_PATH", tmp_path)
    monkeypatch.setattr(
        upsilon.UpsilonFilter,
        "_load_datasets",
        lambda self, path: (
            pd.DataFrame({"feature": [5]}),
            np.array([1]),
            [tmp_path / "class"],
        ),
    )

    filt = upsilon.UpsilonFilter()
    filt.train("data")
    assert filt.classify("dataset").tolist() == [1]

    assert ("transfer", [5], [1], tmp_path) in calls
    assert ("load", tmp_path) in calls
    assert ("predict", "dataset") in calls
    assert filt.classes == ["class"]
