import json

from ...filter import FeatureFilter
from ...paths import DATA_PATH

XGBClassifier = None
train_test_split = None


def _require_xgboost_dependencies():
    global XGBClassifier, train_test_split
    if XGBClassifier is not None and train_test_split is not None:
        return XGBClassifier, train_test_split

    try:
        from sklearn.model_selection import train_test_split as imported_split
        from xgboost import XGBClassifier as imported_classifier
    except ImportError as exc:
        raise ImportError(
            "XGBFilter requires the optional deprecated dependencies. "
            "Install them with `uv sync --extra deprecated --all-groups`."
        ) from exc

    XGBClassifier = imported_classifier
    train_test_split = imported_split
    return XGBClassifier, train_test_split


class XGBFilter(FeatureFilter):
    MODEL_SAVE_PATH = DATA_PATH / "xgboost"
    MODEL_PATH = MODEL_SAVE_PATH / "model.ubj"

    def __init__(
        self,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective="binary:logistic",
    ):
        super().__init__()
        classifier, _ = _require_xgboost_dependencies()

        self.model = classifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective=objective,
        )

        if self.MODEL_PATH.exists():
            self.model.load_model(self.MODEL_PATH)

    def train(self, path):
        _, split = _require_xgboost_dependencies()
        xs, ys, paths = self._load_datasets(path)

        self.train_dataset, self.test_dataset, self.train_labels, self.test_labels = (
            split(xs, ys, test_size=0.2)
        )
        self.model.fit(self.train_dataset, self.train_labels)

        preds = self.model.predict(self.test_dataset)
        print(f"Model accuracy: {(preds == self.test_labels).mean() * 100:.2f}%")

        self.model.save_model(self.MODEL_PATH)
        self._save_classes(paths)

    def classify(self, dataset):
        return self.model.predict(dataset)
