import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from filter import FeatureFilter


class XGBFilter(FeatureFilter):
    MODEL_SAVE_PATH = Path("data/xgboost/")
    MODEL_PATH = MODEL_SAVE_PATH / "model.ubj"

    def __init__(
        self,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective="binary:logistic",
    ):
        super().__init__()
        self.CLASS_SAVE_PATH = self.MODEL_SAVE_PATH / "model_classes.json"

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective=objective,
        )

        if self.MODEL_PATH.exists():
            self.model.load_model(self.MODEL_PATH)

        if self.CLASS_SAVE_PATH.exists():
            with open(self.CLASS_SAVE_PATH, "r") as class_f:
                self.classes = json.load(class_f)

    def train(self, path):
        xs, ys = self._load_datasets(path)

        self.train_dataset, self.test_dataset, self.train_labels, self.test_labels = (
            train_test_split(xs, ys, test_size=0.2)
        )
        self.model.fit(self.train_dataset, self.train_labels)

        preds = self.model.predict(self.test_dataset)
        print(f"Model accuracy: {(preds == self.test_labels).mean() * 100:.2f}%")

        self.model.save_model(self.MODEL_PATH)
        self.classes = [p.name for p in paths]
        with open(self.CLASS_SAVE_PATH, "w") as class_f:
            json.dump(self.classes, class_f)

    def classify(self, dataset):
        return self.model.predict(dataset)
