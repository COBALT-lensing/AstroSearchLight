from pathlib import Path

from upsilont import UPSILoNT

from filter import FeatureFilter


class UpsilonFilter(FeatureFilter):
    MODEL_SAVE_PATH = Path("data/upsilont/")

    def __init__(self):
        super().__init__()
        self.ut = UPSILoNT()

    def train(self, path):
        xs, ys = self._load_datasets(path)
        self.ut.transfer(xs, ys, self.MODEL_SAVE_PATH)

    def classify(self, dataset):
        self.ut.load(self.MODEL_SAVE_PATH)
        return self.ut.predict(dataset)
