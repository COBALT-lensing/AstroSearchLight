from ...filter import FeatureFilter
from ...paths import DATA_PATH

UPSILoNT = None


def _require_upsilont():
    global UPSILoNT
    if UPSILoNT is not None:
        return UPSILoNT

    try:
        from upsilont import UPSILoNT as imported
    except ImportError as exc:
        raise ImportError(
            "UpsilonFilter requires the optional deprecated dependencies. "
            "Install them with `uv sync --extra deprecated --all-groups`."
        ) from exc

    UPSILoNT = imported
    return UPSILoNT


class UpsilonFilter(FeatureFilter):
    MODEL_SAVE_PATH = DATA_PATH / "upsilont"

    def __init__(self):
        super().__init__()
        self.ut = _require_upsilont()()

    def train(self, path):
        xs, ys, paths = self._load_datasets(path)
        self.ut.transfer(xs, ys, self.MODEL_SAVE_PATH)
        self._save_classes(paths)

    def classify(self, dataset):
        self.ut.load(self.MODEL_SAVE_PATH)
        return self.ut.predict(dataset)
