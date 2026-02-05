from pathlib import Path

import pandas
import numpy
from tqdm.contrib.concurrent import process_map

from upsilont.features import VariabilityFeatures

from lightcurves import load
import hashlib
import pickle

FEATURE_CACHE_PATH = "data/features/"


def features(p):
    p_hash = hashlib.md5(str(p).encode()).hexdigest()
    cache_file = Path(FEATURE_CACHE_PATH) / p_hash[:2] / p_hash[2:4] / f"{p_hash}.pkl"

    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError, ValueError):
            pass

    ts = load(p)
    features_data = VariabilityFeatures(ts.time.jd, ts["flux"]).get_features()

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(features_data, f)

    return features_data


class FeatureFilter(object):
    MODEL_SAVE_PATH = None

    def __init__(self):
        if self.MODEL_SAVE_PATH is None:
            raise RuntimeError("Model save path is not set")
        self.MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

    def _load_dir(self, path):
        paths = list(Path(path).glob("*.fits.gz"))
        return paths, self._load_paths(paths, desc=str(path))

    def _load_paths(self, paths, desc=""):
        if len(desc) > 20:
            desc = "..." + desc[-20:]
        return pandas.DataFrame(
            process_map(
                features, paths, desc=f"Loading {desc}", chunksize=1, max_workers=8
            ),
        )

    def _load_datasets(self, path):
        path = Path(path)
        paths = [x for x in path.iterdir() if x.is_dir()]

        xs = []
        ys = []

        for y, c in enumerate(paths):
            _, xs_c = self._load_dir(c)
            xs.append(xs_c)
            ys += [y] * len(xs_c)

        xs = pandas.concat(xs)
        ys = numpy.array(ys)
        return xs, ys

    def train(self, path):
        raise NotImplementedError

    def _classification_dict(self, paths, classifications):
        return dict(
            zip(
                paths,
                list(
                    zip(
                        [self.classes[c] for c in classifications],
                        [None] * len(classifications),
                        [None] * len(classifications),
                    )
                ),
            )
        )

    def classify(self, dataset):
        raise NotImplementedError

    def classify_folder(self, path):
        paths, dataset = self._load_dir(path)
        classifications = self.classify(dataset)
        return self._classification_dict(paths, classifications)

    def classify_paths(self, paths):
        dataset = self._load_paths(paths)
        classifications = self.classify(dataset)
        return self._classification_dict(paths, classifications)
