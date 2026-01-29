import json
from pathlib import Path

import numpy
import pandas
from astropy import units
from astropy.timeseries import LombScargle
from scipy.stats import kurtosis
from sklearn.model_selection import train_test_split
from tqdm.contrib.concurrent import process_map
from xgboost import XGBClassifier

from lightcurves import load


def get_period(lc, times):  # , false_alarm_probability=0.1):
    # Autopower fails with one point, false_alarm_level fails with exactly three points
    # In reality the lower limit could be higher but I'm reluctant to set an arbitrary
    # minimum.
    if len(times) <= 3:
        return []
    # maximum_frequency = 0.5
    # minimum_frequency = 1.0 / (times[-1] - times[0])
    ls = LombScargle(times, lc)
    frequency, power = ls.autopower(
        # maximum_frequency=maximum_frequency, minimum_frequency=minimum_frequency
    )

    # fal = ls.false_alarm_level(false_alarm_probability)
    # frequency = frequency[power > fal]
    return 1 / frequency[numpy.argmax(power)]


def skewness(ts, N, std_flux, mean_flux):
    stdev3 = std_flux**3
    N_stdev3 = N * stdev3
    # Negative to be consistent with value when using magnitudes
    return -sum([(x - mean_flux) ** 3 / N_stdev3 for x in ts["flux"]])


def von_neumann(ts, N, std_flux):
    stdev2 = std_flux**2
    N1_stdev2 = (N - 1) * stdev2
    return sum(
        [(ts["flux"][i] - ts["flux"][i - 1]) ** 2 / N1_stdev2 for i in range(1, N)]
    )


def cumulative_sum_index(ts, mean_flux):
    return numpy.cumsum(ts["flux"].data - mean_flux)


def features(p):
    ts = load(p)
    std_flux = numpy.std(ts["flux"])
    mean_flux = numpy.mean(ts["flux"])
    N = len(ts)
    P = get_period(ts["flux"], ts.time.jd)
    ts_fold = ts.fold(P * units.day)

    return pandas.Series(
        {
            "std flux": std_flux,
            "mean flux": mean_flux,
            "period": P,
            "skewness": skewness(ts, N, std_flux, mean_flux),
            "von Neumann": von_neumann(ts_fold, N, std_flux),
            # "cumulative sum index": cumulative_sum_index(ts_fold, mean_flux),
            "kurtosis": kurtosis(numpy.array(ts["flux"])),
        },
        dtype=float,
    )


class XGBFilter(object):
    MODEL_SAVE_PATH = Path("data/xgboost/model.ubj")

    def __init__(
        self,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective="binary:logistic",
    ):
        self.MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.CLASS_SAVE_PATH = self.MODEL_SAVE_PATH.parent / "model_classes.json"

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective=objective,
        )

        if self.MODEL_SAVE_PATH.exists():
            self.model.load_model(self.MODEL_SAVE_PATH)

        if self.CLASS_SAVE_PATH.exists():
            with open(self.CLASS_SAVE_PATH, "r") as class_f:
                self.classes = json.load(class_f)

    def _load_dir(self, path):
        paths = list(Path(path).glob("*.fits.gz"))
        return self._load_paths(paths, desc=str(path))

    def _load_paths(self, paths, desc=""):
        if len(desc) > 20:
            desc = "..." + desc[-20:]
        return pandas.concat(
            process_map(features, paths, desc=f"Loading {desc}", chunksize=1),
            axis=1,
        ).T

    def train(self, path):
        path = Path(path)
        paths = [x for x in path.iterdir() if x.is_dir()]

        xs = []
        ys = []

        for y, c in enumerate(paths):
            xs_c = self._load_dir(c)
            xs.append(xs_c)
            ys += [y] * len(xs_c)

        xs = pandas.concat(xs)
        ys = numpy.array(ys)

        self.train_dataset, self.test_dataset, self.train_labels, self.test_labels = (
            train_test_split(xs, ys, test_size=0.2)
        )
        self.model.fit(self.train_dataset, self.train_labels)

        preds = self.model.predict(self.test_dataset)
        print(f"Model accuracy: {(preds == self.test_labels).mean() * 100:.2f}%")

        self.model.save_model(self.MODEL_SAVE_PATH)
        self.classes = [p.name for p in paths]
        with open(self.CLASS_SAVE_PATH, "w") as class_f:
            json.dump(self.classes, class_f)

    def classify(self, dataset):
        return self.model.predict(dataset)

    def classify_folder(self, path):
        dataset = self._load_dir(path)
        return self.classify(dataset)

    def classify_paths(self, paths):
        dataset = self._load_paths(paths)
        return self.classify(dataset)
