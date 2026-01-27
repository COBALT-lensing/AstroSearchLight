import numpy
import time

from astropy.timeseries import TimeSeries
from astropy import units
from astropy.utils.data import clear_download_cache
from astropy.timeseries import LombScargle
from matplotlib import pyplot
from pathlib import Path
from urllib.error import URLError

import warnings

warnings.filterwarnings("ignore", module="lightkurve.prf")
from lightkurve.io.tess import read_tess_lightcurve
import os
import sys

LC_DATA_PATH = Path("/Users/adam/Code/lc-simulator/data")


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


def features(p):
    ts = load(p)
    std_flux = numpy.std(ts["flux"])
    mean_flux = numpy.mean(ts["flux"])
    N = len(ts)

    return numpy.array(
        (
            std_flux,
            mean_flux,
            get_period(ts["flux"], ts["time"].jd),
            skewness(ts, N, std_flux, mean_flux),
            von_neumann(ts, N, std_flux),
        )
    )


def load_tess(data_uri):
    max_attempts = 5

    for attempt in range(max_attempts):
        try:
            with open(os.devnull, "w") as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    ts = read_tess_lightcurve(data_uri)
                finally:
                    sys.stdout = old_stdout
        except (TimeoutError, URLError, FileNotFoundError, ValueError, TypeError) as e:
            if attempt == max_attempts - 1:
                raise e
            time.sleep(60 * attempt)
            continue
        break

    # Trim the first few hours from the start as loads of SPOC light
    # curves seem to start with spurious peaks
    ts = ts[ts["time"] > ts["time"][0] + 3 * units.hour]
    ts["flux"] = ts["flux"].value

    # There isn't an easy way to avoid filling the filesystem with
    # astropy's cache so just delete each file as we process it
    clear_download_cache(data_uri)
    return ts


def load(lc_path):
    if "mast:HLSP/tess-spoc/" in str(lc_path):
        return load_tess(lc_path)
    return TimeSeries.read(lc_path, astropy_native=True)


def load_all(lc_paths, limit=numpy.inf):
    x = 0
    for lc_path in lc_paths:
        x += 1
        if x > limit:
            break
        yield load(lc_path)


def plot(lc):
    f = pyplot.figure()
    pyplot.scatter(lc["time"].jd, lc["flux"], s=2, alpha=0.5, c="blue")
    pyplot.axis("off")
    return f
