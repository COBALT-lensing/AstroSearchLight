import numpy
import time

from astropy.timeseries import TimeSeries
from astropy import units
from astropy.utils.data import clear_download_cache
from matplotlib import pyplot
from pathlib import Path
from urllib.error import URLError

import warnings

warnings.filterwarnings("ignore", module="lightkurve.prf")
from lightkurve.io.tess import read_tess_lightcurve
import os
import sys

LC_DATA_PATH = Path("/Users/adam/Code/lc-simulator/data")


def features(ts):
    std_flux = numpy.std(ts["flux"])
    mean_flux = numpy.mean(ts["flux"])
    return {
        "std_flux": std_flux,
        "mean_flux": mean_flux,
    }


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


def load_all(lc_paths):
    for lc_path in lc_paths:
        yield load(lc_path)


def plot(lc):
    f = pyplot.figure()
    pyplot.scatter(lc["time"].jd, lc["flux"], s=2, alpha=0.5, c="blue")
    pyplot.axis("off")
    return f
