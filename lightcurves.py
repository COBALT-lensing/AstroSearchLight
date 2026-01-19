import numpy

from astropy.timeseries import TimeSeries
from matplotlib import pyplot
from pathlib import Path

LC_DATA_PATH = Path("/Users/adam/Code/lc-simulator/data")


def all_invisible():
    return (LC_DATA_PATH / "invisible").glob("*.fits.gz")


def all_visible():
    return (LC_DATA_PATH / "visible").glob("*.fits.gz")


def features(ts):
    std_flux = numpy.std(ts["flux"])
    mean_flux = numpy.mean(ts["flux"])
    return {
        "std_flux": std_flux,
        "mean_flux": mean_flux,
    }


def load(lc_path):
    return TimeSeries.read(lc_path, astropy_native=True)


def load_all(lc_paths):
    for lc_path in lc_paths:
        yield load(lc_path)


def load_visible():
    for lc in load_all(all_visible()):
        yield lc


def load_invisible():
    for lc in load_all(all_invisible()):
        yield lc


def plot(lc):
    f = pyplot.figure()
    pyplot.scatter(lc["time"].jd, lc["flux"], s=2, alpha=0.5, c="blue")
    return f


def plot_as_data(lc):
    return plot(lc).canvas.buffer_rgba()
