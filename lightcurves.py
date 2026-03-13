import os
import sys
import time
import warnings
import logging

import numpy
from pathlib import Path
from urllib.error import URLError
import urllib.parse

from astropy import units
from astropy.time import Time
from astropy.timeseries import TimeSeries
from astropy.utils.data import clear_download_cache
from matplotlib import pyplot

# Suppress lightkurve PRF optional dependency warning.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*tpfmodel submodule is not available without oktopus installed.*",
)
warnings.filterwarnings("ignore", category=UserWarning, module=r"lightkurve\.prf.*")

from lightkurve import search_lightcurve
from lightkurve.io.tess import read_tess_lightcurve

from utils import cache

LC_DATA_PATH = Path("/Users/adam/Code/lc-simulator/data")
if not LC_DATA_PATH.exists():
    LC_DATA_PATH = Path("/home/anm1u24/Code/lc-simulator/data")
LC_VALIDATION_PATH = Path("/Users/adam/Code/lc-simulator/data.validation")
if not LC_VALIDATION_PATH.exists():
    LC_VALIDATION_PATH = Path("/home/anm1u24/Code/lc-simulator/data.validation")
logger = logging.getLogger(__name__)


@cache
def load_tess(data_uri):
    max_attempts = 5
    logger.debug(
        "Loading TESS lightcurve from %s (max_attempts=%d)", data_uri, max_attempts
    )

    for attempt in range(max_attempts):
        try:
            logger.debug("Attempt %d/%d for %s", attempt + 1, max_attempts, data_uri)
            with open(os.devnull, "w") as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    ts = read_tess_lightcurve(data_uri)
                finally:
                    sys.stdout = old_stdout
        except (TimeoutError, URLError, FileNotFoundError, ValueError, TypeError) as e:
            if attempt == max_attempts - 1:
                logger.debug("Final attempt failed for %s: %s", data_uri, e)
                raise e
            print(f"Error {e} for {data_uri}")
            logger.debug(
                "Retrying %s after attempt %d/%d failure: %s",
                data_uri,
                attempt + 1,
                max_attempts,
                e,
            )
            time.sleep(60 * attempt)
            continue
        break

    # Trim the first few hours from the start as loads of SPOC light
    # curves seem to start with spurious peaks.
    ts = ts[ts["time"] > ts["time"][0] + 3 * units.hour]

    # Rebuild a minimal TimeSeries with plain ndarray-backed columns. The
    # original lightkurve object can retain astropy metadata that is not
    # picklable.
    cleaned_time = Time(numpy.asarray(ts["time"].jd), format="jd")
    cleaned_ts = TimeSeries(time=cleaned_time)
    cleaned_ts["flux"] = numpy.asarray(ts["flux"].value)
    cleaned_ts["flux_err"] = numpy.asarray(ts["flux_err"].value)

    # There isn't an easy way to avoid filling the filesystem with
    # astropy's cache so just delete each file as we process it
    clear_download_cache(data_uri)
    logger.debug(
        "Loaded and processed TESS lightcurve for %s (rows=%d)",
        data_uri,
        len(cleaned_ts),
    )
    return cleaned_ts


def load(lc_path):
    logger.debug("Loading lightcurve from path/URI: %s", lc_path)
    if "mast:" in str(lc_path):
        return load_tess(lc_path)
    return TimeSeries.read(lc_path, astropy_native=True)


def load_all(lc_paths, limit=numpy.inf):
    logger.debug("Loading all lightcurves with limit=%s", limit)
    x = 0
    for lc_path in lc_paths:
        x += 1
        if x > limit:
            logger.debug("Reached load_all limit (%s); stopping", limit)
            break
        logger.debug("Yielding lightcurve %d from %s", x, lc_path)
        yield load(lc_path)


def plot(lc):
    logger.debug("Plotting lightcurve with %d points", len(lc))
    f = pyplot.figure()
    pyplot.scatter(lc["time"].jd, lc["flux"], s=2, alpha=0.5, c="blue")
    pyplot.axis("off")
    return f


MAST_DOWNLOAD_BASE = "https://mast.stsci.edu/api/v0.1/Download/file/?uri="


def get_spoc_urls(target_id: int | str) -> list[str]:
    if isinstance(target_id, int):
        target_id = f"TIC {target_id}"

    logger.debug("Searching MAST for TESS SPOC URLs for %s", target_id)

    result = search_lightcurve(target_id, mission="TESS", author="SPOC")
    if len(result) == 0:
        return []

    table = result.table
    if "dataURL" not in table.colnames:
        return []

    urls = [
        f"{MAST_DOWNLOAD_BASE}{urllib.parse.quote((str(uri)), safe=':/')}"
        for uri in table["dataURL"]
        if str(uri).startswith("mast:")
    ]

    logger.debug("Found %d SPOC URLs for %s", len(urls), target_id)
    return urls
