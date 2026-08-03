import sys
from types import SimpleNamespace

import numpy as np
from astropy import units
from astropy.time import Time
from astropy.timeseries import TimeSeries

from searchlight import lightcurves


def _timeseries():
    ts = TimeSeries(time=Time([0.0, 0.05, 0.2, 0.3], format="jd"))
    ts["flux"] = np.array([1.0, 2.0, 3.0, 4.0]) * units.dimensionless_unscaled
    ts["flux_err"] = np.array([0.1, 0.2, 0.3, 0.4]) * units.dimensionless_unscaled
    ts.meta["TICID"] = 123
    ts.meta["SECTOR"] = 42
    ts.meta["RA"] = 10.5
    ts.meta["DEC"] = -20.25
    return ts


def test_load_delegates_mast_paths_to_load_tess(monkeypatch):
    calls = []
    monkeypatch.setattr(lightcurves, "load_tess", lambda path: calls.append(path) or "ts")

    assert lightcurves.load("mast:example") == "ts"
    assert calls == ["mast:example"]


def test_load_reads_non_mast_paths_with_timeseries(monkeypatch):
    calls = []

    def fake_read(path, astropy_native):
        calls.append((path, astropy_native))
        return "ts"

    monkeypatch.setattr(lightcurves.TimeSeries, "read", fake_read)

    assert lightcurves.load("local.fits") == "ts"
    assert calls == [("local.fits", True)]


def test_load_all_is_lazy_and_honors_limit(monkeypatch):
    calls = []
    monkeypatch.setattr(lightcurves, "load", lambda path: calls.append(path) or path)

    assert list(lightcurves.load_all(["a", "b", "c"], limit=2)) == ["a", "b"]
    assert calls == ["a", "b"]


def test_plot_returns_figure_with_axis_hidden():
    fig = lightcurves.plot(_timeseries())

    try:
        assert len(fig.axes) == 1
        assert not fig.axes[0].axison
    finally:
        lightcurves.pyplot.close(fig)


def test_get_spoc_urls_formats_int_targets_and_filters_mast_uris(monkeypatch):
    calls = []

    class FakeTable:
        colnames = ["dataURL"]

        def __getitem__(self, key):
            assert key == "dataURL"
            return [
                "mast:HLSP/example one.fits",
                "https://example.invalid/not-mast.fits",
            ]

    class FakeResult:
        table = FakeTable()

        def __len__(self):
            return 2

    def fake_search(target_id, mission, author):
        calls.append((target_id, mission, author))
        return FakeResult()

    monkeypatch.setattr(lightcurves, "search_lightcurve", fake_search)

    assert lightcurves.get_spoc_urls(123) == [
        "https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:HLSP/example%20one.fits"
    ]
    assert calls == [("TIC 123", "TESS", "SPOC")]


def test_get_spoc_urls_handles_empty_or_missing_dataurl(monkeypatch):
    class EmptyResult:
        table = {"colnames": ["dataURL"]}

        def __len__(self):
            return 0

    monkeypatch.setattr(lightcurves, "search_lightcurve", lambda *args, **kwargs: EmptyResult())
    assert lightcurves.get_spoc_urls("TIC 1") == []

    class MissingColumnResult:
        table = SimpleNamespace(colnames=[])

        def __len__(self):
            return 1

    monkeypatch.setattr(
        lightcurves,
        "search_lightcurve",
        lambda *args, **kwargs: MissingColumnResult(),
    )
    assert lightcurves.get_spoc_urls("TIC 1") == []


def test_load_tess_retries_restores_stdout_cleans_metadata(monkeypatch):
    attempts = []

    def fake_read(uri):
        attempts.append(uri)
        if len(attempts) == 1:
            raise TimeoutError("temporary")
        print("hidden lightkurve output")
        return _timeseries()

    cleared = []
    monkeypatch.setattr(lightcurves, "read_tess_lightcurve", fake_read)
    monkeypatch.setattr(lightcurves, "clear_download_cache", cleared.append)
    monkeypatch.setattr(lightcurves.time, "sleep", lambda _seconds: None)
    original_stdout = sys.stdout

    cleaned = lightcurves.load_tess.__wrapped__("mast:uri")

    assert sys.stdout is original_stdout
    assert attempts == ["mast:uri", "mast:uri"]
    assert cleared == ["mast:uri"]
    assert len(cleaned) == 2
    assert cleaned.meta["TICID"] == 123
    assert cleaned.meta["RA"] == 10.5
    assert np.allclose(cleaned["flux"], [3.0, 4.0])
