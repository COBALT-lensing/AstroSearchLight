"""default proposal parity and shared deterministic/TESSGP simulation contracts."""

import json
from pathlib import Path
import runpy

import numpy as np
import pytest
from astropy.io import fits
from astropy.timeseries import TimeSeries

from searchlight import generation as g
from searchlight import config


def _fixed_ranges():
    return {"a": [10, 10], "i": [np.pi / 2, np.pi / 2], "Macc": [5, 5],
            "Mcomp": [1, 1], "e": [0, 0], "periapsis_phase": [0, 0], "d": [.1, .1]}


def _rows():
    return g.sample_systems(2, np.random.default_rng(7), ranges=_fixed_ranges())


def test_samples_match_default_distributions_exactly():
    # Independent specification of the seven default draws, in their draw order.
    rng = np.random.default_rng(20260825)
    limits = g.DEFAULT_RANGES
    expected = {
        "a": 10 ** rng.uniform(np.log10(.1), np.log10(200), 100),
        "i": rng.uniform(0, np.pi / 2, 100),
        "Macc": 10 ** rng.uniform(np.log10(.1), np.log10(50), 100),
        "Mcomp": rng.uniform(.5, 50, 100),
        "e": 10 ** rng.uniform(0, np.log10(1.0 + .999999), 100) - 1,
        "periapsis_phase": rng.uniform(0, 2 * np.pi, 100),
        "d": rng.uniform(.1, 100, 100),
    }
    actual = g.sample_parameter_rows(np.random.default_rng(20260825), 100)
    for name in limits:
        np.testing.assert_array_equal(actual[name], expected[name])
    assert g.SAMPLING_DISTRIBUTIONS["Mcomp"] == "uniform"
    assert g.SAMPLING_DISTRIBUTIONS["e"] == "uniform_log1p"


def test_fixed_ranges_and_rejection_budget():
    rows = _rows()
    for name, bounds in _fixed_ranges().items():
        np.testing.assert_allclose(rows[name], bounds[0])
    keep, derived = g.evaluate_parameter_rows(rows)
    assert keep.all()
    assert np.all(derived["tau_eff"] > 0)
    with pytest.raises(RuntimeError, match="viable systems"):
        g.sample_systems(2, np.random.default_rng(2), ranges={**_fixed_ranges(), "i": [0, 0]}, max_attempts=5)
    for ranges in ({"e": [0, 1]}, {"Macc": [-1, 1]}, {"typo": [1, 2]}):
        with pytest.raises(ValueError):
            g.resolve_ranges(ranges)


def test_sector_is_exactly_27_days_at_600_seconds():
    times = g.sector_times()
    assert len(times) == 3888
    assert times[0] == -13.5
    np.testing.assert_allclose(np.diff(times), 600 / 86400)
    assert times[-1] < 13.5


def test_noiseless_flux_matches_public_batched_forward_model():
    from selflensing.batch import batched_light_curve_jit
    rows = _rows()
    times = np.arange(-10, 11) * 600 / 86400
    peaks = np.array([0.0, .01])
    expected = np.asarray(batched_light_curve_jit(times, g.parameter_matrix(rows), peak_times=peaks))
    actual = g.LightCurveGenerator(seed=7).generate(rows, times, peak_times=peaks, measure_lensing=True)
    np.testing.assert_array_equal(actual.flux, expected)
    np.testing.assert_array_equal(actual.mean_flux, expected)
    assert np.all(actual.flux_err == 0)
    assert actual.gp_parameters is None
    assert np.all(actual.lensing_amplitude > 0)


def test_tess_gp_inputs_use_count_space_and_independent_priors(monkeypatch):
    import jax.numpy as jnp
    from selflensing.gp import FluxFinderSensitivity as finder
    from selflensing.gp import PriorsSamplers as priors
    class FluxModel:
        def mass_to_mag(self, mass, distance):
            return jnp.asarray(10.), jnp.asarray(.001)
    monkeypatch.setattr(finder, "FluxModel", FluxModel)
    draws = np.array([[50., 100., 100.], [300., 10000., 1000.]])
    calls = []
    def sample_sigma(mu, component, rng, size):
        calls.append((component, size))
        return np.log10(draws[:, 0 if component == "G" else 2] / 1e6)
    monkeypatch.setattr(priors, "sample_log_sigma", sample_sigma)
    monkeypatch.setattr(priors, "sample_quality_factor", lambda rng, size: draws[:, 1])
    generate = g.make_tess_gp_input_generator(np.random.default_rng(42))
    parameters, counts, noise = generate({"Mcomp": np.ones(2), "d": np.ones(2)})
    assert calls == [("G", 2), ("S", 2)]
    np.testing.assert_allclose(parameters[:, 3], draws[:, 1])
    np.testing.assert_allclose(10 ** parameters[:, 1] / counts * 1e6, draws[:, 0])
    np.testing.assert_allclose(10 ** parameters[:, 4] / counts * 1e6, draws[:, 2])
    np.testing.assert_allclose(noise / counts, .001)


def test_real_tess_gp_is_seeded_and_changes_the_flux():
    rows = _rows()
    times = np.arange(-10, 11) * 600 / 86400
    first = g.LightCurveGenerator(seed=17).generate(rows, times, noisy=True)
    second = g.LightCurveGenerator(seed=17).generate(rows, times, noisy=True)
    np.testing.assert_array_equal(first.flux, second.flux)
    np.testing.assert_array_equal(first.gp_parameters, second.gp_parameters)
    assert np.all(np.isfinite(first.flux))
    assert np.all(first.flux_err > 0)
    assert not np.array_equal(first.flux, first.mean_flux)
    # Identical physical rows get independent GP draws.
    assert not np.array_equal(first.flux[0], first.flux[1])


@pytest.mark.parametrize("noisy", [False, True])
def test_three_class_dataset_is_readable_and_records_each_subtype(tmp_path, noisy):
    output = config.dataset_directory("2.0")
    metadata = g.generate_cnn_dataset("2.0", counts={"lensing": 2, "flare": 2, "not-lensing": 4},
        ranges=_fixed_ranges(), noisy=noisy, detection_sigma=0, duration_days=.2,
        batch_size=2, inverted_fraction=.5, progress=False)
    assert metadata["status"] == "complete"
    assert metadata["dataset_version"] == "2.0"
    assert metadata["written_counts"] == {"lensing": 2, "flare": 2, "not-lensing": 4}
    assert len(metadata["files"]) == 8
    assert metadata["sampling_distributions"]["Mcomp"] == "uniform"
    assert metadata["noisy"] == noisy
    json.loads((output / "generation.json").read_text())
    negatives = [item for item in metadata["files"] if item["label"] == "not-lensing"]
    assert sorted(item["subtype"] for item in negatives) == ["inverted", "inverted", "physical", "physical"]
    for entry in metadata["files"]:
        path = output / entry["path"]
        curve = TimeSeries.read(path, astropy_native=True)
        assert len(curve) == len(g.sector_times(duration_days=.2))
        assert len(entry["sha256"]) == 64
        assert np.all(np.isfinite(curve["flux"]))
        header = fits.getheader(path, 1)
        assert header["LABEL"] == entry["label"]
        assert header["NOISY"] == noisy
        assert header["GENVER"] == g.GENERATOR_VERSION
        assert np.all(curve["flux_err"] > 0) if noisy else np.all(curve["flux_err"] == 0)
    with pytest.raises(FileExistsError):
        g.generate_cnn_dataset("2.0", count_per_class=2, progress=False)


def test_training_cli_generates_before_constructing_model(tmp_path, monkeypatch):
    from searchlight.models import simulens
    events = []
    class FakeTrainer:
        LOADER_WORKERS = 0
        def __init__(self, **kwargs):
            events.append("model")
            self.MODEL_SAVE_PATH = config.simulens_directory(kwargs["version"]) / "model_weights.pth"
        def train(self, dataset, **kwargs):
            events.append("train")
            return {"best_epoch": 1}
    monkeypatch.setattr(simulens, "SimuLens", FakeTrainer)
    monkeypatch.setattr(g, "generate_from_args", lambda version, args: events.append(("generate", version)))
    main = runpy.run_path(str(Path(__file__).parents[1] / "bin/train_cnn.py"))["main"]
    main(["--dataset-version", "3.0", "--version", "2.0", "--generate-data"])
    assert events == [("generate", "3.0"), "model", "train"]


def test_dataset_generation_cli_selects_a_version(monkeypatch):
    seen = {}
    def generate(version, args):
        seen.update(version=version, noiseless=args.noiseless)
        return {"written_counts": {"lensing": 2, "flare": 2, "not-lensing": 2}}
    monkeypatch.setattr(g, "generate_from_args", generate)
    g.main(["--version", "4.0", "--noiseless"])
    assert seen == {"version": "4.0", "noiseless": True}
    with pytest.raises(SystemExit):
        g.main(["--version", "/tmp/dataset"])


def test_new_dataset_version_preserves_the_previous_dataset():
    import hashlib
    options = dict(count_per_class=2, ranges=_fixed_ranges(), noisy=False,
                   duration_days=.2, batch_size=2, progress=False)
    g.generate_cnn_dataset("1.0", **options)
    before = {str(path.relative_to(config.dataset_directory("1.0"))): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in config.dataset_directory("1.0").rglob("*") if path.is_file()}
    g.generate_cnn_dataset("2.0", **options)
    after = {str(path.relative_to(config.dataset_directory("1.0"))): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in config.dataset_directory("1.0").rglob("*") if path.is_file()}
    assert before == after
    assert json.loads((config.dataset_directory("2.0") / "generation.json").read_text())["dataset_version"] == "2.0"
