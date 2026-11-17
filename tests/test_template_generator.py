import importlib.util
from pathlib import Path

import numpy as np
import pytest
from astropy import units

from searchlight.matched_filtering import load_template_library
from searchlight import config


def _load_generator_module():
    path = Path(__file__).parents[1] / "bin" / "generate_matched_filter_template_library.py"
    spec = importlib.util.spec_from_file_location("generate_templates", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_shared_generator(monkeypatch, module, factory):
    from types import SimpleNamespace

    class FakeGenerator:
        def generate(self, rows, times, *, peak_times, noisy, sl, ev, db):
            assert noisy is False
            generate = factory(times, peak_times=peak_times, sl=sl, ev=ev, db=db, device=None)
            params = [{name: values[index] for name, values in rows.items()}
                      for index in range(len(rows["a"]))]
            return SimpleNamespace(flux=generate(params))
    monkeypatch.setattr(module.generation, "LightCurveGenerator", FakeGenerator)


def test_batched_generation_uses_generation_batch_size(monkeypatch):
    module = _load_generator_module()
    calls = []
    tasks = [
        (
            {"a": float(index), "i": 0.0, "Macc": 1.0, "Mcomp": 1.0, "e": 0.0, "periapsis_phase": 0.0, "d": 1.0},
            {"Porb": 1.0, "b": 0.0, "tau_eff": 1.0, "Rcomp": 1.0, "Teff": 5800.0},
            np.asarray([-1.0, 0.0]),
            True,
            True,
            False,
            0.0,
        )
        for index in range(5)
    ]

    def fake_make_batched_light_curve_generator(times, **kwargs):
        def generate(params):
            calls.append(len(params))
            return np.vstack([np.ones_like(times) * (row["a"] + 1.0) for row in params])

        return generate

    _patch_shared_generator(monkeypatch, module, fake_make_batched_light_curve_generator)

    flux_rows, parameter_rows, skipped = module._build_lightcurve_rows_batched(
        tasks,
        relative_time_days=np.asarray([-1.0, 0.0]),
        show_progress=False,
        desc="Generating test curves",
        batch_size=2,
    )

    assert calls == [2, 2, 1]
    assert skipped == 0
    assert len(flux_rows) == 5
    assert len(parameter_rows) == 5


def test_batched_filter_matches_selflensing_system_properties():
    from searchlight import generation
    from selflensing.models import SelfLensingSystem
    grid = [
        {
            "a": 10.0,
            "i": 0.5 * np.pi,
            "Macc": 5.0,
            "Mcomp": 1.0,
            "e": 0.0,
            "periapsis_phase": 0.0,
            "d": 1.0,
        },
        {
            "a": 30.0,
            "i": 0.5 * np.pi - 5e-5,
            "Macc": 10.0,
            "Mcomp": 2.0,
            "e": 0.35,
            "periapsis_phase": 0.5 * np.pi,
            "d": 1.0,
        },
        {
            "a": 0.1,
            "i": 1.0,
            "Macc": 1.0,
            "Mcomp": 20.0,
            "e": 0.8,
            "periapsis_phase": 0.0,
            "d": 1.0,
        },
    ]

    expected = []
    for params in grid:
        system = SelfLensingSystem(**{name: value * SelfLensingSystem.param_units[name]
                                      for name, value in params.items()})
        if not bool(system.sl_possible):
            continue
        if bool(system.rlof):
            continue
        derived = {name: float(np.asarray(getattr(system, attr).to_value(unit)))
                   for name, attr, unit in (("Porb", "porb", units.day),
                       ("b", "b", units.dimensionless_unscaled), ("tau_eff", "tau_eff", units.day),
                       ("Rcomp", "Rcomp", units.R_sun), ("Teff", "Teff", units.K))}
        expected.append((params, derived))

    rows = {name: np.asarray([params[name] for params in grid]) for name in generation.PARAMETER_NAMES}
    keep, derived = generation.evaluate_parameter_rows(rows)
    result = [(derived["tau_eff"][index], grid[index],
               {name: derived[name][index] for name in generation.DERIVED_NAMES})
              for index in np.flatnonzero(keep)]

    assert [params for _, params, _ in result] == [params for params, _ in expected]
    for (_, _, derived), (_, expected_derived) in zip(result, expected):
        for name in ("Porb", "b", "tau_eff", "Rcomp", "Teff"):
            assert np.isclose(derived[name], expected_derived[name], rtol=1e-10)


def test_generator_main_writes_default_library(monkeypatch, tmp_path, capsys):
    module = _load_generator_module()
    output = config.template_library_path("2.0")
    ranges = tmp_path / "ranges.json"
    ranges.write_text('{"a": [5, 20], "e": [0, 0.5]}')
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "version": "2.0",
                "overwrite": False,
                "grid_preset": "default",
                "ranges_json": ranges,
                "calibrate_completeness": False,
                "target_recall": 0.9,
                "validation_size": 1000,
                "validation_seed": 42,
                "generation_batch_size": 128,
                "peak_tolerance_samples": 2,
                "cadence_minutes": 60,
                "half_window_days": 1,
                "window_scale": 1.2,
                "include_distance": False,
                "max_templates": None,
                "max_eccentricity": 0.99,
                "allow_rlof": False,
                "min_flux": 0.0,
                "workers": 2,
                "cpu_workers": None,
                "no_sl": False,
                "no_ev": True,
                "no_db": True,
                "no_progress": True,
            },
        )(),
    )
    monkeypatch.setattr(
        module,
        "build_template_library",
        lambda **kwargs: (
            np.asarray([[1.0, 2.0]]),
            np.asarray([0.0, 1.0]),
            np.asarray([(1.0,)], dtype=[("a", "f8")]),
            {
                **kwargs,
                "skipped_invalid_flux": 0,
                "grid_size_before_filters": 2,
                "grid_size_after_filters": 1,
                "filter_interface": "jax.vmap",
                "generation_batch_size": kwargs["generation_batch_size"],
            },
        ),
    )

    module.main()

    library = load_template_library("2.0")
    assert np.array_equal(library.flux, [[1.0, 2.0]])
    assert library.metadata["include_doppler_boosting"] is False
    assert library.metadata["max_eccentricity"] == 0.99
    assert library.metadata["allow_rlof"] is False
    assert library.metadata["min_flux"] == 0.0
    assert library.metadata["filter_interface"] == "jax.vmap"
    assert library.metadata["generation_batch_size"] == 128
    assert library.metadata["ranges"]["a"] == [5.0, 20.0]
    assert library.metadata["ranges"]["e"] == [0.0, 0.5]
    assert '"templates": 1' in capsys.readouterr().out


def test_selflensing_generation_summary_names_batched_jax(monkeypatch):
    module = _load_generator_module()
    monkeypatch.setattr(module, "_jax_devices_for_kind", lambda kind: [])

    summary = module._selflensing_generation_device_summary()

    assert "batched JAX light curves" in summary
    assert "unknown device" in summary


def test_evaluate_template_completeness_scores_centered_matches():
    module = _load_generator_module()
    template_time = np.arange(-5.0, 6.0)
    template_flux = np.exp(-0.5 * (template_time / 2.0) ** 2)
    validation_time = np.arange(-20.0, 21.0)
    validation_flux = np.ones((3, len(validation_time)))
    start = 15
    validation_flux[:, start : start + len(template_flux)] += template_flux

    result = module.evaluate_template_completeness(
        validation_flux=validation_flux,
        validation_time=validation_time,
        template_flux=np.asarray([template_flux]),
        template_time=template_time,
        prominence_sigma=0.1,
        min_score_sigma=None,
    )

    assert result["matched"] == 3
    assert result["total"] == 3
    assert result["recall"] == 1.0


def test_evaluate_template_completeness_can_use_jax_backend():
    module = _load_generator_module()
    pytest.importorskip("jax")
    template_time = np.arange(-5.0, 6.0)
    template_flux = np.exp(-0.5 * (template_time / 2.0) ** 2)
    validation_time = np.arange(-20.0, 21.0)
    validation_flux = np.ones((2, len(validation_time)))
    start = 15
    validation_flux[:, start : start + len(template_flux)] += template_flux

    result = module.evaluate_template_completeness(
        validation_flux=validation_flux,
        validation_time=validation_time,
        template_flux=np.asarray([template_flux]),
        template_time=template_time,
        prominence_sigma=0.1,
        min_score_sigma=None,
        backend="jax",
        batch_size=1,
    )

    assert result["matched"] == 2
    assert result["recall"] == 1.0
    assert result["validation_backend"] == "jax"


def test_auto_validation_backend_uses_cpu_without_gpu(monkeypatch):
    module = _load_generator_module()
    monkeypatch.setattr(module, "_jax_gpu_available", lambda: False)

    assert module._resolve_validation_backend("auto") == "cpu"


def test_evaluate_template_completeness_updates_running_recall(capsys):
    module = _load_generator_module()
    template_time = np.arange(-2.0, 3.0)
    template_flux = np.exp(-0.5 * template_time**2)
    validation_time = np.arange(-10.0, 11.0)
    validation_flux = np.ones((2, len(validation_time)))
    validation_flux[:, 8 : 8 + len(template_flux)] += template_flux

    module.evaluate_template_completeness(
        validation_flux=validation_flux,
        validation_time=validation_time,
        template_flux=np.asarray([template_flux]),
        template_time=template_time,
        prominence_sigma=0.1,
        min_score_sigma=None,
        print_recall=True,
    )

    output = capsys.readouterr().out
    assert "Recall after" not in output
    assert "Recall: 100.00% (2/2 matched)" in output


def test_calibrate_template_library_records_selected_threshold(monkeypatch):
    module = _load_generator_module()
    template_time = np.arange(-5.0, 6.0)
    template_flux = np.exp(-0.5 * (template_time / 2.0) ** 2)
    validation_time = np.arange(-20.0, 21.0)
    validation_flux = np.ones((2, len(validation_time)))
    validation_flux[:, 15 : 15 + len(template_flux)] += template_flux

    monkeypatch.setattr(
        module,
        "build_validation_lightcurves",
        lambda **kwargs: (
            validation_flux,
            validation_time,
            np.asarray([], dtype=module.PARAMETER_FIELDS),
            {"sample_count": 2, "seed": kwargs["seed"]},
        ),
    )
    monkeypatch.setattr(
        module,
        "build_template_library",
        lambda **kwargs: (
            np.asarray([template_flux]),
            template_time,
            np.asarray([], dtype=module.PARAMETER_FIELDS),
            {
                "grid_preset": kwargs["grid_preset"],
                "grid_size_before_filters": 1,
                "grid_size_after_filters": 1,
                "workers": 1,
                "skipped_invalid_flux": 0,
            },
        ),
    )

    _, _, _, metadata = module.calibrate_template_library(
        target_recall=1.0,
        validation_size=2,
        threshold_grid=[(100.0, 100.0), (0.1, None)],
        workers=1,
    )

    selected = metadata["calibration"]["selected"]
    assert selected["recall"] == 1.0
    assert selected["prominence_sigma"] == 0.1
    assert selected["min_score_sigma"] is None


def test_calibrate_template_library_uses_cpu_workers_for_evaluation(monkeypatch):
    module = _load_generator_module()
    template_time = np.arange(-5.0, 6.0)
    template_flux = np.exp(-0.5 * (template_time / 2.0) ** 2)
    validation_time = np.arange(-20.0, 21.0)
    validation_flux = np.ones((1, len(validation_time)))
    validation_flux[:, 15 : 15 + len(template_flux)] += template_flux
    seen = {}

    def fake_build_validation_lightcurves(**kwargs):
        seen["validation_kwargs"] = kwargs
        return (
            validation_flux,
            validation_time,
            np.asarray([], dtype=module.PARAMETER_FIELDS),
            {"sample_count": 1, "seed": kwargs["seed"]},
        )

    def fake_build_template_library(**kwargs):
        seen["template_kwargs"] = kwargs
        return (
            np.asarray([template_flux]),
            template_time,
            np.asarray([], dtype=module.PARAMETER_FIELDS),
            {
                "grid_preset": kwargs["grid_preset"],
                "grid_size_before_filters": 1,
                "grid_size_after_filters": 1,
                "filter_interface": "jax.vmap",
                "skipped_invalid_flux": 0,
            },
        )

    def fake_evaluate_template_completeness(**kwargs):
        seen["evaluation_workers"] = kwargs["workers"]
        return {
            "matched": 1,
            "total": 1,
            "recall": 1.0,
            "prominence_sigma": kwargs["prominence_sigma"],
            "min_score_sigma": kwargs["min_score_sigma"],
            "peak_tolerance_samples": kwargs["peak_tolerance_samples"],
            "median_matched_score": 1.0,
            "validation_backend": "cpu",
        }

    monkeypatch.setattr(
        module,
        "build_validation_lightcurves",
        fake_build_validation_lightcurves,
    )
    monkeypatch.setattr(
        module,
        "build_template_library",
        fake_build_template_library,
    )
    monkeypatch.setattr(
        module,
        "evaluate_template_completeness",
        fake_evaluate_template_completeness,
    )

    module.calibrate_template_library(
        target_recall=1.0,
        validation_size=1,
        threshold_grid=[(0.1, None)],
        workers=8,
        cpu_workers=6,
    )

    assert "generation_workers" not in seen["validation_kwargs"]
    assert "workers" not in seen["template_kwargs"]
    assert "filter_workers" not in seen["template_kwargs"]
    assert seen["evaluation_workers"] == 6


def test_calibrate_template_library_generates_seed_when_omitted(monkeypatch):
    module = _load_generator_module()
    template_time = np.arange(-5.0, 6.0)
    template_flux = np.exp(-0.5 * (template_time / 2.0) ** 2)
    validation_time = np.arange(-20.0, 21.0)
    validation_flux = np.ones((1, len(validation_time)))
    validation_flux[:, 15 : 15 + len(template_flux)] += template_flux
    seen = {}

    monkeypatch.setattr(module, "_generate_validation_seed", lambda: 123456)

    def fake_build_validation_lightcurves(**kwargs):
        seen["seed"] = kwargs["seed"]
        return (
            validation_flux,
            validation_time,
            np.asarray([], dtype=module.PARAMETER_FIELDS),
            {"sample_count": 1, "seed": kwargs["seed"]},
        )

    monkeypatch.setattr(
        module,
        "build_validation_lightcurves",
        fake_build_validation_lightcurves,
    )
    monkeypatch.setattr(
        module,
        "build_template_library",
        lambda **kwargs: (
            np.asarray([template_flux]),
            template_time,
            np.asarray([], dtype=module.PARAMETER_FIELDS),
            {
                "grid_preset": kwargs["grid_preset"],
                "grid_size_before_filters": 1,
                "grid_size_after_filters": 1,
                "workers": 1,
                "skipped_invalid_flux": 0,
            },
        ),
    )

    _, _, _, metadata = module.calibrate_template_library(
        target_recall=1.0,
        validation_size=1,
        threshold_grid=[(0.1, None)],
        workers=1,
    )

    assert seen["seed"] == 123456
    assert metadata["calibration"]["validation"]["seed"] == 123456


def test_validation_cache_honors_seed_in_cache_key(monkeypatch, tmp_path):
    module = _load_generator_module()
    calls = []

    def fake_build_uncached(**kwargs):
        calls.append(kwargs["seed"])
        return (
            np.asarray([[1.0, 1.1, 1.0]]),
            np.asarray([-1.0, 0.0, 1.0]),
            np.asarray([(1.0,)], dtype=[("a", "f8")]),
            {
                "sample_count": kwargs["sample_count"],
                "seed": kwargs["seed"],
                "cadence_minutes": kwargs["cadence_minutes"],
                "half_window_days": kwargs["half_window_days"],
                "window_scale": kwargs["window_scale"],
                "include_distance": kwargs["include_distance"],
                "max_eccentricity": kwargs["max_eccentricity"],
                "allow_rlof": kwargs["allow_rlof"],
                "min_flux": kwargs["min_flux"],
                "skipped_invalid_flux": 0,
            },
        )

    monkeypatch.setattr(
        module,
        "_build_validation_lightcurves_uncached",
        fake_build_uncached,
    )

    first = module.build_validation_lightcurves(
        sample_count=1,
        seed=111,
        cache_dir=tmp_path,
    )
    second = module.build_validation_lightcurves(
        sample_count=1,
        seed=222,
        cache_dir=tmp_path,
    )

    assert calls == [111, 222]
    assert first[3]["cache_hit"] is False
    assert second[3]["cache_hit"] is False
    assert second[3]["seed"] == 222
    assert np.array_equal(second[0], [[1.0, 1.1, 1.0]])

    reused = module.build_validation_lightcurves(sample_count=1, seed=222, cache_dir=tmp_path)
    assert reused[3]["cache_hit"] is True
    assert calls == [111, 222]


def test_validation_cache_invalidates_when_simulator_commit_changes(monkeypatch, tmp_path):
    module = _load_generator_module()
    calls = []
    identity = {"selflensing": {"version": "0.0.1", "commit_id": "before-fix"}}
    monkeypatch.setattr(module, "dependency_provenance", lambda *names: identity.copy())

    def generate(**kwargs):
        calls.append(kwargs["seed"])
        return (np.array([[float(kwargs["seed"])]]), np.array([0.]),
                np.array([(1.,)], dtype=[("a", "f8")]), {"seed": kwargs["seed"]})

    monkeypatch.setattr(module, "_build_validation_lightcurves_uncached", generate)
    old = module.build_validation_lightcurves(sample_count=1, seed=111, cache_dir=tmp_path)
    identity["selflensing"] = {"version": "0.0.1", "commit_id": "after-fix"}
    fixed = module.build_validation_lightcurves(sample_count=1, seed=222, cache_dir=tmp_path)
    reused = module.build_validation_lightcurves(sample_count=1, seed=222, cache_dir=tmp_path)
    assert calls == [111, 222]
    assert old[3]["cache_path"] != fixed[3]["cache_path"]
    assert not fixed[3]["cache_hit"]
    assert reused[3]["cache_hit"]
    assert reused[3]["simulator_provenance"]["selflensing"]["commit_id"] == "after-fix"
    assert reused[0][0, 0] == 222.


def test_validation_cache_can_be_disabled(monkeypatch, tmp_path):
    module = _load_generator_module()
    calls = []

    def fake_build_uncached(**kwargs):
        calls.append(kwargs["seed"])
        return (
            np.asarray([[float(kwargs["seed"])]]),
            np.asarray([0.0]),
            np.asarray([(1.0,)], dtype=[("a", "f8")]),
            {
                "sample_count": kwargs["sample_count"],
                "seed": kwargs["seed"],
                "cadence_minutes": kwargs["cadence_minutes"],
                "half_window_days": kwargs["half_window_days"],
                "window_scale": kwargs["window_scale"],
                "include_distance": kwargs["include_distance"],
                "max_eccentricity": kwargs["max_eccentricity"],
                "allow_rlof": kwargs["allow_rlof"],
                "min_flux": kwargs["min_flux"],
                "skipped_invalid_flux": 0,
            },
        )

    monkeypatch.setattr(
        module,
        "_build_validation_lightcurves_uncached",
        fake_build_uncached,
    )

    module.build_validation_lightcurves(
        sample_count=1,
        seed=111,
        cache_dir=tmp_path,
        use_cache=False,
    )
    module.build_validation_lightcurves(
        sample_count=1,
        seed=222,
        cache_dir=tmp_path,
        use_cache=False,
    )

    assert calls == [111, 222]
    assert not list(tmp_path.glob("*.npz"))


def test_default_templates_use_shared_proposal_and_zero_center(monkeypatch):
    module = _load_generator_module()
    observed = {}
    def sample(count, rng, **kwargs):
        observed.update(count=count, ranges=kwargs["ranges"], min_crossing_time_days=kwargs["min_crossing_time_days"])
        rows = {name: np.full(count, value) for name, value in {
            "a": 10., "i": np.pi/2, "Macc": 5., "Mcomp": 1., "e": 0.,
            "periapsis_phase": 0., "d": 3., "Porb": 100., "b": 0.,
            "tau_eff": .5, "Rcomp": 1., "Teff": 5800.}.items()}
        return (rows, 1024) if kwargs.get("return_attempts") else rows
    monkeypatch.setattr(module.generation, "sample_systems", sample)
    # Exercise the real shared deterministic forward model.
    flux, time, parameters, metadata = module.build_template_library(template_count=2, seed=15)
    assert observed["count"] == 2
    assert observed["ranges"] == module.generation.DEFAULT_RANGES
    assert metadata["grid_preset"] == "default"
    assert metadata["cadence_minutes"] == 144.
    assert metadata["min_crossing_time_days"] == .2
    assert observed["min_crossing_time_days"] == .2
    assert metadata["noisy"] is False
    assert metadata["shared_generator"]["sampling_distributions"]["Mcomp"] == "uniform"
    assert 0. in time
    assert np.all(parameters["d"] == 3.)
    assert flux.shape == (2, len(time))
    np.testing.assert_allclose(np.diff(time), 144 / 1440)


def test_validation_uses_same_shared_proposal_and_custom_ranges(monkeypatch):
    module = _load_generator_module()
    calls = []
    original = module.generation.sample_systems
    def sample(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(module.generation, "sample_systems", sample)
    ranges = {"i": [np.pi/2, np.pi/2], "a": [10,10], "Macc": [5,5],
              "Mcomp": [1,1], "d": [.1,.1], "e": [0,0]}
    flux, time, params, metadata = module.build_validation_lightcurves(sample_count=2, seed=17,
        ranges=ranges, half_window_days=.1, use_cache=False)
    assert calls[0]["ranges"]["Mcomp"] == (1.,1.)
    assert calls[0]["min_crossing_time_days"] == 0.
    assert np.all(params["d"] == .1)
    assert metadata["shared_generator"]["ranges"]["d"] == (.1,.1)
    assert 0 in time
    assert flux.shape == (2,len(time))


def test_build_template_library_skips_invalid_flux(monkeypatch):
    module = _load_generator_module()
    systems = [(3., {"a": float(index + 1), "i": np.pi/2, "Macc": 5., "Mcomp": 1.,
                      "e": 0., "periapsis_phase": 0., "d": 1.},
                {"Porb": 100., "b": 0., "tau_eff": 3., "Rcomp": 1., "Teff": 5800.})
               for index in range(3)]
    monkeypatch.setattr(module, "_sample_template_systems", lambda *args, **kwargs: (systems, 1024))
    def factory(times, **kwargs):
        assert kwargs["db"] is False
        def forward(rows):
            return np.asarray([np.full(len(times), -1. if row["a"] == 2. else 1.) for row in rows])
        return forward
    _patch_shared_generator(monkeypatch, module, factory)
    flux, time, parameters, metadata = module.build_template_library(template_count=3,
        cadence_minutes=1440, half_window_days=1, include_doppler_boosting=False)
    assert flux.shape == (2, 3)
    assert parameters["a"].tolist() == [1., 3.]
    assert metadata["skipped_invalid_flux"] == 1
    assert metadata["grid_size_before_filters"] == 1024
    assert metadata["grid_size_after_filters"] == 3
    assert metadata["include_doppler_boosting"] is False
    assert metadata["grid_preset"] == "default"


@pytest.mark.parametrize("preset", ["coarse", "completeness", "limits"])
def test_removed_grid_presets_are_rejected_by_api_and_cli(monkeypatch, preset):
    import sys
    module = _load_generator_module()
    with pytest.raises(ValueError, match="Only the default preset"):
        module.build_template_library(grid_preset=preset)
    monkeypatch.setattr(sys, "argv", ["generate_templates", "--grid-preset", preset])
    with pytest.raises(SystemExit):
        module.parse_args()


def test_calibration_builds_one_bank_and_selects_best_below_target(monkeypatch):
    module = _load_generator_module()
    builds = []
    monkeypatch.setattr(module, "build_validation_lightcurves", lambda **kwargs: (
        np.ones((4, 3)), np.arange(3.), None, {"seed": kwargs["seed"]}))
    def build(**kwargs):
        builds.append(kwargs)
        return np.ones((2, 3)), np.arange(3.), None, {"grid_preset": "default"}
    monkeypatch.setattr(module, "build_template_library", build)
    recalls = iter([.25, .5, .25])
    def evaluate(**kwargs):
        recall = next(recalls)
        return {"recall": recall, "matched": int(4 * recall), "total": 4,
                "prominence_sigma": kwargs["prominence_sigma"],
                "min_score_sigma": kwargs["min_score_sigma"]}
    monkeypatch.setattr(module, "evaluate_template_completeness", evaluate)
    _, _, _, metadata = module.calibrate_template_library(target_recall=.9,
        validation_size=4, validation_seed=123, threshold_grid=[(3, 4), (2, 3), (1, 2)],
        workers=1, validation_backend="cpu")
    assert len(builds) == 1
    assert builds[0]["grid_preset"] == "default"
    assert len(metadata["calibration"]["evaluations"]) == 3
    assert metadata["calibration"]["selected"]["recall"] == .5
    assert metadata["calibration"]["selected"]["prominence_sigma"] == 2


def test_saved_templates_all_have_two_cadence_intervals_per_crossing():
    from searchlight.matched_filtering import save_template_library
    module = _load_generator_module()
    flux, times, params, metadata = module.build_template_library(template_count=4, seed=17,
        generation_batch_size=2, ranges={"i": [np.pi/2, np.pi/2], "a": [.1, 100.],
            "Macc": [5.,5.], "Mcomp": [1.,1.], "e": [0.,0.], "d": [1.,1.]})
    save_template_library("3.0", flux=flux, time=times, parameters=params, metadata=metadata)
    bank = load_template_library("3.0")
    minimum = 2 * bank.metadata["cadence_minutes"] / 1440
    assert len(bank) == 4
    assert np.all(bank.parameters["tau_eff"] >= minimum)
    assert bank.metadata["min_crossing_time_days"] == minimum
