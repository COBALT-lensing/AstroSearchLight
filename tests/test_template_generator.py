import importlib.util
from pathlib import Path

import numpy as np
import pytest
from astropy import units

from searchlight.matched_filtering import load_template_library


def _load_generator_module():
    path = Path(__file__).parents[1] / "bin" / "generate_matched_filter_template_library.py"
    spec = importlib.util.spec_from_file_location("generate_templates", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_template_library_uses_selflensing_grid(monkeypatch):
    module = _load_generator_module()

    class FakeSystem:
        param_units = {}

        def __init__(self, a, i, Macc, Mcomp, e, periapsis_phase, d):
            self.a_value = a.to_value(units.AU)
            self.i_value = i.to_value(units.rad)
            self.Macc_value = Macc.to_value(units.M_sun)
            self.Mcomp_value = Mcomp.to_value(units.M_sun)
            self.e_value = e.value
            self.periapsis_phase_value = periapsis_phase.to_value(units.rad)
            self.d_value = d.to_value(units.kpc)
            self.sl_possible = self.a_value >= 1.0
            self.rlof = self.Macc_value == 5.0
            self.porb = (self.a_value + self.Macc_value) * units.day
            self.b = type("ImpactParameter", (), {"value": 0.25})()
            self.tau_eff = 2 * units.day
            self.Rcomp = 1 * units.R_sun
            self.Teff = 5800 * units.K

    def fake_make_batched_light_curve_generator(times, **kwargs):
        def generate(params):
            rows = []
            for row in params:
                flux = 1 + times + row["a"]
                if row["Mcomp"] == 20.0:
                    flux = np.asarray([1.0, -1.0, 1.0])
                rows.append(flux)
            return np.vstack(rows)

        return generate

    def fake_filter_systems(grid, **kwargs):
        systems = []
        for params in grid:
            system = FakeSystem(
                a=params["a"] * units.AU,
                i=params["i"] * units.rad,
                Macc=params["Macc"] * units.M_sun,
                Mcomp=params["Mcomp"] * units.M_sun,
                e=params["e"] * units.dimensionless_unscaled,
                periapsis_phase=params["periapsis_phase"] * units.rad,
                d=params["d"] * units.kpc,
            )
            if not system.sl_possible or system.rlof:
                continue
            derived = {
                "Porb": system.porb.to_value(units.day),
                "b": system.b.value,
                "tau_eff": system.tau_eff.to_value(units.day),
                "Rcomp": system.Rcomp.to_value(units.R_sun),
                "Teff": system.Teff.to_value(units.K),
            }
            systems.append((derived["tau_eff"], params, derived))
        return systems

    monkeypatch.setattr(
        module,
        "_require_selflensing",
        lambda: (
            fake_make_batched_light_curve_generator,
            {
                "a": (0.5, 2.0),
                "i": (1.0, 1.1),
                "Macc": (1.0, 4.0),
                "Mcomp": (1.0, 4.0),
                "e": (0.0, 0.5),
                "periapsis_phase": (0.0, 1.0),
                "d": (0.1, 10.0),
            },
            FakeSystem,
        ),
    )
    monkeypatch.setattr(module, "_filter_template_systems_batched", fake_filter_systems)

    flux, time, parameters, metadata = module.build_template_library(
        samples_per_parameter=2,
        cadence_minutes=24 * 60,
        half_window_days=1,
        include_doppler_boosting=False,
    )

    assert flux.shape[0] > 0
    assert flux.shape[1] == 3
    assert np.array_equal(time, [-1.0, 0.0, 1.0])
    assert parameters.dtype.names == tuple(name for name, _ in module.PARAMETER_FIELDS)
    assert np.all(parameters["a"] >= 1.0)
    assert np.all(parameters["Macc"] != 5.0)
    assert np.all(parameters["Mcomp"] != 20.0)
    assert np.max(parameters["e"]) <= module.DEFAULT_MAX_ECCENTRICITY
    assert metadata["include_doppler_boosting"] is False
    assert metadata["max_eccentricity"] == module.DEFAULT_MAX_ECCENTRICITY
    assert metadata["allow_rlof"] is False
    assert metadata["min_flux"] == 0.0
    assert metadata["skipped_invalid_flux"] > 0
    assert metadata["filter_interface"] == "jax.vmap"
    assert metadata["grid_preset"] == "coarse"
    assert metadata["grid_size_before_filters"] == 2000


def test_build_template_library_uses_batched_selflensing_generation(monkeypatch):
    module = _load_generator_module()
    calls = []

    class FakeSystem:
        param_units = {}

        def __init__(self, a, i, Macc, Mcomp, e, periapsis_phase, d):
            self.a_value = a.to_value(units.AU)
            self.i_value = i.to_value(units.rad)
            self.Macc_value = Macc.to_value(units.M_sun)
            self.Mcomp_value = Mcomp.to_value(units.M_sun)
            self.e_value = e.value
            self.periapsis_phase_value = periapsis_phase.to_value(units.rad)
            self.d_value = d.to_value(units.kpc)
            self.sl_possible = True
            self.rlof = False
            self.porb = 1 * units.day
            self.b = type("ImpactParameter", (), {"value": 0.25})()
            self.tau_eff = 1 * units.day
            self.Rcomp = 1 * units.R_sun
            self.Teff = 5800 * units.K

    def fake_make_batched_light_curve_generator(
        times,
        *,
        peak_times,
        sl,
        ev,
        db,
        device,
    ):
        calls.append(
            {
                "samples": len(times),
                "peak_times": peak_times,
                "sl": sl,
                "ev": ev,
                "db": db,
                "device": device,
            }
        )

        def generate(params):
            calls[-1]["systems"] = len(params)
            return np.vstack([1 + row["a"] + times for row in params])

        return generate

    def fake_filter_systems(grid, **kwargs):
        params = grid[0]
        derived = {
            "Porb": 1.0,
            "b": 0.25,
            "tau_eff": 1.0,
            "Rcomp": 1.0,
            "Teff": 5800.0,
        }
        return [(derived["tau_eff"], params, derived)]

    monkeypatch.setattr(
        module,
        "_require_selflensing",
        lambda: (
            fake_make_batched_light_curve_generator,
            {
                "a": (1.0, 2.0),
                "i": (1.0, 1.1),
                "Macc": (1.0, 2.0),
                "Mcomp": (1.0, 2.0),
                "e": (0.0, 0.1),
                "periapsis_phase": (0.0, 1.0),
                "d": (1.0, 1.0),
            },
            FakeSystem,
        ),
    )
    monkeypatch.setattr(module, "_filter_template_systems_batched", fake_filter_systems)

    flux, time, _, metadata = module.build_template_library(
        samples_per_parameter=1,
        cadence_minutes=24 * 60,
        half_window_days=1,
        grid_preset="limits",
        include_doppler_boosting=False,
    )

    assert calls == [
        {
            "systems": 1,
            "samples": 3,
            "peak_times": 0.0,
            "sl": True,
            "ev": True,
            "db": False,
            "device": None,
        }
    ]
    assert np.array_equal(time, [-1.0, 0.0, 1.0])
    assert np.allclose(flux, [[1.5, 2.5, 3.5]])
    assert (
        metadata["generation_interface"]
        == "selflensing.batch.make_batched_light_curve_generator"
    )
    assert metadata["generation_batch_size"] == module.DEFAULT_GENERATION_BATCH_SIZE


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

    monkeypatch.setattr(
        module,
        "_require_selflensing",
        lambda: (fake_make_batched_light_curve_generator, {}, object),
    )

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
    module = _load_generator_module()
    pytest.importorskip("selflensing")
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
        system = module._system_from_params(params)
        if not module._as_bool(system.sl_possible):
            continue
        if module._as_bool(system.rlof):
            continue
        expected.append((params, module._system_derived_values(system)))

    result = module._filter_template_systems_batched(
        grid,
        allow_rlof=False,
        max_templates=None,
        show_progress=False,
        grid_preset="test",
        batch_size=2,
    )

    assert [params for _, params, _ in result] == [params for params, _ in expected]
    for (_, _, derived), (_, expected_derived) in zip(result, expected):
        for name in ("Porb", "b", "tau_eff", "Rcomp", "Teff"):
            assert np.isclose(derived[name], expected_derived[name], rtol=1e-10)


def test_build_parameter_grid_rejects_parabolic_eccentricity(monkeypatch):
    module = _load_generator_module()
    monkeypatch.setattr(
        module,
        "_require_selflensing",
        lambda: (
            object,
            {
                "a": (0.5, 2.0),
                "i": (1.0, 1.1),
                "Macc": (1.0, 4.0),
                "Mcomp": (1.0, 4.0),
                "e": (0.0, 0.999999),
                "periapsis_phase": (0.0, 1.0),
                "d": (0.1, 10.0),
            },
            object,
        ),
    )

    with pytest.raises(ValueError, match="less than 1"):
        list(module.build_parameter_grid(max_eccentricity=1.0))

    grid = list(
        module.build_parameter_grid(samples_per_parameter=3, grid_preset="limits")
    )
    assert np.isclose(
        max(params["e"] for params in grid),
        module.DEFAULT_MAX_ECCENTRICITY,
    )


def test_generator_main_writes_default_library(monkeypatch, tmp_path, capsys):
    module = _load_generator_module()
    output = tmp_path / "templates.npz"
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "output": output,
                "samples_per_parameter": 1,
                "grid_preset": "coarse",
                "calibrate_completeness": False,
                "target_recall": 0.9,
                "validation_size": 1000,
                "validation_seed": 42,
                "generation_batch_size": 128,
                "calibration_grid_presets": "coarse,completeness",
                "peak_tolerance_samples": 2,
                "a_values": "5,20",
                "i_values": None,
                "macc_values": None,
                "mcomp_values": None,
                "e_values": "0,0.5",
                "periapsis_phase_values": None,
                "cadence_minutes": 60,
                "half_window_days": 1,
                "window_scale": 1.2,
                "include_distance": False,
                "distance_samples": 1,
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

    library = load_template_library(output)
    assert np.array_equal(library.flux, [[1.0, 2.0]])
    assert library.metadata["include_doppler_boosting"] is False
    assert library.metadata["max_eccentricity"] == 0.99
    assert library.metadata["allow_rlof"] is False
    assert library.metadata["min_flux"] == 0.0
    assert library.metadata["filter_interface"] == "jax.vmap"
    assert library.metadata["generation_batch_size"] == 128
    assert library.metadata["parameter_values"]["a"] == [5.0, 20.0]
    assert library.metadata["parameter_values"]["e"] == [0.0, 0.5]
    assert '"templates": 1' in capsys.readouterr().out


def test_build_parameter_grid_accepts_completeness_preset(monkeypatch):
    module = _load_generator_module()
    monkeypatch.setattr(
        module,
        "_require_selflensing",
        lambda: (
            object,
            {
                "a": (0.1, 200.0),
                "i": (1.5706963267948966, 1.5707963267948966),
                "Macc": (0.5, 50.0),
                "Mcomp": (0.5, 50.0),
                "e": (0.0, 0.999999),
                "periapsis_phase": (0.0, 2 * np.pi),
                "d": (0.1, 100.0),
            },
            object,
        ),
    )

    grid = module.build_parameter_grid(grid_preset="completeness")
    first = next(grid)

    assert first["a"] == 0.1
    assert np.isclose(first["i"], 0.5 * np.pi - 1e-4)
    assert first["e"] == 0.0


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
        grid_presets=("coarse",),
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
        grid_presets=("coarse",),
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
        grid_presets=("coarse",),
        workers=1,
    )

    assert seen["seed"] == 123456
    assert metadata["calibration"]["validation"]["seed"] == 123456


def test_validation_cache_ignores_seed_in_cache_key(monkeypatch, tmp_path):
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

    assert calls == [111]
    assert first[3]["cache_hit"] is False
    assert second[3]["cache_hit"] is True
    assert second[3]["seed"] == 111
    assert np.array_equal(second[0], [[1.0, 1.1, 1.0]])


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
