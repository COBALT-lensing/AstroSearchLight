#!/usr/bin/env python
"""Generate a self-lensing matched-filter template library."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import itertools
import json
import os
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
from astropy import units
from tqdm import tqdm

from searchlight.matched_filtering import (
    DEFAULT_TEMPLATE_LIBRARY_PATH,
    MatchedFilterTemplateLibrary,
    detect_matched_filter_peaks,
    save_template_library,
)


DEFAULT_VALIDATION_CACHE_DIR = Path(".cache/matched_filter_validation_cache")
PARAMETER_FIELDS = [
    ("a", "f8"),
    ("i", "f8"),
    ("Macc", "f8"),
    ("Mcomp", "f8"),
    ("e", "f8"),
    ("periapsis_phase", "f8"),
    ("d", "f8"),
    ("Porb", "f8"),
    ("b", "f8"),
    ("tau_eff", "f8"),
]
DEFAULT_MAX_ECCENTRICITY = 0.999
DEFAULT_COARSE_GRID = {
    "a": [0.3, 1.0, 10.0, 100.0, 200.0],
    "i": [0.5 * np.pi],
    "Macc": [0.5, 1.0, 5.0, 20.0, 50.0],
    "Mcomp": [0.5, 1.0, 5.0, 20.0, 50.0],
    "e": [0.0, 0.3, 0.6, 0.9],
    "periapsis_phase": [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi],
    "d": [1.0],
}
DEFAULT_COMPLETENESS_GRID = {
    "a": [0.1, 0.2, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0, 200.0],
    "i": [0.5 * np.pi - 1e-4, 0.5 * np.pi - 5e-5, 0.5 * np.pi],
    "Macc": [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
    "Mcomp": [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
    "e": [0.0, 0.15, 0.35, 0.6, 0.8, 0.93, DEFAULT_MAX_ECCENTRICITY],
    "periapsis_phase": list(np.linspace(0.0, 2 * np.pi, 8, endpoint=False)),
    "d": [1.0],
}
DEFAULT_CALIBRATION_THRESHOLD_GRID = [
    (3.0, 4.0),
    (2.5, 3.0),
    (2.0, 2.0),
    (1.5, 1.0),
    (1.0, None),
]
DEFAULT_FILTER_BATCH_SIZE = 65536
DEFAULT_GENERATION_BATCH_SIZE = 512
_VALIDATION_EVALUATOR_CONTEXT = None


def _require_selflensing():
    try:
        from selflensing.batch import make_batched_light_curve_generator
        from selflensing.models import DEFAULT_LIMITS, SelfLensingSystem
    except ImportError as exc:
        raise ImportError(
            "Template generation requires py-self-lensing. Install it with "
            "`uv sync --extra template-generation --all-groups` or add the "
            "local py-self-lensing checkout to PYTHONPATH."
        ) from exc

    return make_batched_light_curve_generator, DEFAULT_LIMITS, SelfLensingSystem


def _grid_values(lower, upper, count, *, distribution):
    if count <= 0:
        raise ValueError("grid counts must be positive")
    if count == 1:
        return np.asarray([(lower + upper) / 2], dtype=float)
    if distribution == "log":
        offset = 1 if lower == 0 else 0
        values = np.geomspace(lower + offset, upper + offset, count) - offset
    else:
        values = np.linspace(lower, upper, count)
    values[0] = lower
    values[-1] = upper
    return values


def _values_from_csv(value):
    if value is None:
        return None
    return [float(item) for item in value.split(",") if item.strip()]


def build_parameter_grid(
    *,
    samples_per_parameter=None,
    distance_samples=1,
    include_distance=False,
    max_eccentricity=DEFAULT_MAX_ECCENTRICITY,
    grid_preset="coarse",
    parameter_values=None,
):
    _, default_limits, _ = _require_selflensing()
    if not 0 <= max_eccentricity < 1:
        raise ValueError("max_eccentricity must be >= 0 and less than 1")

    if grid_preset not in {"coarse", "completeness", "limits"}:
        raise ValueError("grid_preset must be 'coarse', 'completeness', or 'limits'")

    if parameter_values is None:
        parameter_values = {}

    if grid_preset in {"coarse", "completeness"}:
        source_grid = (
            DEFAULT_COMPLETENESS_GRID
            if grid_preset == "completeness"
            else DEFAULT_COARSE_GRID
        )
        values = {
            name: list(grid_values)
            for name, grid_values in source_grid.items()
        }
        if include_distance:
            values["d"] = list(
                _grid_values(
                    float(default_limits["d"][0]),
                    float(default_limits["d"][1]),
                    distance_samples,
                    distribution="log",
                )
            )
    else:
        if samples_per_parameter is None:
            samples_per_parameter = 3
        counts = {
            "a": samples_per_parameter,
            "i": samples_per_parameter,
            "Macc": samples_per_parameter,
            "Mcomp": samples_per_parameter,
            "e": samples_per_parameter,
            "periapsis_phase": samples_per_parameter,
            "d": distance_samples if include_distance else 1,
        }
        log_parameters = {"a", "e", "Macc", "Mcomp", "d"}
        values = {}
        for name, count in counts.items():
            lower, upper = default_limits[name]
            if name == "d" and not include_distance:
                lower = upper = 1.0
            if name == "e":
                upper = min(float(upper), max_eccentricity)
            values[name] = list(
                _grid_values(
                    float(lower),
                    float(upper),
                    count,
                    distribution="log" if name in log_parameters else "linear",
                )
            )

    for name, explicit_values in parameter_values.items():
        if explicit_values is not None:
            values[name] = list(explicit_values)

    values["e"] = [value for value in values["e"] if value <= max_eccentricity]
    if not values["e"]:
        raise ValueError("eccentricity grid is empty after applying max_eccentricity")

    names = list(values)
    for combo in itertools.product(*(values[name] for name in names)):
        yield dict(zip(names, combo))


def _system_from_params(params):
    _, _, self_lensing_system = _require_selflensing()
    return self_lensing_system(
        a=params["a"] * units.AU,
        i=params["i"] * units.rad,
        Macc=params["Macc"] * units.M_sun,
        Mcomp=params["Mcomp"] * units.M_sun,
        e=params["e"] * units.dimensionless_unscaled,
        periapsis_phase=params["periapsis_phase"] * units.rad,
        d=params["d"] * units.kpc,
    )


def _parameter_row(params, derived):
    row = np.zeros((), dtype=PARAMETER_FIELDS)
    for name in ("a", "i", "Macc", "Mcomp", "e", "periapsis_phase", "d"):
        row[name] = params[name]
    row["Porb"] = derived["Porb"]
    row["b"] = derived["b"]
    row["tau_eff"] = derived["tau_eff"]
    return row


def _quantity_to_float(quantity, unit):
    if hasattr(quantity, "to_value"):
        value = quantity.to_value(unit)
    elif hasattr(quantity, "to"):
        value = quantity.to(unit).value
    else:
        value = quantity
    return float(np.asarray(value).item())


def _system_derived_values(system):
    return {
        "Porb": _quantity_to_float(system.porb, units.day),
        "b": _quantity_to_float(system.b, units.dimensionless_unscaled),
        "tau_eff": _quantity_to_float(system.tau_eff, units.day),
        "Rcomp": _quantity_to_float(system.Rcomp, units.R_sun),
        "Teff": _quantity_to_float(system.Teff, units.K),
    }


def _batched_light_curve_params(params, derived):
    return {
        "a": params["a"],
        "i": params["i"],
        "Macc": params["Macc"],
        "Mcomp": params["Mcomp"],
        "Rcomp": derived["Rcomp"],
        "Teff": derived["Teff"],
        "e": params["e"],
        "periapsis_phase": params["periapsis_phase"],
    }


def _as_bool(value):
    if hasattr(value, "value"):
        value = value.value
    return bool(np.asarray(value).item())


def _is_valid_template_flux(flux, *, min_flux):
    flux = np.asarray(flux, dtype=float)
    return np.all(np.isfinite(flux)) and np.min(flux) > min_flux


def _status(message, *, enabled):
    if enabled:
        tqdm.write(message)


def _default_workers():
    return min(4, os.cpu_count() or 1)


def _normalise_workers(value, name):
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _jax_devices_for_kind(kind):
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import jax

            return jax.devices(kind)
    except Exception:
        return []


def _selflensing_generation_device():
    devices = _jax_devices_for_kind("gpu")
    return devices[0] if devices else None


def _selflensing_generation_device_summary():
    gpu_devices = _jax_devices_for_kind("gpu")
    if gpu_devices:
        return "batched JAX light curves; GPU device(s): " + ", ".join(
            str(device) for device in gpu_devices
        )
    cpu_devices = _jax_devices_for_kind("cpu")
    if cpu_devices:
        return "batched JAX light curves; CPU fallback: " + ", ".join(
            str(device) for device in cpu_devices
        )
    return "batched JAX light curves; unknown device, JAX devices unavailable"


def _filter_batch_device_summary():
    gpu_devices = _jax_devices_for_kind("gpu")
    if gpu_devices:
        return "batched JAX system filtering; GPU device(s): " + ", ".join(
            str(device) for device in gpu_devices
        )
    cpu_devices = _jax_devices_for_kind("cpu")
    if cpu_devices:
        return "batched JAX system filtering; CPU fallback: " + ", ".join(
            str(device) for device in cpu_devices
        )
    return "batched JAX system filtering; unknown device, JAX devices unavailable"


def _grid_parameter_array(grid):
    names = ("a", "i", "Macc", "Mcomp", "e", "periapsis_phase", "d")
    return np.asarray([[params[name] for name in names] for params in grid], dtype=float)


def _evaluate_filter_batch(params_array, *, allow_rlof):
    import jax
    import jax.numpy as jnp
    from selflensing.gp.PhysicalRelations import (
        effective_temperature_K,
        stellar_radius_Rsun,
    )
    from selflensing.orbits import (
        AU_M,
        C,
        DAY_S,
        G,
        M_SUN_KG,
        R_SUN_M,
        orbital_period_days,
        orbital_separation_m,
    )

    params_array = jnp.asarray(params_array)

    def evaluate_one(row):
        a, i, macc, mcomp, e, periapsis_phase, _ = row
        a_m = a * AU_M
        macc_kg = macc * M_SUN_KG
        mcomp_kg = mcomp * M_SUN_KG
        total_mass_kg = macc_kg + mcomp_kg
        rcomp_rsun = stellar_radius_Rsun(mcomp)
        rcomp_m = rcomp_rsun * R_SUN_M
        teff = effective_temperature_K(mcomp)
        sin_i = jnp.sin(i)
        porb_days = orbital_period_days(a_m, total_mass_kg)

        separation_at_peak = orbital_separation_m(0.0, a_m, e, periapsis_phase)
        r_e_m = jnp.sqrt((4 * G * macc_kg * separation_at_peak * sin_i) / (C**2))
        b = jnp.abs(a_m * jnp.cos(i) / r_e_m)
        b_max = 1 + (rcomp_m / r_e_m)
        sl_possible = b <= b_max

        q = mcomp / macc
        q23 = q ** (2.0 / 3.0)
        separation_at_periapsis = orbital_separation_m(
            periapsis_phase,
            a_m,
            e,
            periapsis_phase,
        )
        r_roche = separation_at_periapsis * (
            (0.49 * q23) / (0.6 * q23 + jnp.log(1 + q ** (1.0 / 3.0)))
        )
        rlof = r_roche <= rcomp_m

        tau_eff_days = (
            ((porb_days * DAY_S * (r_e_m + rcomp_m)) / (jnp.pi * a_m * sin_i))
            * jnp.sqrt(1 - (b / b_max) ** 2)
        ) / DAY_S

        finite = (
            jnp.isfinite(porb_days)
            & jnp.isfinite(b)
            & jnp.isfinite(tau_eff_days)
            & jnp.isfinite(rcomp_rsun)
            & jnp.isfinite(teff)
            & (r_e_m > 0)
            & (rcomp_m > 0)
            & (tau_eff_days > 0)
        )
        keep = finite & sl_possible
        if not allow_rlof:
            keep = keep & ~rlof
        return keep, porb_days, b, tau_eff_days, rcomp_rsun, teff

    return jax.jit(jax.vmap(evaluate_one))(params_array)


def _filter_template_systems_batched(
    grid,
    *,
    allow_rlof,
    max_templates,
    show_progress,
    grid_preset,
    batch_size=DEFAULT_FILTER_BATCH_SIZE,
):
    systems = []
    progress = tqdm(
        total=len(grid),
        desc=f"Filtering {grid_preset} systems",
        disable=not show_progress,
        unit="system",
    )
    try:
        for start in range(0, len(grid), batch_size):
            chunk = grid[start : start + batch_size]
            params_array = _grid_parameter_array(chunk)
            keep, porb, b, tau_eff, rcomp, teff = _evaluate_filter_batch(
                params_array,
                allow_rlof=allow_rlof,
            )
            if hasattr(keep, "block_until_ready"):
                keep.block_until_ready()
            keep = np.asarray(keep, dtype=bool)
            derived_columns = {
                "Porb": np.asarray(porb, dtype=float),
                "b": np.asarray(b, dtype=float),
                "tau_eff": np.asarray(tau_eff, dtype=float),
                "Rcomp": np.asarray(rcomp, dtype=float),
                "Teff": np.asarray(teff, dtype=float),
            }
            for index, params in enumerate(chunk):
                if not keep[index]:
                    continue
                derived = {
                    name: float(values[index])
                    for name, values in derived_columns.items()
                }
                systems.append((derived["tau_eff"], params, derived))
                if max_templates is not None and len(systems) >= max_templates:
                    return systems
            progress.update(len(chunk))
    finally:
        progress.close()
    return systems


def _build_lightcurve_rows_batched(
    tasks,
    *,
    relative_time_days,
    show_progress,
    desc,
    batch_size=DEFAULT_GENERATION_BATCH_SIZE,
):
    make_batched_light_curve_generator, _, _ = _require_selflensing()
    batch_size = _normalise_workers(batch_size, "generation_batch_size")

    curve_params = []
    params_rows = []
    for task in tasks:
        params = task[0]
        derived = task[1]
        curve_params.append(_batched_light_curve_params(params, derived))
        params_rows.append(_parameter_row(params, derived))

    if not curve_params:
        return [], [], 0

    first_task = tasks[0]
    generate = make_batched_light_curve_generator(
        relative_time_days,
        peak_times=0.0,
        sl=first_task[3],
        ev=first_task[4],
        db=first_task[5],
        device=_selflensing_generation_device(),
    )
    flux_rows = []
    parameter_rows = []
    skipped_invalid_flux = 0

    progress = tqdm(
        total=len(curve_params),
        desc=desc,
        disable=not show_progress,
        unit="curve",
    )
    try:
        for start in range(0, len(curve_params), batch_size):
            params_chunk = curve_params[start : start + batch_size]
            rows_chunk = params_rows[start : start + batch_size]
            flux_array = generate(params_chunk)
            if hasattr(flux_array, "block_until_ready"):
                flux_array = flux_array.block_until_ready()
            flux = np.asarray(flux_array, dtype=float)
            expected_shape = (len(params_chunk), len(relative_time_days))
            if flux.shape != expected_shape:
                raise ValueError(
                    "Batched selflensing generator returned flux with shape "
                    f"{flux.shape}; expected {expected_shape}"
                )
            for flux_row, parameter_row in zip(flux, rows_chunk):
                if not _is_valid_template_flux(flux_row, min_flux=first_task[6]):
                    skipped_invalid_flux += 1
                    continue
                flux_rows.append(flux_row)
                parameter_rows.append(parameter_row)
            progress.update(len(params_chunk))
    finally:
        progress.close()
    return flux_rows, parameter_rows, skipped_invalid_flux


def build_template_library(
    *,
    samples_per_parameter=None,
    cadence_minutes=120.0,
    half_window_days=None,
    window_scale=1.2,
    include_distance=False,
    distance_samples=1,
    max_templates=None,
    include_self_lensing=True,
    include_ellipsoidal_variations=True,
    include_doppler_boosting=True,
    show_progress=False,
    max_eccentricity=DEFAULT_MAX_ECCENTRICITY,
    allow_rlof=False,
    min_flux=0.0,
    generation_batch_size=DEFAULT_GENERATION_BATCH_SIZE,
    grid_preset="coarse",
    parameter_values=None,
):
    generation_batch_size = _normalise_workers(
        generation_batch_size,
        "generation_batch_size",
    )
    _status(f"Building {grid_preset!r} parameter grid...", enabled=show_progress)
    grid = list(
        build_parameter_grid(
            samples_per_parameter=samples_per_parameter,
            include_distance=include_distance,
            distance_samples=distance_samples,
            max_eccentricity=max_eccentricity,
            grid_preset=grid_preset,
            parameter_values=parameter_values,
        )
    )
    _status(
        f"Filtering {len(grid)} {grid_preset!r} grid systems for self-lensing "
        "and Roche-lobe constraints...",
        enabled=show_progress,
    )
    _status(
        f"Self-lensing filter device for {grid_preset!r}: "
        f"{_filter_batch_device_summary()}",
        enabled=show_progress,
    )
    systems = _filter_template_systems_batched(
        grid,
        allow_rlof=allow_rlof,
        max_templates=max_templates,
        show_progress=show_progress,
        grid_preset=grid_preset,
    )

    if not systems:
        raise RuntimeError("No self-lensing-capable systems were generated")
    _status(
        f"{grid_preset!r} grid kept {len(systems)} of {len(grid)} systems.",
        enabled=show_progress,
    )

    if half_window_days is None:
        _status(
            f"Computing shared template window for {grid_preset!r} grid...",
            enabled=show_progress,
        )
        max_tau_eff = max(tau_eff_days for tau_eff_days, _, _ in systems)
        half_window_days = max_tau_eff * window_scale

    cadence_days = cadence_minutes / (24 * 60)
    relative_time_days = np.arange(
        -half_window_days,
        half_window_days + (0.5 * cadence_days),
        cadence_days,
        dtype=float,
    )
    tasks = [
        (
            params,
            derived,
            relative_time_days,
            include_self_lensing,
            include_ellipsoidal_variations,
            include_doppler_boosting,
            min_flux,
        )
        for _, params, derived in systems
    ]
    _status(
        f"Self-lensing generation device for {grid_preset!r}: "
        f"{_selflensing_generation_device_summary()}",
        enabled=show_progress,
    )

    flux_rows, parameter_rows, skipped_invalid_flux = _build_lightcurve_rows_batched(
        tasks,
        relative_time_days=relative_time_days,
        show_progress=show_progress,
        desc="Generating templates",
        batch_size=generation_batch_size,
    )

    if not flux_rows:
        raise RuntimeError("No valid template flux rows were generated")

    metadata = {
        "cadence_minutes": cadence_minutes,
        "half_window_days": half_window_days,
        "window_scale": window_scale,
        "samples_per_parameter": samples_per_parameter,
        "include_distance": include_distance,
        "distance_samples": distance_samples,
        "include_self_lensing": include_self_lensing,
        "include_ellipsoidal_variations": include_ellipsoidal_variations,
        "include_doppler_boosting": include_doppler_boosting,
        "max_eccentricity": max_eccentricity,
        "allow_rlof": allow_rlof,
        "min_flux": min_flux,
        "skipped_invalid_flux": skipped_invalid_flux,
        "filter_interface": "jax.vmap",
        "filter_batch_size": DEFAULT_FILTER_BATCH_SIZE,
        "generation_batch_size": generation_batch_size,
        "generation_interface": "selflensing.batch.make_batched_light_curve_generator",
        "grid_preset": grid_preset,
        "parameter_values": {
            name: None if values is None else list(values)
            for name, values in (parameter_values or {}).items()
        },
        "grid_size_before_filters": len(grid),
        "grid_size_after_filters": len(systems),
    }
    return (
        np.vstack(flux_rows),
        relative_time_days,
        np.asarray(parameter_rows, dtype=PARAMETER_FIELDS),
        metadata,
    )


def _random_values(lower, upper, count, *, distribution, rng):
    lower = float(lower)
    upper = float(upper)
    if distribution == "log":
        if lower <= 0:
            raise ValueError("log-distributed random values require a positive lower")
        return np.exp(rng.uniform(np.log(lower), np.log(upper), count))
    return rng.uniform(lower, upper, count)


def _generate_validation_seed():
    return int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _validation_cache_key(settings):
    payload = json.dumps(_json_safe(settings), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _validation_cache_path(cache_dir, settings):
    return Path(cache_dir) / f"{_validation_cache_key(settings)}.npz"


def _load_validation_cache(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        parameters = data["parameters"] if "parameters" in data else None
        return data["flux"], data["time"], parameters, metadata


def _save_validation_cache(path, *, flux, time, parameters, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        flux=flux,
        time=time,
        parameters=parameters,
        metadata_json=np.asarray(json.dumps(_json_safe(metadata))),
    )


def sample_validation_parameters(
    *,
    sample_count=1000,
    seed=None,
    include_distance=False,
    max_eccentricity=DEFAULT_MAX_ECCENTRICITY,
    allow_rlof=False,
    max_attempts=None,
):
    _, default_limits, _ = _require_selflensing()
    if seed is None:
        seed = _generate_validation_seed()
    rng = np.random.default_rng(seed)
    if max_attempts is None:
        max_attempts = max(sample_count * 1000, 1000)

    samples = []
    attempts = 0
    log_parameters = {"a", "Macc", "Mcomp", "d"}
    while len(samples) < sample_count and attempts < max_attempts:
        attempts += 1
        params = {}
        for name in ("a", "i", "Macc", "Mcomp", "e", "periapsis_phase", "d"):
            lower, upper = default_limits[name]
            if name == "d" and not include_distance:
                params[name] = 1.0
                continue
            if name == "e":
                upper = min(float(upper), max_eccentricity)
            distribution = "log" if name in log_parameters else "linear"
            params[name] = float(
                _random_values(lower, upper, 1, distribution=distribution, rng=rng)[0]
            )

        system = _system_from_params(params)
        if not _as_bool(system.sl_possible):
            continue
        if not allow_rlof and _as_bool(system.rlof):
            continue
        samples.append((system, params))

    if len(samples) < sample_count:
        raise RuntimeError(
            f"Only generated {len(samples)} validation systems after "
            f"{attempts} attempts"
        )
    return samples


def _build_validation_lightcurves_uncached(
    *,
    sample_count=1000,
    seed,
    cadence_minutes=120.0,
    half_window_days=None,
    window_scale=1.2,
    include_distance=False,
    include_self_lensing=True,
    include_ellipsoidal_variations=True,
    include_doppler_boosting=True,
    show_progress=False,
    max_eccentricity=DEFAULT_MAX_ECCENTRICITY,
    allow_rlof=False,
    min_flux=0.0,
    generation_batch_size=DEFAULT_GENERATION_BATCH_SIZE,
):
    generation_batch_size = _normalise_workers(
        generation_batch_size,
        "generation_batch_size",
    )
    systems = sample_validation_parameters(
        sample_count=sample_count,
        seed=seed,
        include_distance=include_distance,
        max_eccentricity=max_eccentricity,
        allow_rlof=allow_rlof,
    )
    if half_window_days is None:
        max_tau_eff = max(_system_derived_values(system)["tau_eff"] for system, _ in systems)
        half_window_days = max_tau_eff * window_scale

    cadence_days = cadence_minutes / (24 * 60)
    relative_time_days = np.arange(
        -half_window_days,
        half_window_days + (0.5 * cadence_days),
        cadence_days,
        dtype=float,
    )
    tasks = [
        (
            params,
            _system_derived_values(system),
            relative_time_days,
            include_self_lensing,
            include_ellipsoidal_variations,
            include_doppler_boosting,
            min_flux,
        )
        for system, params in systems
    ]
    _status(
        "Self-lensing generation device for validation curves: "
        f"{_selflensing_generation_device_summary()}",
        enabled=show_progress,
    )
    flux_rows, parameter_rows, skipped_invalid_flux = _build_lightcurve_rows_batched(
        tasks,
        relative_time_days=relative_time_days,
        show_progress=show_progress,
        desc="Generating validation curves",
        batch_size=generation_batch_size,
    )

    if not flux_rows:
        raise RuntimeError("No valid validation light curves were generated")

    metadata = {
        "sample_count": sample_count,
        "seed": seed,
        "cadence_minutes": cadence_minutes,
        "half_window_days": half_window_days,
        "window_scale": window_scale,
        "include_distance": include_distance,
        "max_eccentricity": max_eccentricity,
        "allow_rlof": allow_rlof,
        "min_flux": min_flux,
        "skipped_invalid_flux": skipped_invalid_flux,
        "generation_batch_size": generation_batch_size,
        "generation_interface": "selflensing.batch.make_batched_light_curve_generator",
    }
    return (
        np.vstack(flux_rows),
        relative_time_days,
        np.asarray(parameter_rows, dtype=PARAMETER_FIELDS),
        metadata,
    )


def build_validation_lightcurves(
    *,
    sample_count=1000,
    seed=None,
    cadence_minutes=120.0,
    half_window_days=None,
    window_scale=1.2,
    include_distance=False,
    include_self_lensing=True,
    include_ellipsoidal_variations=True,
    include_doppler_boosting=True,
    show_progress=False,
    max_eccentricity=DEFAULT_MAX_ECCENTRICITY,
    allow_rlof=False,
    min_flux=0.0,
    use_cache=True,
    cache_dir=DEFAULT_VALIDATION_CACHE_DIR,
    generation_batch_size=DEFAULT_GENERATION_BATCH_SIZE,
):
    if seed is None:
        seed = _generate_validation_seed()
    cache_settings = {
        "sample_count": sample_count,
        "cadence_minutes": cadence_minutes,
        "half_window_days": half_window_days,
        "window_scale": window_scale,
        "include_distance": include_distance,
        "include_self_lensing": include_self_lensing,
        "include_ellipsoidal_variations": include_ellipsoidal_variations,
        "include_doppler_boosting": include_doppler_boosting,
        "max_eccentricity": max_eccentricity,
        "allow_rlof": allow_rlof,
        "min_flux": min_flux,
        "cache_version": 3,
    }
    cache_path = _validation_cache_path(cache_dir, cache_settings)
    if use_cache and cache_path.exists():
        _status(
            f"Loading validation curves from cache: {cache_path}",
            enabled=show_progress,
        )
        flux, time, parameters, metadata = _load_validation_cache(cache_path)
        metadata = {
            **metadata,
            "cache_hit": True,
            "cache_path": str(cache_path),
        }
        return flux, time, parameters, metadata

    flux, time, parameters, metadata = _build_validation_lightcurves_uncached(
        sample_count=sample_count,
        seed=seed,
        cadence_minutes=cadence_minutes,
        half_window_days=half_window_days,
        window_scale=window_scale,
        include_distance=include_distance,
        include_self_lensing=include_self_lensing,
        include_ellipsoidal_variations=include_ellipsoidal_variations,
        include_doppler_boosting=include_doppler_boosting,
        show_progress=show_progress,
        max_eccentricity=max_eccentricity,
        allow_rlof=allow_rlof,
        min_flux=min_flux,
        generation_batch_size=generation_batch_size,
    )
    metadata = {
        **metadata,
        "cache_hit": False,
        "cache_path": str(cache_path) if use_cache else None,
    }
    if use_cache:
        _status(f"Saving validation curves to cache: {cache_path}", enabled=show_progress)
        _save_validation_cache(
            cache_path,
            flux=flux,
            time=time,
            parameters=parameters,
            metadata=metadata,
        )
    return flux, time, parameters, metadata


def _initialise_validation_evaluator(context):
    global _VALIDATION_EVALUATOR_CONTEXT
    _VALIDATION_EVALUATOR_CONTEXT = {
        **context,
        "library": MatchedFilterTemplateLibrary(
            flux=context["template_flux"],
            time=context["template_time"],
        ),
    }


def _evaluate_validation_curve(flux):
    context = _VALIDATION_EVALUATOR_CONTEXT
    detection_flux = flux
    if context["pad_samples"]:
        baseline = float(np.median(flux))
        detection_flux = np.pad(
            flux,
            context["pad_samples"],
            mode="constant",
            constant_values=baseline,
        )
    matches = detect_matched_filter_peaks(
        context["detection_time"],
        detection_flux,
        context["library"],
        prominence_sigma=context["prominence_sigma"],
        min_score_sigma=context["min_score_sigma"],
        max_matches=5,
    )
    event_scores = [
        match.score
        for match in matches
        if abs(match.peak_index - context["expected_peak_index"])
        <= context["peak_tolerance_samples"]
    ]
    if not event_scores:
        return False, None
    return True, float(max(event_scores))


def _jax_gpu_available():
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import jax

            devices = jax.devices()
    except ImportError:
        return False
    except Exception:
        return False
    return any(device.platform == "gpu" for device in devices)


def _resolve_validation_backend(backend):
    if backend not in {"auto", "cpu", "jax"}:
        raise ValueError("validation_backend must be 'auto', 'cpu', or 'jax'")
    if backend == "auto":
        return "jax" if _jax_gpu_available() else "cpu"
    return backend


def _evaluate_template_completeness_jax(
    *,
    validation_flux,
    validation_time,
    template_flux,
    template_time,
    prominence_sigma,
    min_score_sigma,
    peak_tolerance_samples=2,
    show_progress=False,
    print_recall=False,
    batch_size=32,
    template_batch_size=512,
):
    import jax
    import jax.numpy as jnp
    from jax import lax

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if template_batch_size < 1:
        raise ValueError("template_batch_size must be >= 1")

    detection_time = validation_time
    detection_flux = validation_flux
    pad_samples = 0
    if len(validation_time) <= len(template_time):
        cadence = float(np.median(np.diff(validation_time)))
        pad_samples = len(template_time)
        left = validation_time[0] - cadence * np.arange(pad_samples, 0, -1)
        right = validation_time[-1] + cadence * np.arange(1, pad_samples + 1)
        detection_time = np.concatenate([left, validation_time, right])
        baseline = np.median(validation_flux, axis=1)
        detection_flux = np.pad(
            validation_flux,
            ((0, 0), (pad_samples, pad_samples)),
            mode="constant",
        )
        detection_flux[:, :pad_samples] = baseline[:, None]
        detection_flux[:, -pad_samples:] = baseline[:, None]

    expected_peak_index = pad_samples + (len(validation_time) // 2)
    template_len = len(template_time)
    template_center = template_len // 2
    output_len = len(detection_time) - template_len + 1
    if output_len < 1:
        raise ValueError("template_time cannot be longer than padded validation_time")

    # Keep shared matched-filter semantics in sync with the CPU detector in
    # searchlight.matched_filtering.detect_matched_filter_peaks.
    candidate_centers = jnp.arange(output_len) + template_center
    center_mask = jnp.abs(candidate_centers - expected_peak_index) <= (
        peak_tolerance_samples
    )

    def score_batch(batch_flux, template_batch):
        template_batch = template_batch - jnp.mean(
            template_batch,
            axis=1,
            keepdims=True,
        )
        template_energy = jnp.sqrt(jnp.sum(template_batch**2, axis=1, keepdims=True))
        template_batch = template_batch / template_energy
        filters = jnp.swapaxes(template_batch, 0, 1)[:, None, :]
        batch_flux = batch_flux - jnp.mean(batch_flux, axis=1, keepdims=True)
        lhs = batch_flux[:, :, None]
        scores = lax.conv_general_dilated(
            lhs,
            filters,
            window_strides=(1,),
            padding="VALID",
            dimension_numbers=("NWC", "WIO", "NWC"),
        )
        median = jnp.median(scores, axis=1, keepdims=True)
        mad = jnp.median(jnp.abs(scores - median), axis=1, keepdims=True)
        std = jnp.std(scores, axis=1, keepdims=True)
        scatter = jnp.where(mad > 0, 1.4826 * mad, jnp.where(std > 0, std, 1.0))

        left = jnp.pad(
            scores[:, :-1, :],
            ((0, 0), (1, 0), (0, 0)),
            constant_values=-jnp.inf,
        )
        right = jnp.pad(
            scores[:, 1:, :],
            ((0, 0), (0, 1), (0, 0)),
            constant_values=-jnp.inf,
        )
        local_max = (scores >= left) & (scores >= right)
        local_prominence = scores - jnp.maximum(left, right)
        prominent = local_prominence >= (prominence_sigma * scatter)
        if min_score_sigma is None:
            high_enough = jnp.ones_like(local_max)
        else:
            high_enough = scores >= (median + min_score_sigma * scatter)

        candidate_mask = center_mask[None, :, None]
        valid = local_max & prominent & high_enough & candidate_mask
        masked_scores = jnp.where(valid, scores, -jnp.inf)
        best_scores = jnp.max(masked_scores, axis=(1, 2))
        matched = jnp.isfinite(best_scores)
        best_scores = jnp.where(matched, best_scores, jnp.nan)
        return matched, best_scores

    score_batch = jax.jit(score_batch)
    matched_count = 0
    best_scores = []
    total = int(len(validation_flux))
    template_flux = np.asarray(template_flux, dtype=np.float32)
    progress = tqdm(
        total=total,
        desc="Evaluating validation curves",
        disable=not show_progress,
        unit="curve",
    )
    try:
        for batch_start in range(0, total, batch_size):
            batch = detection_flux[batch_start : batch_start + batch_size]
            batch_best_scores = np.full(len(batch), -np.inf, dtype=np.float32)
            batch_flux = jnp.asarray(batch, dtype=jnp.float32)
            for template_start in range(0, len(template_flux), template_batch_size):
                template_batch = template_flux[
                    template_start : template_start + template_batch_size
                ]
                matched_batch, score_batch_values = score_batch(
                    batch_flux,
                    jnp.asarray(template_batch, dtype=jnp.float32),
                )
                score_batch_values = np.asarray(score_batch_values)
                batch_best_scores = np.maximum(
                    batch_best_scores,
                    np.nan_to_num(score_batch_values, nan=-np.inf),
                )
            matched_batch = np.isfinite(batch_best_scores)
            for is_match, best_score in zip(matched_batch, batch_best_scores):
                if is_match:
                    matched_count += 1
                    best_scores.append(float(best_score))
            processed = min(batch_start + len(batch), total)
            if print_recall:
                recall = 100 * matched_count / processed
                if show_progress:
                    progress.set_postfix(
                        recall=f"{recall:.2f}%",
                        matched=f"{matched_count}/{processed}",
                        backend="jax",
                    )
                else:
                    end = "\n" if processed == total else "\r"
                    print(
                        f"Recall: {recall:.2f}% ({matched_count}/{processed} matched)",
                        end=end,
                        flush=True,
                    )
            progress.update(len(batch))
    finally:
        progress.close()

    return {
        "matched": int(matched_count),
        "total": total,
        "recall": float(matched_count / total) if total else 0.0,
        "prominence_sigma": prominence_sigma,
        "min_score_sigma": min_score_sigma,
        "peak_tolerance_samples": peak_tolerance_samples,
        "median_matched_score": (
            float(np.median(best_scores)) if best_scores else None
        ),
        "validation_backend": "jax",
        "jax_devices": [str(device) for device in jax.devices()],
    }


def evaluate_template_completeness(
    *,
    validation_flux,
    validation_time,
    template_flux,
    template_time,
    prominence_sigma,
    min_score_sigma,
    peak_tolerance_samples=2,
    show_progress=False,
    print_recall=False,
    workers=1,
    backend="auto",
    batch_size=32,
    template_batch_size=512,
):
    resolved_backend = _resolve_validation_backend(backend)
    if resolved_backend == "jax":
        return _evaluate_template_completeness_jax(
            validation_flux=validation_flux,
            validation_time=validation_time,
            template_flux=template_flux,
            template_time=template_time,
            prominence_sigma=prominence_sigma,
            min_score_sigma=min_score_sigma,
            peak_tolerance_samples=peak_tolerance_samples,
            show_progress=show_progress,
            print_recall=print_recall,
            batch_size=batch_size,
            template_batch_size=template_batch_size,
        )

    detection_time = validation_time
    pad_samples = 0
    if len(validation_time) <= len(template_time):
        cadence = float(np.median(np.diff(validation_time)))
        pad_samples = len(template_time)
        left = validation_time[0] - cadence * np.arange(pad_samples, 0, -1)
        right = validation_time[-1] + cadence * np.arange(1, pad_samples + 1)
        detection_time = np.concatenate([left, validation_time, right])
    expected_peak_index = pad_samples + (len(validation_time) // 2)
    context = {
        "template_flux": template_flux,
        "template_time": template_time,
        "detection_time": detection_time,
        "pad_samples": pad_samples,
        "expected_peak_index": expected_peak_index,
        "prominence_sigma": prominence_sigma,
        "min_score_sigma": min_score_sigma,
        "peak_tolerance_samples": peak_tolerance_samples,
    }
    matched = 0
    best_scores = []
    total = int(len(validation_flux))
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be >= 1")
    progress = tqdm(
        total=total,
        desc="Evaluating validation curves",
        disable=not show_progress,
        unit="curve",
    )

    def record_result(result, curve_index):
        nonlocal matched
        is_match, best_score = result
        if is_match:
            matched += 1
            best_scores.append(best_score)
        if print_recall:
            recall = 100 * matched / curve_index
            if show_progress:
                progress.set_postfix(
                    recall=f"{recall:.2f}%",
                    matched=f"{matched}/{curve_index}",
                )
            else:
                end = "\n" if curve_index == total else "\r"
                print(
                    f"Recall: {recall:.2f}% ({matched}/{curve_index} matched)",
                    end=end,
                    flush=True,
                )
        progress.update()

    try:
        if workers == 1:
            _initialise_validation_evaluator(context)
            for curve_index, flux in enumerate(validation_flux, start=1):
                record_result(_evaluate_validation_curve(flux), curve_index)
        else:
            chunksize = max(1, total // (workers * 4))
            with get_context("spawn").Pool(
                processes=workers,
                initializer=_initialise_validation_evaluator,
                initargs=(context,),
            ) as pool:
                results = pool.imap(
                    _evaluate_validation_curve,
                    validation_flux,
                    chunksize=chunksize,
                )
                for curve_index, result in enumerate(results, start=1):
                    record_result(result, curve_index)
    finally:
        progress.close()

    return {
        "matched": int(matched),
        "total": total,
        "recall": float(matched / total) if total else 0.0,
        "prominence_sigma": prominence_sigma,
        "min_score_sigma": min_score_sigma,
        "peak_tolerance_samples": peak_tolerance_samples,
        "median_matched_score": (
            float(np.median(best_scores)) if best_scores else None
        ),
        "validation_backend": "cpu",
    }


def calibrate_template_library(
    *,
    target_recall=0.9,
    validation_size=1000,
    validation_seed=None,
    threshold_grid=None,
    grid_presets=("coarse", "completeness"),
    samples_per_parameter=None,
    cadence_minutes=120.0,
    half_window_days=None,
    window_scale=1.2,
    include_distance=False,
    distance_samples=1,
    max_templates=None,
    include_self_lensing=True,
    include_ellipsoidal_variations=True,
    include_doppler_boosting=True,
    show_progress=False,
    max_eccentricity=DEFAULT_MAX_ECCENTRICITY,
    allow_rlof=False,
    min_flux=0.0,
    workers=None,
    cpu_workers=None,
    parameter_values=None,
    peak_tolerance_samples=2,
    use_validation_cache=True,
    validation_cache_dir=DEFAULT_VALIDATION_CACHE_DIR,
    validation_backend="auto",
    validation_batch_size=32,
    validation_template_batch_size=512,
    generation_batch_size=DEFAULT_GENERATION_BATCH_SIZE,
):
    if workers is None:
        workers = _default_workers()
    if cpu_workers is None:
        cpu_workers = workers
    cpu_workers = _normalise_workers(cpu_workers, "cpu_workers")
    if threshold_grid is None:
        threshold_grid = DEFAULT_CALIBRATION_THRESHOLD_GRID
    if validation_seed is None:
        validation_seed = _generate_validation_seed()
    resolved_validation_backend = _resolve_validation_backend(validation_backend)

    _status(
        f"Preparing {validation_size} validation curves "
        f"(requested seed={validation_seed})...",
        enabled=show_progress,
    )
    validation_flux, validation_time, _, validation_metadata = (
        build_validation_lightcurves(
            sample_count=validation_size,
            seed=validation_seed,
            cadence_minutes=cadence_minutes,
            half_window_days=half_window_days,
            window_scale=window_scale,
            include_distance=include_distance,
            include_self_lensing=include_self_lensing,
            include_ellipsoidal_variations=include_ellipsoidal_variations,
            include_doppler_boosting=include_doppler_boosting,
            show_progress=show_progress,
            max_eccentricity=max_eccentricity,
            allow_rlof=allow_rlof,
            min_flux=min_flux,
            use_cache=use_validation_cache,
            cache_dir=validation_cache_dir,
            generation_batch_size=generation_batch_size,
        )
    )
    _status(
        f"Validation curves ready "
        f"(seed={validation_metadata['seed']}, "
        f"cache_hit={validation_metadata.get('cache_hit', False)})",
        enabled=show_progress,
    )
    _status(
        f"Validation evaluation backend: {resolved_validation_backend}",
        enabled=show_progress,
    )

    best = None
    evaluations = []
    if not grid_presets:
        raise ValueError("At least one calibration grid preset is required")
    total_grid_presets = len(grid_presets)
    for grid_index, grid_preset in enumerate(grid_presets, start=1):
        _status(
            f"Calibration grid step {grid_index}/{total_grid_presets}: "
            f"{grid_preset!r}",
            enabled=show_progress,
        )
        flux, time, parameters, metadata = build_template_library(
            samples_per_parameter=samples_per_parameter,
            cadence_minutes=cadence_minutes,
            half_window_days=half_window_days,
            window_scale=window_scale,
            include_distance=include_distance,
            distance_samples=distance_samples,
            max_templates=max_templates,
            include_self_lensing=include_self_lensing,
            include_ellipsoidal_variations=include_ellipsoidal_variations,
            include_doppler_boosting=include_doppler_boosting,
            show_progress=show_progress,
            max_eccentricity=max_eccentricity,
            allow_rlof=allow_rlof,
            min_flux=min_flux,
            generation_batch_size=generation_batch_size,
            grid_preset=grid_preset,
            parameter_values=parameter_values,
        )
        _status(
            f"Generated {flux.shape[0]} templates for {grid_preset!r}; "
            f"evaluating {len(threshold_grid)} threshold settings.",
            enabled=show_progress,
        )
        for threshold_index, (prominence_sigma, min_score_sigma) in enumerate(
            threshold_grid,
            start=1,
        ):
            _status(
                f"Threshold step {threshold_index}/{len(threshold_grid)} for "
                f"{grid_preset!r}: prominence_sigma={prominence_sigma}, "
                f"min_score_sigma={min_score_sigma}",
                enabled=show_progress,
            )
            evaluation = evaluate_template_completeness(
                validation_flux=validation_flux,
                validation_time=validation_time,
                template_flux=flux,
                template_time=time,
                prominence_sigma=prominence_sigma,
                min_score_sigma=min_score_sigma,
                peak_tolerance_samples=peak_tolerance_samples,
                show_progress=show_progress,
                print_recall=show_progress,
                workers=cpu_workers,
                backend=resolved_validation_backend,
                batch_size=validation_batch_size,
                template_batch_size=validation_template_batch_size,
            )
            evaluation["grid_preset"] = grid_preset
            evaluation["templates"] = int(flux.shape[0])
            evaluations.append(evaluation)
            _status(
                f"Completed {grid_preset!r} threshold step "
                f"{threshold_index}/{len(threshold_grid)}: "
                f"recall={100 * evaluation['recall']:.2f}% "
                f"({evaluation['matched']}/{evaluation['total']} matched)",
                enabled=show_progress,
            )
            if best is None or evaluation["recall"] > best["evaluation"]["recall"]:
                best = {
                    "flux": flux,
                    "time": time,
                    "parameters": parameters,
                    "metadata": metadata,
                    "evaluation": evaluation,
                }
            if evaluation["recall"] >= target_recall:
                _status(
                    f"Target recall reached with {grid_preset!r}: "
                    f"{100 * evaluation['recall']:.2f}% >= "
                    f"{100 * target_recall:.2f}%",
                    enabled=show_progress,
                )
                metadata["calibration"] = {
                    "target_recall": target_recall,
                    "validation": validation_metadata,
                    "selected": evaluation,
                    "evaluations": evaluations,
                }
                return flux, time, parameters, metadata

    _status(
        f"Target recall was not reached. Selecting best result: "
        f"{best['evaluation']['grid_preset']!r} at "
        f"{100 * best['evaluation']['recall']:.2f}% recall.",
        enabled=show_progress,
    )
    best["metadata"]["calibration"] = {
        "target_recall": target_recall,
        "validation": validation_metadata,
        "selected": best["evaluation"],
        "evaluations": evaluations,
    }
    return best["flux"], best["time"], best["parameters"], best["metadata"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate self-lensing matched-filter templates."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_TEMPLATE_LIBRARY_PATH,
        help=f"Output .npz path. Defaults to {DEFAULT_TEMPLATE_LIBRARY_PATH}.",
    )
    parser.add_argument(
        "--samples-per-parameter",
        type=int,
        help=(
            "Number of samples for a, i, Macc, Mcomp, e, and periapsis_phase "
            "when --grid-preset limits is used."
        ),
    )
    parser.add_argument(
        "--grid-preset",
        choices=("coarse", "completeness", "limits"),
        default="coarse",
        help="Use the curated coarse grid or sample py-self-lensing DEFAULT_LIMITS.",
    )
    parser.add_argument(
        "--calibrate-completeness",
        action="store_true",
        help="Empirically choose a grid preset and thresholds on validation curves.",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.9,
        help="Validation-set recall target used by --calibrate-completeness.",
    )
    parser.add_argument(
        "--validation-size",
        type=int,
        default=1000,
        help="Number of random self-lensing validation curves to generate.",
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        help=(
            "Random seed for the validation set. If omitted, a non-deterministic "
            "seed is generated and recorded in the output metadata."
        ),
    )
    parser.add_argument(
        "--no-validation-cache",
        action="store_true",
        help="Disable reading and writing cached validation light curves.",
    )
    parser.add_argument(
        "--validation-cache-dir",
        type=Path,
        default=DEFAULT_VALIDATION_CACHE_DIR,
        help=(
            "Directory for cached validation light curves. Defaults to "
            f"{DEFAULT_VALIDATION_CACHE_DIR}."
        ),
    )
    parser.add_argument(
        "--validation-backend",
        choices=("auto", "cpu", "jax"),
        default="auto",
        help=(
            "Backend for validation evaluation. 'auto' uses JAX when a GPU is "
            "visible and otherwise uses the CPU detector."
        ),
    )
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=32,
        help="Validation-curve batch size for the JAX backend.",
    )
    parser.add_argument(
        "--validation-template-batch-size",
        type=int,
        default=512,
        help="Template batch size for the JAX validation backend.",
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=DEFAULT_GENERATION_BATCH_SIZE,
        help=(
            "Number of self-lensing systems per batched JAX light-curve "
            f"generation call. Defaults to {DEFAULT_GENERATION_BATCH_SIZE}."
        ),
    )
    parser.add_argument(
        "--calibration-grid-presets",
        default="coarse,completeness",
        help="Comma-separated grid presets to try during calibration.",
    )
    parser.add_argument(
        "--peak-tolerance-samples",
        type=int,
        default=2,
        help="Allowed offset from the injected event center when scoring recall.",
    )
    parser.add_argument(
        "--a-values",
        help="Comma-separated semi-major axis grid values in AU.",
    )
    parser.add_argument(
        "--i-values",
        help="Comma-separated inclination grid values in radians.",
    )
    parser.add_argument(
        "--macc-values",
        help="Comma-separated compact/accretor mass grid values in solar masses.",
    )
    parser.add_argument(
        "--mcomp-values",
        help="Comma-separated companion mass grid values in solar masses.",
    )
    parser.add_argument(
        "--e-values",
        help="Comma-separated eccentricity grid values.",
    )
    parser.add_argument(
        "--periapsis-phase-values",
        help="Comma-separated periapsis phase grid values in radians.",
    )
    parser.add_argument(
        "--cadence-minutes",
        type=float,
        default=120.0,
        help="Shared template cadence in minutes.",
    )
    parser.add_argument(
        "--half-window-days",
        type=float,
        help="Half-width of the shared relative-time grid. Defaults from tau_eff.",
    )
    parser.add_argument(
        "--window-scale",
        type=float,
        default=1.2,
        help="Multiplier applied to the largest tau_eff when half-window is omitted.",
    )
    parser.add_argument(
        "--include-distance",
        action="store_true",
        help="Vary distance over the selflensing default range instead of fixing 1 kpc.",
    )
    parser.add_argument(
        "--distance-samples",
        type=int,
        default=1,
        help="Number of distance samples when --include-distance is used.",
    )
    parser.add_argument(
        "--max-templates",
        type=int,
        help="Stop after this many self-lensing-capable systems.",
    )
    parser.add_argument(
        "--max-eccentricity",
        type=float,
        default=DEFAULT_MAX_ECCENTRICITY,
        help="Maximum eccentricity to include. Must be less than 1.",
    )
    parser.add_argument(
        "--allow-rlof",
        action="store_true",
        help="Include Roche-lobe overflow systems. Disabled by default.",
    )
    parser.add_argument(
        "--min-flux",
        type=float,
        default=0.0,
        help="Skip generated templates with flux at or below this value.",
    )
    parser.add_argument("--no-sl", action="store_true", help="Disable self-lensing.")
    parser.add_argument(
        "--no-ev",
        action="store_true",
        help="Disable ellipsoidal variations.",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Disable Doppler boosting.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the template generation progress bar.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_default_workers(),
        help=(
            "Compatibility alias for --cpu-workers. Used for CPU validation "
            "evaluation unless --cpu-workers is set."
        ),
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        help=(
            "Worker processes for CPU validation evaluation. Defaults to --workers."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    parameter_values = {
        "a": _values_from_csv(args.a_values),
        "i": _values_from_csv(args.i_values),
        "Macc": _values_from_csv(args.macc_values),
        "Mcomp": _values_from_csv(args.mcomp_values),
        "e": _values_from_csv(args.e_values),
        "periapsis_phase": _values_from_csv(args.periapsis_phase_values),
    }
    cpu_workers = args.cpu_workers if args.cpu_workers is not None else args.workers
    if args.calibrate_completeness:
        grid_presets = tuple(
            item.strip()
            for item in args.calibration_grid_presets.split(",")
            if item.strip()
        )
        flux, time, parameters, metadata = calibrate_template_library(
            target_recall=args.target_recall,
            validation_size=args.validation_size,
            validation_seed=args.validation_seed,
            grid_presets=grid_presets,
            samples_per_parameter=args.samples_per_parameter,
            cadence_minutes=args.cadence_minutes,
            half_window_days=args.half_window_days,
            window_scale=args.window_scale,
            include_distance=args.include_distance,
            distance_samples=args.distance_samples,
            max_templates=args.max_templates,
            include_self_lensing=not args.no_sl,
            include_ellipsoidal_variations=not args.no_ev,
            include_doppler_boosting=not args.no_db,
            show_progress=not args.no_progress,
            max_eccentricity=args.max_eccentricity,
            allow_rlof=args.allow_rlof,
            min_flux=args.min_flux,
            workers=args.workers,
            cpu_workers=cpu_workers,
            parameter_values=parameter_values,
            peak_tolerance_samples=args.peak_tolerance_samples,
            use_validation_cache=not args.no_validation_cache,
            validation_cache_dir=args.validation_cache_dir,
            validation_backend=args.validation_backend,
            validation_batch_size=args.validation_batch_size,
            validation_template_batch_size=args.validation_template_batch_size,
            generation_batch_size=args.generation_batch_size,
        )
    else:
        flux, time, parameters, metadata = build_template_library(
            samples_per_parameter=args.samples_per_parameter,
            cadence_minutes=args.cadence_minutes,
            half_window_days=args.half_window_days,
            window_scale=args.window_scale,
            include_distance=args.include_distance,
            distance_samples=args.distance_samples,
            max_templates=args.max_templates,
            include_self_lensing=not args.no_sl,
            include_ellipsoidal_variations=not args.no_ev,
            include_doppler_boosting=not args.no_db,
            show_progress=not args.no_progress,
            max_eccentricity=args.max_eccentricity,
            allow_rlof=args.allow_rlof,
            min_flux=args.min_flux,
            generation_batch_size=args.generation_batch_size,
            grid_preset=args.grid_preset,
            parameter_values=parameter_values,
        )
    output = save_template_library(
        args.output,
        flux=flux,
        time=time,
        parameters=parameters,
        metadata=metadata,
    )
    print(
        json.dumps(
            {
                "path": str(output),
                "templates": int(flux.shape[0]),
                "samples": int(flux.shape[1]),
                "skipped_invalid_flux": int(metadata["skipped_invalid_flux"]),
                "grid_size_before_filters": int(metadata["grid_size_before_filters"]),
                "grid_size_after_filters": int(metadata["grid_size_after_filters"]),
                "filter_interface": metadata.get("filter_interface"),
                "generation_batch_size": int(metadata["generation_batch_size"]),
                "calibration": metadata.get("calibration"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
