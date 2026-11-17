#!/usr/bin/env python
"""Generate a self-lensing matched-filter template library."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
from tqdm import tqdm

from searchlight.matched_filtering import (
    DEFAULT_TEMPLATE_COUNT,
    DEFAULT_TEMPLATE_CADENCE_MINUTES,
    MatchedFilterTemplateLibrary,
    detect_matched_filter_peaks,
    save_template_library,
)
from searchlight import generation, config
from searchlight.paths import CACHE_PATH
from searchlight.evaluation.provenance import dependency_provenance


DEFAULT_VALIDATION_CACHE_DIR = CACHE_PATH / "matched_filter_validation_cache"
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
DEFAULT_MAX_ECCENTRICITY = generation.DEFAULT_RANGES["e"][1]
DEFAULT_CALIBRATION_THRESHOLD_GRID = [
    (3.0, 4.0),
    (2.5, 3.0),
    (2.0, 2.0),
    (1.5, 1.0),
    (1.0, None),
]
DEFAULT_GENERATION_BATCH_SIZE = generation.DEFAULT_BATCH_SIZE
_VALIDATION_EVALUATOR_CONTEXT = None


def _parameter_row(params, derived):
    row = np.zeros((), dtype=PARAMETER_FIELDS)
    for name in ("a", "i", "Macc", "Mcomp", "e", "periapsis_phase", "d"):
        row[name] = params[name]
    row["Porb"] = derived["Porb"]
    row["b"] = derived["b"]
    row["tau_eff"] = derived["tau_eff"]
    return row


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


def _build_lightcurve_rows_batched(
    tasks,
    *,
    relative_time_days,
    show_progress,
    desc,
    batch_size=DEFAULT_GENERATION_BATCH_SIZE,
):
    batch_size = _normalise_workers(batch_size, "generation_batch_size")
    if not tasks:
        return [], [], 0
    simulator = generation.LightCurveGenerator()
    flux_rows, parameter_rows, skipped = [], [], 0
    first = tasks[0]
    for start in tqdm(range(0, len(tasks), batch_size), desc=desc, disable=not show_progress):
        chunk = tasks[start:start + batch_size]
        rows = {name: np.asarray([({**task[0], **task[1]})[name] for task in chunk])
                for name in (*generation.PARAMETER_NAMES, *generation.DERIVED_NAMES)}
        result = simulator.generate(rows, relative_time_days, peak_times=0.0, noisy=False,
                                    sl=first[3], ev=first[4], db=first[5])
        expected_shape = (len(chunk), len(relative_time_days))
        if result.flux.shape != expected_shape:
            raise ValueError(f"Shared generator returned {result.flux.shape}; expected {expected_shape}")
        for flux, task in zip(result.flux, chunk):
            if not _is_valid_template_flux(flux, min_flux=first[6]):
                skipped += 1
                continue
            flux_rows.append(flux)
            parameter_rows.append(_parameter_row(task[0], task[1]))
    return flux_rows, parameter_rows, skipped


def _default_ranges(ranges, *, include_distance, max_eccentricity):
    if not 0 <= max_eccentricity < 1:
        raise ValueError("max_eccentricity must be >= 0 and less than 1")
    bounds = generation.resolve_ranges(ranges)
    bounds["e"] = (bounds["e"][0], min(bounds["e"][1], max_eccentricity))
    if not include_distance:
        bounds["d"] = (1., 1.)
    return generation.resolve_ranges(bounds)


def _sample_template_systems(count, seed, *, include_distance=True,
                             max_eccentricity=DEFAULT_MAX_ECCENTRICITY,
                             allow_rlof=False, ranges=None, max_attempts=None, return_attempts=False,
                             min_crossing_time_days=0.0):
    bounds = _default_ranges(ranges, include_distance=include_distance, max_eccentricity=max_eccentricity)
    sampled = generation.sample_systems(count, np.random.default_rng(seed), ranges=bounds,
        allow_rlof=allow_rlof, max_attempts=max_attempts, return_attempts=return_attempts,
        min_crossing_time_days=min_crossing_time_days)
    rows, attempts = sampled if return_attempts else (sampled, None)
    systems = [(float(rows["tau_eff"][index]),
             {name: float(rows[name][index]) for name in generation.PARAMETER_NAMES},
             {name: float(rows[name][index]) for name in generation.DERIVED_NAMES})
            for index in range(count)]
    return (systems, attempts) if return_attempts else systems


def build_template_library(
    *,
    cadence_minutes=DEFAULT_TEMPLATE_CADENCE_MINUTES,
    half_window_days=None,
    window_scale=1.2,
    include_distance=True,
    max_templates=None,
    include_self_lensing=True,
    include_ellipsoidal_variations=True,
    include_doppler_boosting=True,
    show_progress=False,
    max_eccentricity=DEFAULT_MAX_ECCENTRICITY,
    allow_rlof=False,
    min_flux=0.0,
    generation_batch_size=DEFAULT_GENERATION_BATCH_SIZE,
    grid_preset="default",
    template_count=DEFAULT_TEMPLATE_COUNT,
    seed=generation.DEFAULT_SEED + 1,
    ranges=None,
):
    generation_batch_size = _normalise_workers(
        generation_batch_size,
        "generation_batch_size",
    )
    if grid_preset != "default":
        raise ValueError("Only the default preset is supported")
    cadence_days = cadence_minutes / (24 * 60)
    if not np.isfinite(cadence_days) or cadence_days <= 0:
        raise ValueError("Cadence must be positive and finite")
    # Reject unresolved events before generating their flux, and keep drawing
    # replacements until the requested number of eligible systems is reached.
    min_crossing_time_days = 2 * cadence_days
    count = min(template_count, max_templates) if max_templates is not None else template_count
    systems, proposal_draws = _sample_template_systems(count, seed, include_distance=include_distance,
        max_eccentricity=max_eccentricity, allow_rlof=allow_rlof, ranges=ranges, return_attempts=True,
        min_crossing_time_days=min_crossing_time_days)

    if not systems:
        raise RuntimeError("No self-lensing-capable systems were generated")
    _status(
        f"{grid_preset!r} proposal kept {len(systems)} of {proposal_draws} systems.",
        enabled=show_progress,
    )

    if half_window_days is None:
        _status(
            "Computing shared template window...",
            enabled=show_progress,
        )
        max_tau_eff = max(tau_eff_days for tau_eff_days, _, _ in systems)
        half_window_days = max_tau_eff * window_scale

    if not np.isfinite([cadence_days, half_window_days]).all() or min(cadence_days, half_window_days) <= 0:
        raise ValueError("Cadence and template half-window must be positive and finite")
    half_samples = int(np.ceil(half_window_days / cadence_days))
    relative_time_days = np.arange(-half_samples, half_samples + 1, dtype=float) * cadence_days
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
        "min_crossing_time_days": min_crossing_time_days,
        "half_window_days": half_window_days,
        "window_scale": window_scale,
        "include_distance": include_distance,
        "include_self_lensing": include_self_lensing,
        "include_ellipsoidal_variations": include_ellipsoidal_variations,
        "include_doppler_boosting": include_doppler_boosting,
        "max_eccentricity": max_eccentricity,
        "allow_rlof": allow_rlof,
        "min_flux": min_flux,
        "skipped_invalid_flux": skipped_invalid_flux,
        "filter_interface": "jax.vmap",
        "generation_batch_size": generation_batch_size,
        "generation_interface": "selflensing.batch.batched_light_curve_jit",
        "shared_generator": generation.generator_metadata(ranges=_default_ranges(ranges,
            include_distance=include_distance, max_eccentricity=max_eccentricity)),
        "seed": seed,
        "template_count": template_count,
        "noisy": False,
        "simulator_provenance": dependency_provenance("selflensing", "rustlens", "jaxoplanet"),
        "grid_preset": grid_preset,
        "grid_size_before_filters": proposal_draws,
        "grid_size_after_filters": len(systems),
    }
    return (
        np.vstack(flux_rows),
        relative_time_days,
        np.asarray(parameter_rows, dtype=PARAMETER_FIELDS),
        metadata,
    )


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


def _build_validation_lightcurves_uncached(
    *,
    sample_count=1000,
    seed,
    cadence_minutes=DEFAULT_TEMPLATE_CADENCE_MINUTES,
    half_window_days=None,
    window_scale=1.2,
    include_distance=True,
    include_self_lensing=True,
    include_ellipsoidal_variations=True,
    include_doppler_boosting=True,
    show_progress=False,
    max_eccentricity=DEFAULT_MAX_ECCENTRICITY,
    allow_rlof=False,
    min_flux=0.0,
    generation_batch_size=DEFAULT_GENERATION_BATCH_SIZE,
    ranges=None,
):
    generation_batch_size = _normalise_workers(
        generation_batch_size,
        "generation_batch_size",
    )
    systems = _sample_template_systems(
        count=sample_count,
        seed=seed,
        include_distance=include_distance,
        max_eccentricity=max_eccentricity,
        allow_rlof=allow_rlof,
        ranges=ranges,
    )
    if half_window_days is None:
        max_tau_eff = max(derived["tau_eff"] for _, _, derived in systems)
        half_window_days = max_tau_eff * window_scale

    cadence_days = cadence_minutes / (24 * 60)
    if not np.isfinite([cadence_days, half_window_days]).all() or min(cadence_days, half_window_days) <= 0:
        raise ValueError("Cadence and template half-window must be positive and finite")
    half_samples = int(np.ceil(half_window_days / cadence_days))
    relative_time_days = np.arange(-half_samples, half_samples + 1, dtype=float) * cadence_days
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
        "generation_interface": "selflensing.batch.batched_light_curve_jit",
        "shared_generator": generation.generator_metadata(ranges=_default_ranges(ranges,
            include_distance=include_distance, max_eccentricity=max_eccentricity)),
        "simulator_provenance": dependency_provenance("selflensing", "rustlens", "jaxoplanet"),
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
    cadence_minutes=DEFAULT_TEMPLATE_CADENCE_MINUTES,
    half_window_days=None,
    window_scale=1.2,
    include_distance=True,
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
    ranges=None,
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
        # Package versions alone do not distinguish the radius-conversion fix.
        "simulator_provenance": dependency_provenance("selflensing", "rustlens", "jaxoplanet"),
        "cache_version": 5,
        "seed": seed,
        "shared_generator": generation.generator_metadata(ranges=_default_ranges(ranges,
            include_distance=include_distance, max_eccentricity=max_eccentricity)),
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
        ranges=ranges,
    )
    metadata = {
        **metadata,
        "cache_hit": False,
        "simulator_provenance": cache_settings["simulator_provenance"],
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
    cadence_minutes=DEFAULT_TEMPLATE_CADENCE_MINUTES,
    half_window_days=None,
    window_scale=1.2,
    include_distance=True,
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
    peak_tolerance_samples=2,
    use_validation_cache=True,
    validation_cache_dir=DEFAULT_VALIDATION_CACHE_DIR,
    validation_backend="auto",
    validation_batch_size=32,
    validation_template_batch_size=512,
    generation_batch_size=DEFAULT_GENERATION_BATCH_SIZE,
    template_count=DEFAULT_TEMPLATE_COUNT,
    template_seed=generation.DEFAULT_SEED + 1,
    ranges=None,
):
    if include_distance is None:
        include_distance = True
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
            ranges=ranges,
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

    if not threshold_grid:
        raise ValueError("At least one calibration threshold setting is required")
    flux, time, parameters, metadata = build_template_library(
        cadence_minutes=cadence_minutes,
        half_window_days=half_window_days,
        window_scale=window_scale,
        include_distance=include_distance,
        max_templates=max_templates,
        include_self_lensing=include_self_lensing,
        include_ellipsoidal_variations=include_ellipsoidal_variations,
        include_doppler_boosting=include_doppler_boosting,
        show_progress=show_progress,
        max_eccentricity=max_eccentricity,
        ranges=ranges,
        allow_rlof=allow_rlof,
        min_flux=min_flux,
        generation_batch_size=generation_batch_size,
        grid_preset="default",
        template_count=template_count,
        seed=template_seed,
    )
    _status(f"Generated {flux.shape[0]} templates; evaluating {len(threshold_grid)} threshold settings.",
            enabled=show_progress)
    evaluations = []
    best = None
    for index, (prominence_sigma, min_score_sigma) in enumerate(threshold_grid, start=1):
        _status(f"Threshold step {index}/{len(threshold_grid)}: "
                f"prominence_sigma={prominence_sigma}, min_score_sigma={min_score_sigma}",
                enabled=show_progress)
        evaluation = evaluate_template_completeness(
            validation_flux=validation_flux, validation_time=validation_time,
            template_flux=flux, template_time=time,
            prominence_sigma=prominence_sigma, min_score_sigma=min_score_sigma,
            peak_tolerance_samples=peak_tolerance_samples,
            show_progress=show_progress, print_recall=show_progress,
            workers=cpu_workers, backend=resolved_validation_backend,
            batch_size=validation_batch_size, template_batch_size=validation_template_batch_size,
        )
        evaluation.update(grid_preset="default", templates=int(flux.shape[0]))
        evaluations.append(evaluation)
        _status(f"Completed threshold step {index}/{len(threshold_grid)}: "
                f"recall={100 * evaluation['recall']:.2f}% "
                f"({evaluation['matched']}/{evaluation['total']} matched)", enabled=show_progress)
        if best is None or evaluation["recall"] > best["recall"]:
            best = evaluation
        if evaluation["recall"] >= target_recall:
            _status(f"Target recall reached: {100 * evaluation['recall']:.2f}%", enabled=show_progress)
            break
    else:
        _status(f"Target recall was not reached. Best recall: {100 * best['recall']:.2f}%",
                enabled=show_progress)
    metadata["calibration"] = {
        "target_recall": target_recall, "validation": validation_metadata,
        "selected": best, "evaluations": evaluations,
    }
    return flux, time, parameters, metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate self-lensing matched-filter templates."
    )
    parser.add_argument("--version", type=config.validate_version, required=True,
                        help="Template version to write, e.g. 2.0")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing template version")
    parser.add_argument(
        "--grid-preset",
        choices=("default",),
        default="default",
        help="Generate random physical-parameter draws with the default preset.",
    )
    parser.add_argument(
        "--calibrate-completeness",
        action="store_true",
        help="Empirically choose thresholds on validation curves.",
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
        "--peak-tolerance-samples",
        type=int,
        default=2,
        help="Allowed offset from the injected event center when scoring recall.",
    )
    parser.add_argument(
        "--cadence-minutes",
        type=float,
        default=DEFAULT_TEMPLATE_CADENCE_MINUTES,
        help=f"Shared template cadence in minutes (default: {DEFAULT_TEMPLATE_CADENCE_MINUTES:g}).",
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
        default=True,
        help="Sample distance over its configured range (enabled by default).",
    )
    parser.add_argument("--fixed-distance", dest="include_distance", action="store_false",
                        help="Fix distance at 1 kpc (otherwise sample the default distance range)")
    parser.add_argument("--template-count", type=int, default=DEFAULT_TEMPLATE_COUNT,
                        help=f"Number of templates (default: {DEFAULT_TEMPLATE_COUNT:,}).")
    parser.add_argument("--template-seed", type=int, default=generation.DEFAULT_SEED + 1)
    parser.add_argument("--ranges-json", type=Path, help="Custom physical ranges for the default proposal")
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
    if config.template_library_path(args.version).exists() and not args.overwrite:
        raise FileExistsError(f"Template version {args.version} already exists; select a new version")
    cpu_workers = args.cpu_workers if args.cpu_workers is not None else args.workers
    if args.calibrate_completeness:
        flux, time, parameters, metadata = calibrate_template_library(
            target_recall=args.target_recall,
            validation_size=args.validation_size,
            validation_seed=args.validation_seed,
            template_count=getattr(args, "template_count", DEFAULT_TEMPLATE_COUNT),
            template_seed=getattr(args, "template_seed", generation.DEFAULT_SEED + 1),
            ranges=json.loads(args.ranges_json.read_text()) if getattr(args, "ranges_json", None) else None,
            cadence_minutes=args.cadence_minutes,
            half_window_days=args.half_window_days,
            window_scale=args.window_scale,
            include_distance=args.include_distance,
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
            cadence_minutes=args.cadence_minutes,
            half_window_days=args.half_window_days,
            window_scale=args.window_scale,
            include_distance=args.include_distance,
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
            template_count=getattr(args, "template_count", DEFAULT_TEMPLATE_COUNT),
            seed=getattr(args, "template_seed", generation.DEFAULT_SEED + 1),
            ranges=json.loads(args.ranges_json.read_text()) if getattr(args, "ranges_json", None) else None,
        )
    output = save_template_library(
        args.version,
        overwrite=args.overwrite,
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
