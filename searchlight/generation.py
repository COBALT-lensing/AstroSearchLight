"""Shared physical-parameter proposals and batched selflensing/TESSGP simulation.

Public physical units are AU, radians, solar masses, kpc and days. Imports of
selflensing/JAX are deferred so reading datasets does not require the simulator.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
from astropy.time import Time
from tqdm.auto import tqdm

from .evaluation.provenance import dependency_provenance
from . import config

GENERATOR_VERSION = "photometry-batch-v1"
DEFAULT_SEED = 20260825
DEFAULT_BATCH_SIZE = 64
DEFAULT_CADENCE_SECONDS = 600.0
DEFAULT_DURATION_DAYS = 27.0
DEFAULT_PEAK_JD = 2460000.0
# Keep default limits explicit so upstream changes cannot silently alter generated datasets.
DEFAULT_RANGES = {
    "a": (0.1, 200.0), "i": (0.0, np.pi / 2), "Macc": (0.1, 50.0),
    "Mcomp": (0.5, 50.0), "e": (0.0, 0.999999),
    "periapsis_phase": (0.0, 2 * np.pi), "d": (0.1, 100.0),
}
SAMPLING_DISTRIBUTIONS = {
    "a": "log_uniform", "i": "uniform_angle", "Macc": "log_uniform",
    "Mcomp": "uniform", "e": "uniform_log1p", "periapsis_phase": "uniform", "d": "uniform",
}
PARAMETER_NAMES = tuple(DEFAULT_RANGES)
DERIVED_NAMES = ("Porb", "b", "tau_eff", "Rcomp", "Teff")


def resolve_ranges(ranges=None):
    if set(ranges or {}) - set(DEFAULT_RANGES):
        raise ValueError("Unknown physical parameter range")
    result = {**DEFAULT_RANGES, **(ranges or {})}
    for name, pair in result.items():
        if len(pair) != 2 or not np.isfinite(pair).all() or pair[0] > pair[1]:
            raise ValueError(f"Invalid range for {name}")
        result[name] = tuple(float(value) for value in pair)
    if any(result[key][0] <= 0 for key in ("a", "Macc", "Mcomp", "d")):
        raise ValueError("Distances, masses and semi-major axes must be positive")
    if not 0 <= result["e"][0] <= result["e"][1] < 1:
        raise ValueError("Eccentricity must be in [0, 1)")
    if not 0 <= result["i"][0] <= result["i"][1] <= np.pi / 2:
        raise ValueError("Inclination must be in [0, pi/2]")
    return result


def sample_parameter_rows(rng, size, *, ranges=None):
    """Draw the default proposal before physical and detectability selection."""
    if size < 1:
        raise ValueError("Sample size must be positive")
    bounds = resolve_ranges(ranges)
    rows = {}
    for name, (low, high) in bounds.items():
        if name in {"a", "Macc"}:
            values = 10.0 ** rng.uniform(np.log10(low), np.log10(high), size=size)
        elif name == "e":
            values = 10.0 ** rng.uniform(np.log10(1 + low), np.log10(1 + high), size=size) - 1
        else:
            values = rng.uniform(low, high, size=size)
        rows[name] = values
    return rows


def select_rows(rows, indices):
    return {name: np.asarray(values)[indices] for name, values in rows.items()}


def _require_selflensing():
    try:
        from selflensing import batch
    except ImportError as exc:
        raise ImportError("Simulation requires py-self-lensing: uv sync --extra template-generation") from exc
    return batch


def evaluate_parameter_rows(rows, *, allow_rlof=False):
    """Use the pinned simulator's full eccentric geometry, not a*cos(i)."""
    _require_selflensing()
    matrix = np.column_stack([rows[name] for name in PARAMETER_NAMES])
    values = _filter_kernel(allow_rlof)(matrix)
    values = [np.asarray(value) for value in values]
    return values[0].astype(bool), dict(zip(DERIVED_NAMES, values[1:]))


def sample_systems(count, rng, *, ranges=None, allow_rlof=False, max_attempts=None, return_attempts=False):
    """Draw exactly count viable systems with bounded, vectorised rejection."""
    if count < 1:
        raise ValueError("System count must be positive")
    bounds = resolve_ranges(ranges)
    max_attempts = max(100000, count * 10000) if max_attempts is None else max_attempts
    kept, attempts, chunks = 0, 0, []
    while kept < count and attempts < max_attempts:
        size = min(max_attempts - attempts, 65536, max(1024, 4 * (count - kept)))
        candidates = sample_parameter_rows(rng, size, ranges=bounds)
        mask, derived = evaluate_parameter_rows(candidates, allow_rlof=allow_rlof)
        selected = select_rows({**candidates, **derived}, np.flatnonzero(mask)[:count - kept])
        kept += len(selected["a"])
        attempts += size
        chunks.append(selected)
    if kept < count:
        raise RuntimeError(f"Only found {kept}/{count} viable systems after {attempts} draws; revise ranges or max_attempts")
    result = {name: np.concatenate([chunk[name] for chunk in chunks]) for name in chunks[0]}
    return (result, attempts) if return_attempts else result


def parameter_matrix(rows):
    """Public batched selflensing order; derive stellar properties upstream."""
    from selflensing.gp.PhysicalRelations import effective_temperature_K, stellar_radius_Rsun
    mass = np.asarray(rows["Mcomp"])
    radius = rows["Rcomp"] if "Rcomp" in rows else stellar_radius_Rsun(mass)
    temperature = rows["Teff"] if "Teff" in rows else effective_temperature_K(mass)
    return np.column_stack([rows["a"], rows["i"], rows["Macc"], mass, radius,
                            temperature, rows["e"], rows["periapsis_phase"]])


def sector_times(*, cadence_seconds=DEFAULT_CADENCE_SECONDS, duration_days=DEFAULT_DURATION_DAYS):
    if not np.isfinite([cadence_seconds, duration_days]).all() or min(cadence_seconds, duration_days) <= 0:
        raise ValueError("Cadence and duration must be positive and finite")
    times = np.arange(0.0, duration_days, cadence_seconds / 86400.0) - duration_days / 2
    if len(times) < 2:
        raise ValueError("The observation window must contain at least two cadences")
    return times


@dataclass
class LightCurveBatch:
    flux: np.ndarray
    mean_flux: np.ndarray
    flux_err: np.ndarray
    lensing_amplitude: np.ndarray
    peak_times: np.ndarray
    gp_parameters: np.ndarray | None
    count_baselines: np.ndarray
    event_parameters: dict


class LightCurveGenerator:
    """Reuse module-level JITs and one TESSGP input factory across batches."""

    def __init__(self, *, seed=DEFAULT_SEED, cadence_seconds=DEFAULT_CADENCE_SECONDS,
                 device=None, noise_sampler=None):
        self.rng = np.random.default_rng(seed)
        self.cadence_seconds = cadence_seconds
        self.device = device
        self.noise_sampler = noise_sampler
        self._gp_inputs = None

    def generate(self, rows, times, *, peak_times=0.0, noisy=False,
                 sl=True, ev=True, db=True, mode="physical", measure_lensing=False):
        batch = _require_selflensing()
        import jax
        import jax.numpy as jnp
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Times must be a finite, increasing one-dimensional array")
        if mode not in {"physical", "flare", "inverted"}:
            raise ValueError("Unknown light-curve mode")
        params = jax.device_put(jnp.asarray(parameter_matrix(rows)), self.device)
        device_times = jax.device_put(jnp.asarray(times), self.device)
        peaks = np.broadcast_to(np.asarray(peak_times, float), (len(params),)).copy()
        mean = batch.batched_light_curve_jit(device_times, params,
            peak_times=jax.device_put(peaks, self.device), sl=sl, ev=ev, db=db)
        mean = np.asarray(mean).copy()
        amplitude = np.zeros(len(params))
        if measure_lensing and sl:
            sl_flux = batch.batched_light_curve_jit(device_times, params,
                peak_times=jax.device_put(peaks, self.device), sl=True, ev=False, db=False)
            amplitude = np.max(np.asarray(sl_flux), axis=1) - 1
        events = {}
        if mode == "inverted":
            mean = 2 - mean
        elif mode == "flare":
            # Exponential flare prescription for CNN confounders.
            events = {"amplitude": self.rng.uniform(0, 9, len(params)),
                      "decay_seconds": 10 ** self.rng.uniform(2, 6.5, len(params))}
            delta = (times[None, :] - peaks[:, None]) * 86400
            mean *= 1 + events["amplitude"][:, None] * np.exp(
                -np.maximum(delta, 0) / events["decay_seconds"][:, None]) * (delta >= 0)
        gp_params, counts = None, np.ones(len(params))
        errors = np.zeros_like(mean)
        flux = mean.copy()
        if noisy:
            if self._gp_inputs is None:
                self._gp_inputs = make_tess_gp_input_generator(self.rng,
                    cadence_seconds=self.cadence_seconds, noise_sampler=self.noise_sampler)
            gp_params, counts, noise = self._gp_inputs(rows)
            key = jax.random.PRNGKey(int(self.rng.integers(0, 2**32, dtype=np.uint32)))
            noisy_counts = batch.batched_gp_light_curve_jit(key, device_times,
                jax.device_put(mean * counts[:, None], self.device),
                jax.device_put(gp_params, self.device), kernel_type="GS",
                noise_measure=jax.device_put(noise, self.device), times_unit="day")
            flux = np.asarray(noisy_counts) / counts[:, None]
            errors = np.broadcast_to((noise / counts)[:, None], mean.shape).copy()
        return LightCurveBatch(flux, mean, errors, amplitude, peaks, gp_params, counts, events)


def generator_metadata(*, ranges=None):
    return {"generator": GENERATOR_VERSION, "ranges": resolve_ranges(ranges),
            "generator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "sampling_distributions": dict(SAMPLING_DISTRIBUTIONS),
            "geometry": "pinned-selflensing-eccentric-observer-frame",
            "deterministic_interface": "selflensing.batch.batched_light_curve_jit",
            "noise_interface": "selflensing.batch.batched_gp_light_curve_jit",
            "gp_kernel": "GS", "gp_prior": "sampled-v1",
            "simulator_provenance": dependency_provenance("selflensing", "rustlens", "jaxoplanet", "jax")}


@lru_cache(maxsize=2)
def _filter_kernel(allow_rlof):
    import jax
    import jax.numpy as jnp
    from selflensing.gp.PhysicalRelations import (
        effective_temperature_K,
        stellar_radius_Rsun,
    )
    from selflensing.orbits import (
        AU_M,
        M_SUN_KG,
        R_SUN_M,
        crossing_time_days,
        einstein_radius_m,
        impact_parameter_source_radii,
        orbital_period_days,
        orbital_separation_m,
    )

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

        r_e_m = einstein_radius_m(a_m, sin_i, macc_kg, e, periapsis_phase)
        projected = impact_parameter_source_radii(a_m, i, rcomp_m, e, periapsis_phase)*rcomp_m
        b = projected/jnp.where(r_e_m > 0, r_e_m, 1.)
        sl_possible = (r_e_m > 0) & (projected <= rcomp_m+r_e_m)

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

        tau_eff_days = crossing_time_days(a_m, i, macc_kg, total_mass_kg,
                                          rcomp_m, e, periapsis_phase)

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

    return jax.jit(jax.vmap(evaluate_one))



def make_tess_gp_input_generator(rng=None, *, cadence_seconds=DEFAULT_CADENCE_SECONDS, noise_sampler=None):
    """Build batched GS inputs with independently sampled stellar noise.

    ``TESSGP`` is convenient for individual systems but calls
    ``conv_mass_to_mag`` for each one.  That function reloads the instrument
    response and noise files every time.  Stage 0 has thousands of systems, so
    retain one ``FluxModel`` and vectorise the small amount of TESS-GP setup.
    The returned values retain TESSGP's count-space conventions exactly.
    """
    _require_selflensing()
    import jax
    import jax.numpy as jnp
    from selflensing.gp import FluxFinderSensitivity as flux_finder
    from selflensing.gp import PhysicalRelations as physical_relations
    from selflensing.gp import PriorsSamplers as gp_priors

    rng = np.random.default_rng() if rng is None else rng
    flux_model = flux_finder.FluxModel()

    @jax.jit
    def _generate(masses, distances, noise_draws):
        magnitudes, fractional_noise = jax.vmap(flux_model.mass_to_mag)(masses, distances)
        log_count_baselines = jnp.log10(
            flux_finder.counts_per_cadence_from_mag(
                magnitudes, cadence_s=cadence_seconds
            )
        )
        count_baselines = 10.0**log_count_baselines
        log_granulation_frequency = jnp.log10(
            jax.vmap(physical_relations.granulation_frequency_relation)(masses)
        )
        # Draws are in linear ppm; only convert after sampling.
        log_granulation_sigma = log_count_baselines + jnp.log10(noise_draws[:, 0] / 1e6)
        log_seismic_frequency = jnp.log10(
            jax.vmap(physical_relations.asteroseismic_frequency_relation)(masses)
        )
        log_seismic_sigma = log_count_baselines + jnp.log10(noise_draws[:, 2] / 1e6)
        parameters = jnp.stack(
            [
                log_granulation_frequency,
                log_granulation_sigma,
                log_seismic_frequency,
                noise_draws[:, 1],
                log_seismic_sigma,
            ],
            axis=1,
        )
        return parameters, count_baselines, fractional_noise * count_baselines

    def generate(parameter_rows):
        masses = jnp.asarray(parameter_rows["Mcomp"])
        distances = jnp.asarray(parameter_rows["d"])
        # A custom sampler may use measured component amplitudes or star-specific
        # priors. It receives the physical rows and returns [G ppm, Q, S ppm].
        if noise_sampler is None:
            size = len(masses)
            granulation_ppm = 1e6 * 10.0 ** np.asarray(
                gp_priors.sample_log_sigma(0.0, "G", rng, size=size)
            )
            quality = gp_priors.sample_quality_factor(rng, size=size)
            seismic_ppm = 1e6 * 10.0 ** np.asarray(
                gp_priors.sample_log_sigma(0.0, "S", rng, size=size)
            )
            draws = np.column_stack((granulation_ppm, quality, seismic_ppm))
        else:
            draws = np.asarray(noise_sampler(rng, parameter_rows), dtype=float)
        if (
            draws.shape != (len(masses), 3)
            or not np.all(np.isfinite(draws))
            or np.any(draws <= 0)
        ):
            raise ValueError("Noise sampler must return positive finite [G ppm, Q, S ppm] rows")
        values = _generate(masses, distances, jnp.asarray(draws))
        return tuple(np.asarray(value) for value in jax.device_get(values))

    return generate



def _write_curve(path, times, batch, rows, index, *, label, subtype, peak_jd):
    from astropy.io import fits
    from astropy.table import Table

    table = Table()
    table["time"] = Time(peak_jd + times, format="jd", scale="utc")
    table["flux"] = batch.flux[index]
    table["flux_err"] = batch.flux_err[index]
    hdu = fits.BinTableHDU(table)
    names = {"a": "A", "i": "I", "Macc": "MBH", "Mcomp": "MSTAR", "e": "E",
             "periapsis_phase": "PERI", "d": "D", "Rcomp": "RSTAR", "Teff": "TEFF",
             "Porb": "P", "b": "B", "tau_eff": "TAU_EFF"}
    for name, header in names.items():
        hdu.header[header] = float(rows[name][index])
    hdu.header["GENVER"] = GENERATOR_VERSION
    hdu.header["LABEL"] = label
    hdu.header["SUBTYPE"] = subtype
    hdu.header["PEAK_T"] = float(peak_jd + batch.peak_times[index])
    hdu.header["AMP_SL"] = float(batch.lensing_amplitude[index])
    hdu.header["NOISY"] = batch.gp_parameters is not None
    if batch.gp_parameters is not None:
        gp = batch.gp_parameters[index]
        counts = batch.count_baselines[index]
        hdu.header["GPKERNEL"] = "GS"
        hdu.header["GPPRIOR"] = "sampled-v1"
        hdu.header["GPPPMG"] = float(1e6 * 10 ** gp[1] / counts)
        hdu.header["GPQS"] = float(gp[3])
        hdu.header["GPPPMS"] = float(1e6 * 10 ** gp[4] / counts)
    if batch.event_parameters:
        hdu.header["FLAREAMP"] = float(batch.event_parameters["amplitude"][index])
        hdu.header["FLARETAU"] = float(batch.event_parameters["decay_seconds"][index])
    temporary = path.with_name(path.name + ".new.fits.gz")
    try:
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(temporary, overwrite=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def generate_cnn_dataset(version, *, count_per_class=10000, counts=None, seed=DEFAULT_SEED,
                         batch_size=DEFAULT_BATCH_SIZE, noisy=True, ranges=None,
                         cadence_seconds=DEFAULT_CADENCE_SECONDS, duration_days=DEFAULT_DURATION_DAYS,
                         detection_sigma=None, inverted_fraction=0.3, peak_jd=DEFAULT_PEAK_JD,
                         max_attempts=None, progress=True):
    """Write a new version of three-class FITS examples and generation provenance.

    Only lensing positives receive the amplitude/white-noise detectability
    cut (5 sigma when noisy; disabled by default for noiseless examples).
    Non-lensing controls include inverted physical signals; flares use the
    exponential flare prescription recorded in the dataset metadata.
    """
    version = config.resolve_version("dataset", version)
    output = config.dataset_directory(version)
    if output.exists():
        raise FileExistsError(f"Dataset version {version} already exists: {output}; select a new version")
    counts = dict(counts or {name: count_per_class for name in ("lensing", "flare", "not-lensing")})
    if set(counts) != {"lensing", "flare", "not-lensing"} or any(
        not isinstance(value, int) or value < 2 for value in counts.values()
    ):
        raise ValueError("Provide at least two examples for each of lensing, flare and not-lensing")
    if batch_size < 1 or not 0 <= inverted_fraction <= 1 or not np.isfinite(peak_jd):
        raise ValueError("Invalid batch size, inverted fraction or peak JD")
    detection_sigma = (5.0 if noisy else 0.0) if detection_sigma is None else detection_sigma
    if not np.isfinite(detection_sigma) or detection_sigma < 0 or (not noisy and detection_sigma > 0):
        raise ValueError("Detectability sigma must be nonnegative; noiseless data uses sigma=0")
    times = sector_times(cadence_seconds=cadence_seconds, duration_days=duration_days)
    bounds = resolve_ranges(ranges)
    max_attempts = max(100000, sum(counts.values()) * 10000) if max_attempts is None else max_attempts
    generator = LightCurveGenerator(seed=seed, cadence_seconds=cadence_seconds)
    metadata = {**generator_metadata(ranges=bounds), "dataset_version": version, "seed": seed, "batch_size": batch_size,
        "noisy": noisy, "cadence_seconds": cadence_seconds, "duration_days": duration_days,
        "peak_jd": peak_jd, "peak_placement": "uniform_in_sector", "counts": counts,
        "detection_sigma": detection_sigma, "inverted_fraction": inverted_fraction,
        "flare_prescription": {"relative_amplitude": [0, 9], "decay_log10_seconds": [2, 6.5]},
        "status": "generating", "files": []}
    output.mkdir(parents=True, exist_ok=False)
    metadata_path = output / "generation.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    for name in counts:
        (output / name).mkdir()
    inverted_count = int(counts["not-lensing"] * inverted_fraction)
    jobs = [("lensing", "physical", counts["lensing"]), ("flare", "flare", counts["flare"]),
            ("not-lensing", "physical", counts["not-lensing"] - inverted_count),
            ("not-lensing", "inverted", inverted_count)]
    written = {name: 0 for name in counts}
    attempts = 0
    with tqdm(total=sum(counts.values()), desc="CNN dataset", disable=not progress) as bar:
        for label, mode, required in jobs:
            kept = 0
            while kept < required:
                size = min(batch_size, required - kept)
                rows, draws = sample_systems(size, generator.rng, ranges=bounds,
                    max_attempts=max_attempts - attempts, return_attempts=True)
                attempts += draws
                peaks = generator.rng.uniform(-duration_days / 2, duration_days / 2, size)
                batch = generator.generate(rows, times, peak_times=peaks, noisy=noisy,
                    sl=(label == "lensing" or mode == "inverted"), mode=mode,
                    measure_lensing=(label == "lensing" or mode == "inverted"))
                valid = np.all(np.isfinite(batch.flux) & (batch.flux > 0), axis=1)
                valid &= np.all(np.isfinite(batch.flux_err) & (batch.flux_err >= 0), axis=1)
                valid &= np.all(np.isfinite(batch.mean_flux) & (batch.mean_flux > 0), axis=1)
                if label == "lensing":
                    valid &= batch.lensing_amplitude > 0
                    valid &= batch.lensing_amplitude >= detection_sigma * batch.flux_err[:, 0]
                if mode == "inverted":
                    valid &= batch.lensing_amplitude > 0
                for index in np.flatnonzero(valid):
                    path = output / label / f"simulation_{written[label]:06d}.fits.gz"
                    _write_curve(path, times, batch, rows, index, label=label, subtype=mode, peak_jd=peak_jd)
                    with path.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    metadata["files"].append({"path": str(path.relative_to(output)), "sha256": digest,
                                               "label": label, "subtype": mode})
                    written[label] += 1
                    kept += 1
                    bar.update(1)
    metadata.update(status="complete", proposal_draws=attempts, written_counts=written)
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, metadata_path)
    return metadata


def add_generation_arguments(parser):
    parser.add_argument("--count-per-class", type=int, default=10000)
    parser.add_argument("--generation-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--generation-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--noiseless", action="store_true", help="Disable TESSGP noise and the default detectability cut")
    parser.add_argument("--cadence-seconds", type=float, default=DEFAULT_CADENCE_SECONDS)
    parser.add_argument("--duration-days", type=float, default=DEFAULT_DURATION_DAYS)
    parser.add_argument("--detection-sigma", type=float, help="Positive selection threshold; defaults to 5 for noisy curves, 0 for noiseless")
    parser.add_argument("--ranges-json", type=Path, help="JSON mapping physical parameters to [lower, upper]")
    parser.add_argument("--inverted-fraction", type=float, default=0.3)
    parser.add_argument("--max-proposal-draws", type=int, help="Bound physical draws including all rejection")


def generate_from_args(version, args):
    return generate_cnn_dataset(version, count_per_class=args.count_per_class,
        seed=args.generation_seed, batch_size=args.generation_batch_size,
        noisy=not args.noiseless, cadence_seconds=args.cadence_seconds,
        duration_days=args.duration_days, detection_sigma=args.detection_sigma,
        inverted_fraction=args.inverted_fraction, max_attempts=args.max_proposal_draws,
        ranges=json.loads(args.ranges_json.read_text()) if args.ranges_json else None)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate three-class CNN data with batched self-lensing physics and TESSGP")
    parser.add_argument("--version", type=config.validate_version, required=True, help="New dataset version, e.g. 2.0")
    add_generation_arguments(parser)
    args = parser.parse_args(argv)
    metadata = generate_from_args(args.version, args)
    print(f"Generated {metadata['written_counts']} in {config.dataset_directory(args.version)}")


if __name__ == "__main__":
    main()
