"""Exposure-integrated injections before preprocessing, with paired controls."""

from dataclasses import replace
import hashlib
import importlib.metadata
from pathlib import Path

import numpy as np

from .datasets import EvaluationManifest
from .preprocessing import LightCurve, PreprocessingConfig, read_lightcurve
from .provenance import dependency_provenance


def integrate_profile(time, exposure_seconds, profile, *, nodes=15):
    """Gauss–Legendre exposure average of fractional signal at actual timestamps."""
    time, exposure = np.asarray(time, float), np.asarray(exposure_seconds, float)
    if time.ndim != 1 or exposure.shape != time.shape or nodes < 2:
        raise ValueError("Invalid integration grid or quadrature order")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(exposure)) or np.any(exposure <= 0):
        raise ValueError("Injection times/exposures must be finite and positive")
    offsets, weights = np.polynomial.legendre.leggauss(nodes)
    sampled = time[:, None] + exposure[:, None] / 172800 * offsets
    values = np.asarray(profile(sampled.ravel()), float).reshape(sampled.shape)
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite injected signal")
    return values @ (weights / 2)


def inject_signal(curve, profile, *, dilution=1., nodes=15):
    if not np.isfinite(dilution) or not 0 <= dilution <= 1:
        raise ValueError("dilution is the source fraction in [0, 1]")
    finite = np.isfinite(curve.time) & np.isfinite(curve.flux) & np.isfinite(curve.exposure_seconds) & (curve.exposure_seconds > 0)
    if not finite.any():
        raise ValueError("No usable cadences for injection")
    baseline = float(np.median(curve.flux[finite]))
    if baseline <= 0:
        raise ValueError("Injection requires positive background flux")
    injected = np.array(curve.flux, copy=True)
    injected[finite] += baseline * dilution * integrate_profile(
        curve.time[finite], curve.exposure_seconds[finite], profile, nodes=nodes)
    metadata = {**curve.metadata, "injection_flux_convention": "add median(background) * dilution * fractional_signal",
                "injection_error_convention": "retain measured uncertainties; deterministic flux injection",
                "integration_nodes": nodes, "dilution": dilution}
    return LightCurve(curve.time.copy(), injected, curve.flux_err.copy(), curve.quality.copy(),
                      curve.exposure_seconds.copy(), metadata)


def sample_physical_system(rng, *, ranges=None, max_attempts=10000, proposal="evaluation"):
    """Independent continuous physical draws, conditional on lensing and no RLOF.

    Orientation is sampled uniformly in cos(i) in [0, 0.02], then rejected by
    physical constraints. This is a completeness proposal, not a population prior.
    """
    if proposal != "evaluation":
        from .proposals import sample_rv_system
        return sample_rv_system(rng, ranges=ranges, max_attempts=max_attempts, proposal=proposal)
    try:
        from selflensing.models import SelfLensingSystem
        from astropy import units as u
    except ImportError as exc:
        raise ImportError("Physical injection requires the template-generation extra") from exc
    bounds = {"a": (0.1, 5.), "Macc": (3., 30.), "Mcomp": (0.7, 3.),
              "e": (0., 0.7), "cos_i": (0., 0.02), "periapsis_phase": (0., 2 * np.pi)}
    bounds.update(ranges or {})
    for name, pair in bounds.items():
        if len(pair) != 2 or not np.all(np.isfinite(pair)) or pair[0] >= pair[1]:
            raise ValueError(f"Invalid physical range: {name}")
        if name in {"a", "Macc", "Mcomp"} and pair[0] <= 0:
            raise ValueError(f"Physical range must be positive: {name}")
    if not 0 <= bounds["e"][0] < bounds["e"][1] < 1 or not 0 <= bounds["cos_i"][0] < bounds["cos_i"][1] <= 1:
        raise ValueError("Eccentricity and orientation ranges exceed physical bounds")
    for _ in range(max_attempts):
        params = {name: float(np.exp(rng.uniform(np.log(low), np.log(high))))
                  if name in {"a", "Macc", "Mcomp"} else float(rng.uniform(low, high))
                  for name, (low, high) in bounds.items()}
        params["i"] = float(np.arccos(params.pop("cos_i")))
        system = SelfLensingSystem(a=params["a"] * u.au, i=params["i"] * u.rad,
            Macc=params["Macc"] * u.M_sun, Mcomp=params["Mcomp"] * u.M_sun,
            e=params["e"] * u.dimensionless_unscaled, periapsis_phase=params["periapsis_phase"] * u.rad, d=1 * u.kpc)
        if not bool(system.sl_possible) or bool(system.rlof):
            continue
        for name, value, unit in (("Rcomp", system.Rcomp, u.R_sun), ("Teff", system.Teff, u.K),
                                  ("period_days", system.porb, u.day), ("duration_days", system.tau_eff, u.day)):
            params[name] = float(np.asarray(value.to(unit).value))
        if all(np.isfinite(value) for value in params.values()) and params["duration_days"] > 0:
            return params
    raise RuntimeError("Physical proposal failed to find a valid non-RLOF lensing system")


def magnification_tail_bound_ppm(source_offset, re_over_rstar):
    """Upper bound outside the disc using the nearest surface point.

    Positive brightness weights imply A_disc <= A_point((s-1)/q), s>1.
    Rationalise A_point-1 to avoid subtracting nearly equal large numbers.
    """
    separation, ratio = np.broadcast_arrays(np.asarray(source_offset, float),
                                             np.asarray(re_over_rstar, float))
    if not np.isfinite(ratio).all() or np.any(ratio <= 0):
        raise ValueError("Einstein/source radius ratio must be positive")
    bound = np.full(separation.shape, np.inf)
    outside = separation > 1
    u = (separation[outside]-1)/ratio[outside]
    denominator = u*np.sqrt(u*u+4)
    bound[outside] = 4e6/(denominator*(u*u+2+denominator))
    return bound


PHYSICAL_GEOMETRY = "eccentric-observer-v2"


def physical_profile(params, center, *, tail_cutoff_ppm=None):
    """Native finite-source profiles using the installed orbital geometry.

    Optional tail truncation is bounded at each sample's foreground RE.
    """
    import jax
    from jax import numpy as jnp
    from selflensing.lightcurves import physical_parameters
    from selflensing.orbits import (orbital_phase, orbital_period_days,
        impact_parameter_source_radii, source_offset_source_radii, einstein_radius_m)
    from selflensing.lensing import native_magnification_callback
    row = jnp.array([params[name] for name in ("a", "i", "Macc", "Mcomp", "Rcomp", "Teff", "e", "periapsis_phase")])
    a, inclination, mass, _, total_mass, radius, _, eccentricity, periapsis, sin_i = physical_parameters(row)
    period = orbital_period_days(a, total_mass)
    source_radius = float(radius)
    if tail_cutoff_ppm is not None and (not np.isfinite(tail_cutoff_ppm) or tail_cutoff_ppm <= 0):
        raise ValueError("Tail cutoff must be positive or None")
    impact = impact_parameter_source_radii(a, inclination, radius, eccentricity, periapsis)
    @jax.jit
    def geometry(time):
        phase = orbital_phase(time, period, eccentricity, periapsis)
        offset = source_offset_source_radii(phase, a, radius, eccentricity, periapsis, impact)
        return offset, einstein_radius_m(a, sin_i, mass, eccentricity, periapsis, phase)

    def profile(time):
        separation, radii = (np.asarray(x, float) for x in geometry(jnp.asarray(time)-center))
        evaluate = radii > 0
        if tail_cutoff_ppm is not None and evaluate.any():
            bounded = np.zeros(separation.shape, bool)
            bounded[evaluate] = magnification_tail_bound_ppm(
                separation[evaluate], radii[evaluate]/source_radius) > tail_cutoff_ppm
            evaluate &= bounded
        signal = np.zeros(separation.shape)
        if evaluate.any():
            signal[evaluate] = native_magnification_callback(
                separation[evaluate], source_radius, radii[evaluate])-1
        return signal
    return profile


def build_injections(manifest, output, *, seed=42, variants=1, kind="physical", ranges=None,
                     nodes=15, dilution=1., preprocessing=None, center_mode="uniform", events_per_curve=1,
                     proposal="evaluation", tail_cutoff_ppm=None):
    manifest.validate()
    if kind not in {"physical", "gaussian"} or variants < 1 or events_per_curve < 1:
        raise ValueError("Invalid injection kind/count")
    if center_mode not in {"uniform", "boundary", "gap"}:
        raise ValueError("center_mode must be uniform, boundary, or gap")
    settings = PreprocessingConfig(**(preprocessing or {})).validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    samples = []
    for original in manifest.samples:
        if original.flux_product is not None and original.flux_product != settings.flux_product:
            raise ValueError("Injection flux product differs from prepared manifest")
        curve = read_lightcurve(original.source, settings)
        raw_path = output / (hashlib.sha256(original.sample_id.encode()).hexdigest()[:20] + "-control.npz")
        curve.write(raw_path)
        samples.append(replace(original, source=str(raw_path.resolve()), parent_id=None))
        for variant in range(variants):
            variant_seed = int.from_bytes(hashlib.sha256(f"{seed}:{original.sample_id}:{variant}".encode()).digest()[:8], "big")
            rng = np.random.default_rng(variant_seed)
            finite_time = np.sort(curve.time[np.isfinite(curve.time)])
            cadence = float(np.nanmedian(curve.exposure_seconds)) / 86400
            def center_time():
                if center_mode == "boundary":
                    return float(rng.choice([finite_time[0], finite_time[-1]]))
                if center_mode == "gap":
                    index = int(np.argmax(np.diff(finite_time)))
                    return float((finite_time[index] + finite_time[index + 1]) / 2)
                return float(rng.uniform(finite_time[0], finite_time[-1]))
            events = []
            if kind == "physical":
                params = sample_physical_system(rng, ranges=ranges, proposal=proposal)
                center = center_time()
                profile = physical_profile(params, center, tail_cutoff_ppm=tail_cutoff_ppm)
                width, period = params["duration_days"], params["period_days"]
                low = int(np.ceil((finite_time[0] - width - center) / period))
                high = int(np.floor((finite_time[-1] + width - center) / period))
                for cycle in range(low, high + 1):
                    events.append({"time": center + cycle * period, "duration_days": width,
                                   "tolerance_days": max(2 * cadence, width), "system": params})
                amplitude = float(profile(np.array([center]))[0]) * dilution
                for event in events:
                    event["amplitude"] = amplitude
            else:
                params = {}
                for _ in range(events_per_curve):
                    width = float(rng.uniform(*(ranges or {}).get("duration_days", (0.005, 0.04))))
                    amplitude = float(rng.uniform(*(ranges or {}).get("amplitude", (0.001, 0.015))))
                    events.append({"time": center_time(), "duration_days": width,
                                   "tolerance_days": max(2 * cadence, width), "amplitude": amplitude * dilution})
                def profile(time):
                    return sum(event["amplitude"] / max(dilution, np.finfo(float).eps)
                               * np.exp(-0.5 * ((time - event["time"]) / event["duration_days"])**2) for event in events)
            injected = inject_signal(curve, profile, dilution=dilution, nodes=nodes)
            key = hashlib.sha256(f"{original.sample_id}:{variant_seed}".encode()).hexdigest()[:20]
            path = output / f"{key}.npz"
            injected.write(path)
            parameters = {**original.parameters, "events": events, "injection_kind": kind, "seed": variant_seed,
                          "physical_parameters": params, "dilution": dilution, "integration_nodes": nodes,
                          "tail_cutoff_ppm": tail_cutoff_ppm,
                          "geometry_model": PHYSICAL_GEOMETRY if kind == "physical" else None,
                          "duration_days": events[0]["duration_days"] if events else None,
                          "amplitude": events[0]["amplitude"] if events else None}
            if len(events) == 1:
                parameters.update(event_time=events[0]["time"], match_tolerance_days=events[0]["tolerance_days"])
            samples.append(replace(original, sample_id=f"{original.sample_id}:injection:{variant}",
                source=str(path.resolve()), suite="spoc_injection" if original.suite != "simulated" else "simulated",
                label="lensing", label_provenance=f"{kind} exposure-integrated injection truth",
                parent_id=original.sample_id, parameters=parameters))
    metadata = {**manifest.metadata, "time_unit": "day", "injection": {"seed": seed, "kind": kind,
        "variants": variants, "ranges": ranges, "nodes": nodes, "dilution": dilution,
        "center_mode": center_mode, "events_per_curve": events_per_curve, "proposal": proposal,
        "tail_cutoff_ppm": tail_cutoff_ppm,
        "upstream_scope": "Delivered light-curve injections do not measure upstream SPOC attenuation"}}
    if kind == "physical":
        from .proposals import sampling_distributions
        metadata["injection"]["sampling_distributions"] = sampling_distributions(proposal)
        metadata["injection"]["geometry_model"] = PHYSICAL_GEOMETRY
        metadata["injection"]["simulator_version"] = importlib.metadata.version("selflensing")
        metadata["injection"]["physics_interface"] = "source-radius offsets to rustlens finite-source magnification"
        metadata["injection"]["rustlens_version"] = importlib.metadata.version("rustlens")
        metadata["injection"]["simulator_provenance"] = dependency_provenance("selflensing", "rustlens", "jaxoplanet")
    result = EvaluationManifest(tuple(samples), metadata=metadata)
    result.write(output / "manifest.json")
    return result
