"""Analytic engineering fixture, not a physical self-lensing simulator."""

from pathlib import Path

import numpy as np

from .datasets import EvaluationManifest, EvaluationSample
from ..matched_filtering import MatchedFilterTemplate


def demo_templates():
    time = np.arange(-60, 61) * 120 / 86400
    return [MatchedFilterTemplate(np.exp(-0.5 * (time / width)**2), time)
            for width in (0.006, 0.012, 0.024)]


def create_demo(directory, *, seed=42, tuning_groups=40, test_groups=80):
    """Write paired independent noise backgrounds and Gaussian injections."""
    if min(tuning_groups, test_groups) < 1:
        raise ValueError("Each split requires at least one group")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    time = np.arange(512) * 120 / 86400
    samples = []
    for split, count in (("tuning", tuning_groups), ("test", test_groups)):
        for index in range(count):
            group = f"{split}-{index:04d}"
            noise = float(rng.uniform(0.001, 0.004))
            background = 1 + rng.normal(0, noise, len(time))
            background += np.convolve(rng.normal(0, noise * 0.7, len(time)),
                                      np.ones(9) / 9, mode="same")
            # Include confounding flare-like and transit-like backgrounds.
            subtype = ("noise", "flare", "transit")[index % 3]
            location = rng.uniform(time[80], time[-80])
            if subtype == "flare":
                delta = time - location
                background += rng.uniform(0.003, 0.015) * np.exp(-np.maximum(delta, 0) / 0.012) * (delta >= 0)
            elif subtype == "transit":
                background -= 0.008 * np.exp(-0.5 * ((time - location) / 0.018)**2)
            center = float(rng.uniform(time[80], time[-80]))
            width = float(rng.uniform(0.005, 0.028))
            amplitude = float(rng.uniform(0.001, 0.015))
            injection = amplitude * np.exp(-0.5 * ((time - center) / width)**2)
            for injected in (False, True):
                sample_id = group + ("-injected" if injected else "-control")
                path = directory / f"{sample_id}.npz"
                flux = background + injection if injected else background
                np.savez_compressed(path, time=time, flux=flux)
                parameters = {"noise": noise, "background_subclass": subtype}
                if injected:
                    parameters.update(event_time=center, amplitude=amplitude,
                                      duration_days=width, match_tolerance_days=max(2 * 120 / 86400, width))
                samples.append(EvaluationSample(
                    sample_id=sample_id, source=str(path.resolve()), suite="simulated",
                    split=split, group_id=group, label="lensing" if injected else subtype,
                    label_provenance="analytic fixture truth, not physical self-lensing",
                    cadence_seconds=120, parent_id=group + "-control" if injected else None,
                    parameters=parameters,
                ))
    manifest = EvaluationManifest(tuple(samples), metadata={
        "seed": seed, "generator": "analytic-gaussian-v1", "time_unit": "day",
        "description": "Engineering demonstration only; no physical or survey completeness claims",
        "positive_class": "lensing", "tuning_groups": tuning_groups, "test_groups": test_groups,
        "template_parameters": [{"width_days": width} for width in (0.006, 0.012, 0.024)],
    })
    manifest.write(directory / "manifest.json")
    return manifest


def create_backgrounds(directory, *, seed=42, tuning_groups=20, test_groups=40,
                       points=2048, cadence_seconds=120., noise=0.001, gap_fraction=0.):
    """Synthetic controls for physical injection tests, independent of training data."""
    from .preprocessing import LightCurve
    if min(tuning_groups, test_groups) < 1 or points < 16 or noise <= 0 or not 0 <= gap_fraction < 0.8:
        raise ValueError("Invalid background generation settings")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    samples = []
    for split, count in (("tuning", tuning_groups), ("test", test_groups)):
        for index in range(count):
            time = 2457000 + np.arange(points) * cadence_seconds / 86400
            flux = 1 + rng.normal(0, noise, points)
            flux += np.convolve(rng.normal(0, noise, points), np.ones(11) / 11, mode="same")
            label = ("noise", "flare", "transit")[index % 3]
            center = rng.uniform(time[0], time[-1])
            if label == "flare":
                delta = time - center
                flux += 0.008 * np.exp(-np.maximum(delta, 0) / 0.03) * (delta >= 0)
            elif label == "transit":
                flux -= 0.01 * np.exp(-0.5 * ((time - center) / 0.04)**2)
            keep = np.ones(points, bool)
            if gap_fraction:
                start = int(points * 0.4)
                keep[start:start + int(points * gap_fraction)] = False
            sample_id = f"background-{seed}-{split}-{index:04d}"
            path = directory / f"{sample_id}.npz"
            LightCurve(time[keep], flux[keep], np.full(keep.sum(), noise),
                       np.zeros(keep.sum(), np.int64), np.full(keep.sum(), cadence_seconds),
                       {"time_unit": "day", "time_format": "jd", "generator": "white-plus-correlated-background-v1"}).write(path)
            samples.append(EvaluationSample(sample_id=sample_id, source=str(path.resolve()),
                suite="simulated", split=split, group_id=sample_id, label=label,
                label_provenance="synthetic background truth", cadence_seconds=cadence_seconds,
                parameters={"seed": seed, "noise": noise, "background_subclass": label}))
    result = EvaluationManifest(tuple(samples), metadata={"time_unit": "day", "seed": seed,
        "generator": "white-plus-correlated-background-v1", "points": points,
        "cadence_seconds": cadence_seconds, "gap_fraction": gap_fraction})
    result.write(directory / "manifest.json")
    return result


def _template_profile(job):
    from .injections import physical_profile
    system, time, tail_cutoff_ppm = job
    return 1 + physical_profile(system, 0., tail_cutoff_ppm=tail_cutoff_ppm)(time)


def create_physical_templates(path, *, seed=123, count=16, cadence_seconds=120., ranges=None, proposal="evaluation",
                              max_half_window_days=None, tail_cutoff_ppm=None, progress=False, workers=1):
    from .injections import physical_profile, sample_physical_system, PHYSICAL_GEOMETRY
    from .provenance import dependency_provenance
    from .proposals import sampling_distributions
    from ..matched_filtering import MatchedFilterTemplateLibrary, _save_template_library_file as save_template_library
    import importlib.metadata
    rng = np.random.default_rng(seed)
    systems = [sample_physical_system(rng, ranges=ranges, proposal=proposal) for _ in range(count)]
    if not systems:
        raise ValueError("At least one template required")
    half_window = max(min(system["duration_days"] * 4, system["period_days"] * .45) for system in systems)
    if max_half_window_days is not None:
        if not np.isfinite(max_half_window_days) or max_half_window_days <= 0:
            raise ValueError("Template half-window cap must be positive")
        half_window = min(half_window, max_half_window_days)
    time = np.arange(-half_window, half_window + cadence_seconds / 172800, cadence_seconds / 86400)
    from .parallel import process_map
    # Tiny banks finish sooner than spawning scientific Python interpreters.
    template_workers = workers if len(time) * len(systems) >= 50000 else 1
    flux = np.array(process_map(_template_profile, [(system, time, tail_cutoff_ppm) for system in systems],
                                workers=template_workers, desc='Generate templates', progress=progress))
    for system, row in zip(systems, flux):
        row[np.abs(time) > min(system["duration_days"] * 4, system["period_days"] * .45)] = 1.
    if not np.all(np.isfinite(flux)):
        raise ValueError("Nonfinite physical template flux")
    parameters = np.array([(system["duration_days"],) for system in systems], dtype=[("tau_eff", "f8")])
    library = MatchedFilterTemplateLibrary(flux=flux, time=time, parameters=parameters, metadata={
        "seed": seed, "generator": "independent-physical-template-proposal-v1", "systems": systems,
        "geometry_model": PHYSICAL_GEOMETRY,
        "proposal": proposal, "ranges": ranges,
        "sampling_distributions": sampling_distributions(proposal),
        "max_half_window_days": max_half_window_days,
        "tail_cutoff_ppm": tail_cutoff_ppm,
        "simulator_version": importlib.metadata.version("selflensing"), "cadence_seconds": cadence_seconds,
        "simulator_provenance": dependency_provenance("selflensing", "rustlens", "jaxoplanet"),
        "physics_interface": "source-radius offsets to rustlens finite-source magnification", "window_scale": 1.})
    save_template_library(path, flux=library.flux, time=library.time, parameters=parameters, metadata=library.metadata)
    return library
