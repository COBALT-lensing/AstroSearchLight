"""Traceable target-sector scores; no threshold selection or event recovery."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ClassificationScore:
    sample_id: str
    classes: tuple[str, ...]
    logits: tuple[float, ...]
    probabilities: tuple[float, ...]
    positive_class: str
    score: float


def cnn_score(sample_id, classes, logits, *, positive_class):
    """Convert raw logits using explicit checkpoint class order, stably."""
    classes = tuple(classes)
    values = np.asarray(logits, dtype=float)
    if not classes or len(set(classes)) != len(classes) or positive_class not in classes:
        raise ValueError("Unique classes and an explicit positive class are required")
    if values.shape != (len(classes),) or not np.all(np.isfinite(values)):
        raise ValueError("Expected one finite logit per class")
    weights = np.exp(values - values.max())
    probabilities = weights / weights.sum()
    return ClassificationScore(
        sample_id, classes, tuple(values.tolist()), tuple(probabilities.tolist()),
        positive_class, float(probabilities[classes.index(positive_class)]),
    )


def score_cnn_paths(model, samples, *, positive_class):
    """Adapt SimuLens without assuming class indices or collapsing duplicate paths.

    Callers must validate their manifest first. Model/checkpoint configuration
    belongs in run metadata; probabilities here are uncalibrated ranking scores.
    """
    samples = list(samples)
    if len({s.sample_id for s in samples}) != len(samples):
        raise ValueError("Duplicate sample IDs")
    if not samples:
        return []
    predictions = model.classify_paths(list(dict.fromkeys(s.source for s in samples)))
    missing = {s.source for s in samples} - predictions.keys()
    if missing:
        raise ValueError(f"CNN returned no prediction for: {sorted(missing)}")
    return [cnn_score(s.sample_id, model.classes, predictions[s.source][1:],
                      positive_class=positive_class) for s in samples]


@dataclass(frozen=True)
class MatchedFilterScore:
    sample_id: str
    score: float
    peak_time: float
    template_index: int
    score_kind: str = "maximum_raw_correlation_before_detection_cuts"


def score_matched_filter(sample_id, time, flux, templates, *, backend="cpu"):
    """Maximum correlation at observed centers, before peak/flux/polarity cuts.

    Uses production template resampling, mean subtraction, energy normalization,
    padding and correlation. Searches positive-centered templates, like the
    default detector. This is a raw correlation, not an SNR or event detection.
    Currently restricted to regular finite grids: gap-aware processing is a
    later preprocessing milestone. No eligible templates is an error, not a
    negative classification.
    """
    from .. import matched_filtering as mf

    times = mf._as_float_array(time, "time")
    values = mf._as_float_array(flux, "flux")
    if len(times) != len(values):
        raise ValueError("time and flux must have the same length")
    if len(times) < 2:
        raise ValueError("Scoring requires at least two observations")
    cadence = mf._median_cadence(times, "time")
    if not np.allclose(np.diff(times), cadence, rtol=1e-5, atol=1e-10):
        raise ValueError("Scoring requires a regular time grid; split or preprocess gaps")
    if backend == "cuda_fft":
        from .cuda_filter import engine
        return engine(templates).score(sample_id, times, values)
    prepared = mf._prepare_envelope_templates(templates, cadence, allow_negative_peaks=False)
    resolved_backend = mf._resolve_backend(backend)
    best = None
    for template in prepared:
        _, padded, padding = mf._pad_search_signal_for_template(
            times, values - values.mean(), template.length, cadence)
        scores = mf._correlate_valid_1d(padded, template.flux, backend=resolved_backend)
        centers = np.arange(len(scores)) + template.center - padding
        eligible = (centers >= 0) & (centers < len(times)) & np.isfinite(scores)
        indices = np.flatnonzero(eligible)
        if not len(indices):
            continue
        index = int(indices[np.argmax(scores[indices])])
        candidate = MatchedFilterScore(sample_id, float(scores[index]),
                                       float(times[centers[index]]), template.index)
        if best is None or candidate.score > best.score:
            best = candidate
    if best is None or not math.isfinite(best.score):
        raise ValueError("No eligible finite matched-filter score")
    return best


def matched_filter_envelope(time, flux, templates, *, backend="cpu", normalize=False):
    """Continuous best-template scores at observed centers before detection cuts.

    Returns arrays plus unthresholded interior local maxima with prominences.
    Uncovered edge cadences have NaN scores and winning index -1. No peak height,
    center-flux, distance, deduplication, or candidate-count cuts are applied.
    As in the default production detector, nonpositive templates are excluded.
    """
    from scipy.signal import find_peaks
    from .. import matched_filtering as mf
    from .preprocessing import noise_scale
    times, values = mf._as_float_array(time, "time"), mf._as_float_array(flux, "flux")
    if len(times) != len(values) or len(times) < 2:
        raise ValueError("Expected matching arrays with at least two observations")
    cadence = mf._median_cadence(times, "time")
    if not np.allclose(np.diff(times), cadence, rtol=1e-5, atol=1e-10):
        raise ValueError("Envelope scoring requires a regular time grid")
    prepared = mf._prepare_envelope_templates(templates, cadence, allow_negative_peaks=False)
    envelope, winner = np.full(len(times), -np.inf), np.full(len(times), -1, int)
    resolved_backend = mf._resolve_backend(backend)
    for template in prepared:
        _, padded, padding = mf._pad_search_signal_for_template(times, values - values.mean(), template.length, cadence)
        scores = mf._correlate_valid_1d(padded, template.flux, backend=resolved_backend)
        centers = np.arange(len(scores)) + template.center - padding
        valid = (centers >= 0) & (centers < len(times)) & np.isfinite(scores)
        centers, scores = centers[valid], scores[valid]
        improve = scores > envelope[centers]
        envelope[centers[improve]] = scores[improve]
        winner[centers[improve]] = template.index
    finite = np.flatnonzero(np.isfinite(envelope))
    if not len(finite):
        raise ValueError("No eligible finite matched-filter envelope")
    scale = noise_scale(values) if normalize else 1.
    envelope = envelope / scale
    candidates = []
    # Split finite runs rather than bridging any uncovered cadence.
    breaks = np.r_[0, np.flatnonzero(np.diff(finite) > 1) + 1, len(finite)]
    for start, stop in zip(breaks[:-1], breaks[1:]):
        indices = finite[start:stop]
        peaks, properties = find_peaks(envelope[indices], prominence=0)
        for peak, prominence in zip(peaks, properties["prominences"]):
            index = int(indices[peak])
            candidates.append({"peak_index": index, "peak_time": float(times[index]),
                               "score": float(envelope[index]), "prominence": float(prominence),
                               "template_index": int(winner[index])})
    envelope[~np.isfinite(envelope)] = np.nan
    return {"time": times, "score": envelope, "template_index": winner,
            "noise_scale": scale, "candidates": candidates}
