"""Matched-filter searches over a bank of light-curve templates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import signal

DEFAULT_TEMPLATE_LIBRARY_PATH = Path("data/matched_filter_templates.npz")
DEFAULT_PROMINENCE_SIGMA = 3.0
DEFAULT_MIN_SCORE_SIGMA = 4.0


@dataclass(frozen=True)
class MatchedFilterTemplate:
    """A sampled template for matched-filter searches.

    The template is resampled from ``time`` onto the median cadence of the target
    light curve before filtering.
    """

    flux: np.ndarray
    time: np.ndarray


@dataclass(frozen=True)
class MatchedFilterMatch:
    """A matched-filter detection mapped back onto the input light curve."""

    template_index: int
    peak_index: int
    start_index: int
    end_index: int
    peak_time: float
    start_time: float
    end_time: float
    score: float
    prominence: float


@dataclass(frozen=True)
class MatchedFilterTemplateLibrary:
    """A bank of templates stored as arrays with a shared relative time grid."""

    flux: np.ndarray
    time: np.ndarray
    parameters: np.ndarray | None = None
    metadata: dict | None = None

    def __post_init__(self):
        flux = np.asarray(self.flux, dtype=float)
        time = np.asarray(self.time, dtype=float)
        if flux.ndim != 2:
            raise ValueError("template library flux must be a two-dimensional array")
        if time.ndim != 1:
            raise ValueError("template library time must be a one-dimensional array")
        if flux.shape[1] != len(time):
            raise ValueError("template library flux rows must match the time length")
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "time", time)

    def __len__(self):
        return self.flux.shape[0]

    def __iter__(self):
        for row in self.flux:
            yield MatchedFilterTemplate(flux=row, time=self.time)


def save_template_library(
    path=DEFAULT_TEMPLATE_LIBRARY_PATH,
    *,
    flux,
    time,
    parameters=None,
    metadata=None,
):
    """Save a template library as a single NumPy ``.npz`` file."""

    library = MatchedFilterTemplateLibrary(
        flux=flux,
        time=time,
        parameters=parameters,
        metadata=metadata,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        "flux": library.flux,
        "time": library.time,
        "metadata_json": np.asarray(json.dumps(library.metadata or {})),
    }
    if parameters is not None:
        arrays["parameters"] = np.asarray(parameters)

    np.savez(path, **arrays)
    return path


def load_template_library(path=DEFAULT_TEMPLATE_LIBRARY_PATH):
    """Load a template library saved by :func:`save_template_library`."""

    with np.load(Path(path), allow_pickle=False) as data:
        metadata_json = str(data["metadata_json"]) if "metadata_json" in data else "{}"
        parameters = data["parameters"] if "parameters" in data else None
        return MatchedFilterTemplateLibrary(
            flux=data["flux"],
            time=data["time"],
            parameters=parameters,
            metadata=json.loads(metadata_json),
        )


def _as_float_array(values, name: str) -> np.ndarray:
    if hasattr(values, "jd"):
        values = values.jd
    elif hasattr(values, "value"):
        values = values.value

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _normalise_signal(values: np.ndarray, name: str) -> np.ndarray:
    centred = values - np.mean(values)
    energy = np.sqrt(np.sum(centred**2))
    if energy <= 0:
        raise ValueError(f"{name} has zero energy after mean subtraction")
    return centred / energy


def _median_cadence(time: np.ndarray, name: str) -> float:
    diffs = np.diff(time)
    if not np.all(diffs > 0):
        raise ValueError(f"{name} must be strictly increasing")
    return float(np.median(diffs))


def _resample_template_to_cadence(
    template_flux: np.ndarray,
    template_time,
    target_cadence: float,
    template_label: str,
) -> np.ndarray:
    template_time_array = _as_float_array(template_time, f"{template_label}.time")
    if len(template_time_array) != len(template_flux):
        raise ValueError(
            f"{template_label}.time and {template_label}.flux differ in length"
        )

    template_cadence = _median_cadence(template_time_array, f"{template_label}.time")
    if np.isclose(template_cadence, target_cadence, rtol=1e-6, atol=0.0):
        return template_flux

    resampled_time = np.arange(
        template_time_array[0],
        template_time_array[-1] + (0.5 * target_cadence),
        target_cadence,
    )
    if len(resampled_time) < 2:
        raise ValueError(f"{template_label} is too short to resample")
    return np.interp(resampled_time, template_time_array, template_flux)


def _trim_template_edges_to_length(
    template_flux: np.ndarray,
    max_length: int,
) -> np.ndarray:
    if len(template_flux) <= max_length:
        return template_flux

    excess = len(template_flux) - max_length
    start = excess // 2
    return template_flux[start : start + max_length]


def _robust_scatter(values: np.ndarray) -> float:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad > 0:
        return 1.4826 * mad
    std = np.std(values)
    return float(std) if std > 0 else 1.0


def _default_prominence(values: np.ndarray, sigma: float) -> float:
    return sigma * _robust_scatter(values)


def _default_height(values: np.ndarray, sigma: float) -> float:
    return float(np.median(values) + sigma * _robust_scatter(values))


def _dedupe_overlapping_matches(
    matches: list[MatchedFilterMatch],
) -> list[MatchedFilterMatch]:
    if not matches:
        return []

    sorted_matches = sorted(matches, key=lambda match: match.start_index)
    deduped: list[MatchedFilterMatch] = []
    current_group = [sorted_matches[0]]
    current_end = sorted_matches[0].end_index

    for match in sorted_matches[1:]:
        if match.start_index <= current_end:
            current_group.append(match)
            current_end = max(current_end, match.end_index)
            continue

        deduped.append(max(current_group, key=lambda match: match.score))
        current_group = [match]
        current_end = match.end_index

    deduped.append(max(current_group, key=lambda match: match.score))
    return sorted(deduped, key=lambda match: match.peak_time)


def detect_matched_filter_peaks(
    time,
    flux,
    templates: Iterable[MatchedFilterTemplate] | None = None,
    *,
    template_library_path=None,
    prominence: float | None = None,
    prominence_sigma: float = DEFAULT_PROMINENCE_SIGMA,
    height: float | None = None,
    min_score_sigma: float | None = DEFAULT_MIN_SCORE_SIGMA,
    distance: int | None = None,
    dedupe_overlaps: bool = True,
    max_matches: int | None = None,
) -> list[MatchedFilterMatch]:
    """Find template-like peaks in a light curve.

    The input flux and each template are mean-subtracted. Templates with their
    own time grid are resampled to the median light-curve cadence. Templates are
    then scaled to unit energy before a valid cross-correlation is computed.
    Peaks in each correlation trace are proposed with
    :func:`scipy.signal.find_peaks`.

    Parameters
    ----------
    time, flux : array-like
        One-dimensional light-curve time and flux arrays. ``astropy.time.Time``
        and quantity-like arrays are accepted.
    templates : iterable, optional
        Template objects with flux and time arrays. Each template is resampled
        to the median cadence of the input light curve. If omitted, the default
        template library in ``data`` is loaded. If a resampled template is
        longer than the light curve, its edges are trimmed symmetrically.
    template_library_path : optional
        Path to a template library saved by :func:`save_template_library`.
    prominence : float, optional
        Direct prominence threshold for ``find_peaks``. If omitted, a robust
        per-template threshold is estimated from the matched-filter output.
    prominence_sigma : float
        Robust-sigma multiplier used when ``prominence`` is omitted.
    height : optional
        Direct matched-filter score threshold passed to
        ``scipy.signal.find_peaks``.
    min_score_sigma : float, optional
        Robust-sigma score threshold used when ``height`` is omitted. Set to
        ``None`` to disable the default score threshold.
    distance : optional
        Passed through to ``scipy.signal.find_peaks``.
    dedupe_overlaps : bool
        If true, overlapping matches across the template bank are collapsed to
        the highest-scoring match.
    max_matches : int, optional
        If provided, return only the highest-scoring matches across all
        templates.
    """

    if template_library_path is not None:
        if templates is not None:
            raise ValueError("Pass either templates or template_library_path, not both")
        templates = load_template_library(template_library_path)
    elif templates is None:
        templates = load_template_library()

    time_array = _as_float_array(time, "time")
    flux_array = _as_float_array(flux, "flux")
    if len(time_array) != len(flux_array):
        raise ValueError("time and flux must have the same length")
    target_cadence = _median_cadence(time_array, "time")

    search_flux = flux_array - np.mean(flux_array)
    matches: list[MatchedFilterMatch] = []

    for template_index, template in enumerate(templates):
        if not isinstance(template, MatchedFilterTemplate):
            raise TypeError("templates must contain MatchedFilterTemplate objects")
        raw_template_flux = _as_float_array(
            template.flux,
            f"templates[{template_index}].flux",
        )
        raw_template_flux = _resample_template_to_cadence(
            raw_template_flux,
            template.time,
            target_cadence,
            f"templates[{template_index}]",
        )
        raw_template_flux = _trim_template_edges_to_length(
            raw_template_flux,
            len(search_flux),
        )
        template_len = len(raw_template_flux)
        template_flux = _normalise_signal(
            raw_template_flux,
            f"templates[{template_index}].flux",
        )

        matched_filter_output = signal.correlate(
            search_flux,
            template_flux,
            mode="valid",
            method="auto",
        )
        peak_prominence = (
            prominence
            if prominence is not None
            else _default_prominence(matched_filter_output, prominence_sigma)
        )
        peak_height = (
            height
            if height is not None or min_score_sigma is None
            else _default_height(matched_filter_output, min_score_sigma)
        )
        peak_indices, properties = signal.find_peaks(
            matched_filter_output,
            prominence=peak_prominence,
            height=peak_height,
            distance=distance,
        )
        prominences = properties.get("prominences", np.zeros_like(peak_indices))
        template_center = template_len // 2

        for peak_index, peak_prominence_value in zip(peak_indices, prominences):
            start_index = int(peak_index)
            end_index = start_index + template_len - 1
            center_index = start_index + template_center
            matches.append(
                MatchedFilterMatch(
                    template_index=template_index,
                    peak_index=center_index,
                    start_index=start_index,
                    end_index=end_index,
                    peak_time=float(time_array[center_index]),
                    start_time=float(time_array[start_index]),
                    end_time=float(time_array[end_index]),
                    score=float(matched_filter_output[peak_index]),
                    prominence=float(peak_prominence_value),
                )
            )

    if dedupe_overlaps:
        matches = _dedupe_overlapping_matches(matches)

    matches = sorted(matches, key=lambda match: match.score, reverse=True)
    if max_matches is not None:
        matches = matches[:max_matches]
    return sorted(matches, key=lambda match: match.peak_time)
