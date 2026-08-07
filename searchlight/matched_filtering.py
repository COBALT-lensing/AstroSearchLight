"""Matched-filter searches over a bank of light-curve templates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import signal

from .paths import DATA_PATH

DEFAULT_TEMPLATE_LIBRARY_PATH = DATA_PATH / "matched_filter_templates.npz"
DEFAULT_PROMINENCE_SIGMA = 3.0
DEFAULT_MIN_SCORE_SIGMA = 4.0
DEFAULT_MATCH_WINDOW_REL_HEIGHT = 0.5
_UNSET = object()


@dataclass(frozen=True)
class MatchedFilterTemplate:
    """A sampled template for matched-filter searches.

    The template is resampled from ``time`` onto the median cadence of the target
    light curve before filtering. ``event_half_width`` can be set to limit the
    searched and reported window around relative time zero.
    """

    flux: np.ndarray
    time: np.ndarray
    event_half_width: float | None = None


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
        event_half_widths = _template_event_half_widths(
            self.parameters,
            self.metadata,
        )
        for template_index, row in enumerate(self.flux):
            event_half_width = (
                None
                if event_half_widths is None
                else float(event_half_widths[template_index])
            )
            yield MatchedFilterTemplate(
                flux=row,
                time=self.time,
                event_half_width=event_half_width,
            )


@dataclass(frozen=True)
class _PreparedMatchedFilterTemplate:
    index: int
    flux: np.ndarray
    length: int
    center: int


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


def _as_float_matrix(values, name: str) -> np.ndarray:
    if hasattr(values, "value"):
        values = values.value

    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _template_event_half_widths(
    parameters: np.ndarray | None,
    metadata: dict | None,
) -> np.ndarray | None:
    if parameters is None or parameters.dtype.names is None:
        return None
    if "tau_eff" not in parameters.dtype.names:
        return None

    window_scale = float((metadata or {}).get("window_scale", 1.0))
    event_half_widths = np.asarray(parameters["tau_eff"], dtype=float) * window_scale
    if not np.all(np.isfinite(event_half_widths)):
        return None
    if np.any(event_half_widths <= 0):
        return None
    return event_half_widths


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
) -> tuple[np.ndarray, np.ndarray]:
    template_time_array = _as_float_array(template_time, f"{template_label}.time")
    if len(template_time_array) != len(template_flux):
        raise ValueError(
            f"{template_label}.time and {template_label}.flux differ in length"
        )

    template_cadence = _median_cadence(template_time_array, f"{template_label}.time")
    if np.isclose(template_cadence, target_cadence, rtol=1e-6, atol=0.0):
        return template_flux, template_time_array

    resampled_time = np.arange(
        template_time_array[0],
        template_time_array[-1] + (0.5 * target_cadence),
        target_cadence,
    )
    if len(resampled_time) < 2:
        raise ValueError(f"{template_label} is too short to resample")
    return (
        np.interp(resampled_time, template_time_array, template_flux),
        resampled_time,
    )


def _trim_template_to_event_window(
    template_flux: np.ndarray,
    template_time: np.ndarray,
    event_half_width: float | None,
    cadence: float,
) -> tuple[np.ndarray, np.ndarray]:
    if event_half_width is None:
        return template_flux, template_time

    center_index = int(np.argmin(np.abs(template_time)))
    half_samples = max(1, int(np.ceil(event_half_width / cadence)))
    start_index = max(0, center_index - half_samples)
    end_index = min(len(template_flux), center_index + half_samples + 1)
    return (
        template_flux[start_index:end_index],
        template_time[start_index:end_index],
    )


def _trim_template_edges_to_length(
    template_flux: np.ndarray,
    max_length: int,
) -> np.ndarray:
    if len(template_flux) <= max_length:
        return template_flux

    excess = len(template_flux) - max_length
    start = excess // 2
    return template_flux[start : start + max_length]


def _pad_search_signal_for_template(
    time: np.ndarray,
    flux: np.ndarray,
    template_length: int,
    cadence: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    if len(time) > template_length:
        return time, flux, 0

    pad_samples = template_length
    left_time = time[0] - cadence * np.arange(pad_samples, 0, -1)
    right_time = time[-1] + cadence * np.arange(1, pad_samples + 1)
    padded_time = np.concatenate([left_time, time, right_time])
    padded_flux = np.pad(
        flux,
        (pad_samples, pad_samples),
        mode="constant",
        constant_values=0.0,
    )
    return padded_time, padded_flux, pad_samples


def _pad_search_matrix_for_template(
    time: np.ndarray,
    flux: np.ndarray,
    template_length: int,
    cadence: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    if len(time) > template_length:
        return time, flux, 0

    pad_samples = template_length
    left_time = time[0] - cadence * np.arange(pad_samples, 0, -1)
    right_time = time[-1] + cadence * np.arange(1, pad_samples + 1)
    padded_time = np.concatenate([left_time, time, right_time])
    padded_flux = np.pad(
        flux,
        ((0, 0), (pad_samples, pad_samples)),
        mode="constant",
        constant_values=0.0,
    )
    return padded_time, padded_flux, pad_samples


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


def _resolve_threshold_sigmas(
    metadata: dict | None,
    prominence_sigma,
    min_score_sigma,
) -> tuple[float, float | None]:
    selected_calibration = (metadata or {}).get("calibration", {}).get("selected", {})

    if prominence_sigma is _UNSET:
        prominence_sigma = selected_calibration.get(
            "prominence_sigma",
            DEFAULT_PROMINENCE_SIGMA,
        )
    if min_score_sigma is _UNSET:
        min_score_sigma = selected_calibration.get(
            "min_score_sigma",
            DEFAULT_MIN_SCORE_SIGMA,
        )

    return float(prominence_sigma), (
        None if min_score_sigma is None else float(min_score_sigma)
    )


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


def _resolve_templates(
    templates: Iterable[MatchedFilterTemplate] | None,
    template_library_path,
) -> tuple[Iterable[MatchedFilterTemplate], dict | None]:
    template_metadata = getattr(templates, "metadata", None)
    if template_library_path is not None:
        if templates is not None:
            raise ValueError("Pass either templates or template_library_path, not both")
        templates = load_template_library(template_library_path)
        template_metadata = templates.metadata
    elif templates is None:
        templates = load_template_library()
        template_metadata = templates.metadata

    return templates, template_metadata


def _prepare_templates(
    templates: Iterable[MatchedFilterTemplate],
    target_cadence: float,
) -> list[_PreparedMatchedFilterTemplate]:
    prepared_templates = []
    for template_index, template in enumerate(templates):
        if not isinstance(template, MatchedFilterTemplate):
            raise TypeError("templates must contain MatchedFilterTemplate objects")
        raw_template_flux = _as_float_array(
            template.flux,
            f"templates[{template_index}].flux",
        )
        raw_template_flux, raw_template_time = _resample_template_to_cadence(
            raw_template_flux,
            template.time,
            target_cadence,
            f"templates[{template_index}]",
        )
        raw_template_flux, raw_template_time = _trim_template_to_event_window(
            raw_template_flux,
            raw_template_time,
            template.event_half_width,
            target_cadence,
        )
        template_len = len(raw_template_flux)
        template_flux = _normalise_signal(
            raw_template_flux,
            f"templates[{template_index}].flux",
        )
        prepared_templates.append(
            _PreparedMatchedFilterTemplate(
                index=template_index,
                flux=template_flux,
                length=template_len,
                center=int(np.argmin(np.abs(raw_template_time))),
            )
        )
    return prepared_templates


def _finalize_matches(
    matches: list[MatchedFilterMatch],
    *,
    dedupe_overlaps: bool,
    max_matches: int | None,
) -> list[MatchedFilterMatch]:
    if dedupe_overlaps:
        matches = _dedupe_overlapping_matches(matches)

    matches = sorted(matches, key=lambda match: match.score, reverse=True)
    if max_matches is not None:
        matches = matches[:max_matches]
    return sorted(matches, key=lambda match: match.peak_time)


def _time_at_fractional_index(
    time: np.ndarray,
    index: float,
) -> float:
    clipped_index = float(np.clip(index, 0, len(time) - 1))
    return float(np.interp(clipped_index, np.arange(len(time)), time))


def _match_window_from_correlation_width(
    detection_time: np.ndarray,
    *,
    left_ip: float,
    right_ip: float,
    template_center: int,
    pad_samples: int,
) -> tuple[int, int, float, float]:
    detection_start = left_ip + template_center
    detection_end = right_ip + template_center
    start_index = int(np.floor(detection_start)) - pad_samples
    end_index = int(np.ceil(detection_end)) - pad_samples
    start_time = _time_at_fractional_index(detection_time, detection_start)
    end_time = _time_at_fractional_index(detection_time, detection_end)
    return start_index, end_index, start_time, end_time


def _clip_match_window_to_observation(
    *,
    start_index: int,
    end_index: int,
    start_time: float,
    end_time: float,
    observation_time: np.ndarray,
) -> tuple[int, int, float, float]:
    return (
        max(start_index, 0),
        min(end_index, len(observation_time) - 1),
        max(start_time, float(observation_time[0])),
        min(end_time, float(observation_time[-1])),
    )


def _is_observed_peak(center_index: int, observation_length: int) -> bool:
    return 0 <= center_index < observation_length


def detect_matched_filter_peaks(
    time,
    flux,
    templates: Iterable[MatchedFilterTemplate] | None = None,
    *,
    template_library_path=None,
    prominence: float | None = None,
    prominence_sigma: float = _UNSET,
    height: float | None = None,
    min_score_sigma: float | None = _UNSET,
    distance: int | None = None,
    dedupe_overlaps: bool = True,
    max_matches: int | None = None,
    batch_size: int | None = None,
    match_window_rel_height: float = DEFAULT_MATCH_WINDOW_REL_HEIGHT,
) -> list[MatchedFilterMatch] | list[list[MatchedFilterMatch]]:
    """Find template-like peaks in a light curve.

    The input flux and each template are mean-subtracted. Templates with their
    own time grid are resampled to the median light-curve cadence. Templates are
    then scaled to unit energy before a valid cross-correlation is computed.
    Peaks in each correlation trace are proposed with
    :func:`scipy.signal.find_peaks`.

    Parameters
    ----------
    time, flux : array-like
        One-dimensional light-curve time and flux arrays, or a shared
        one-dimensional time array and a two-dimensional flux array with one
        light curve per row. ``astropy.time.Time`` and quantity-like arrays are
        accepted.
    templates : iterable, optional
        Template objects with flux and time arrays. Each template is resampled
        to the median cadence of the input light curve. If omitted, the default
        template library in ``~/.astrosearchlight`` is loaded. If a resampled
        template is at least as long as the light curve, the mean-subtracted
        search flux is zero-padded so the full template can slide across the
        observation.
    template_library_path : optional
        Path to a template library saved by :func:`save_template_library`.
    prominence : float, optional
        Direct prominence threshold for ``find_peaks``. If omitted, a robust
        per-template threshold is estimated from the matched-filter output.
    prominence_sigma : float
        Robust-sigma multiplier used when ``prominence`` is omitted. If omitted
        and the template library contains calibrated thresholds, the calibrated
        value is used.
    height : optional
        Direct matched-filter score threshold passed to
        ``scipy.signal.find_peaks``.
    min_score_sigma : float, optional
        Robust-sigma score threshold used when ``height`` is omitted. Set to
        ``None`` to disable the score threshold. If omitted and the template
        library contains calibrated thresholds, the calibrated value is used.
    distance : optional
        Passed through to ``scipy.signal.find_peaks``.
    dedupe_overlaps : bool
        If true, overlapping matches across the template bank are collapsed to
        the highest-scoring match.
    max_matches : int, optional
        If provided, return only the highest-scoring matches across all
        templates.
    batch_size : int, optional
        Number of light curves per correlation batch when ``flux`` is
        two-dimensional. Ignored for one-dimensional flux.
    match_window_rel_height : float
        Relative height passed to ``scipy.signal.peak_widths`` for the reported
        match window. The default reports the matched-filter full width at half
        prominence around the peak.
    """

    templates, template_metadata = _resolve_templates(templates, template_library_path)
    if match_window_rel_height < 0:
        raise ValueError("match_window_rel_height must be non-negative")

    prominence_sigma, min_score_sigma = _resolve_threshold_sigmas(
        template_metadata,
        prominence_sigma,
        min_score_sigma,
    )

    time_array = _as_float_array(time, "time")
    flux_array = np.asarray(flux)
    if flux_array.ndim == 2:
        return detect_matched_filter_peaks_batched(
            time_array,
            flux_array,
            templates,
            prominence=prominence,
            prominence_sigma=prominence_sigma,
            height=height,
            min_score_sigma=min_score_sigma,
            distance=distance,
            dedupe_overlaps=dedupe_overlaps,
            max_matches=max_matches,
            batch_size=batch_size,
            match_window_rel_height=match_window_rel_height,
        )

    flux_array = _as_float_array(flux_array, "flux")
    if len(time_array) != len(flux_array):
        raise ValueError("time and flux must have the same length")
    target_cadence = _median_cadence(time_array, "time")

    search_flux = flux_array - np.mean(flux_array)
    prepared_templates = _prepare_templates(templates, target_cadence)
    matches: list[MatchedFilterMatch] = []

    for template in prepared_templates:
        detection_time, detection_flux, pad_samples = _pad_search_signal_for_template(
            time_array,
            search_flux,
            template.length,
            target_cadence,
        )

        matched_filter_output = signal.correlate(
            detection_flux,
            template.flux,
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
        if len(peak_indices):
            _, _, left_ips, right_ips = signal.peak_widths(
                matched_filter_output,
                peak_indices,
                rel_height=match_window_rel_height,
            )
        else:
            left_ips = right_ips = np.asarray([], dtype=float)

        for peak_index, peak_prominence_value, left_ip, right_ip in zip(
            peak_indices,
            prominences,
            left_ips,
            right_ips,
        ):
            detection_start_index = int(peak_index)
            detection_center_index = detection_start_index + template.center
            center_index = detection_center_index - pad_samples
            if not _is_observed_peak(center_index, len(time_array)):
                continue
            start_index, end_index, start_time, end_time = (
                _match_window_from_correlation_width(
                    detection_time,
                    left_ip=float(left_ip),
                    right_ip=float(right_ip),
                    template_center=template.center,
                    pad_samples=pad_samples,
                )
            )
            start_index, end_index, start_time, end_time = (
                _clip_match_window_to_observation(
                    start_index=start_index,
                    end_index=end_index,
                    start_time=start_time,
                    end_time=end_time,
                    observation_time=time_array,
                )
            )
            matches.append(
                MatchedFilterMatch(
                    template_index=template.index,
                    peak_index=center_index,
                    start_index=start_index,
                    end_index=end_index,
                    peak_time=float(detection_time[detection_center_index]),
                    start_time=start_time,
                    end_time=end_time,
                    score=float(matched_filter_output[peak_index]),
                    prominence=float(peak_prominence_value),
                )
            )

    return _finalize_matches(
        matches,
        dedupe_overlaps=dedupe_overlaps,
        max_matches=max_matches,
    )


def detect_matched_filter_peaks_batched(
    time,
    flux,
    templates: Iterable[MatchedFilterTemplate] | None = None,
    *,
    template_library_path=None,
    prominence: float | None = None,
    prominence_sigma: float = _UNSET,
    height: float | None = None,
    min_score_sigma: float | None = _UNSET,
    distance: int | None = None,
    dedupe_overlaps: bool = True,
    max_matches: int | None = None,
    batch_size: int | None = None,
    match_window_rel_height: float = DEFAULT_MATCH_WINDOW_REL_HEIGHT,
) -> list[list[MatchedFilterMatch]]:
    """Find template-like peaks for many light curves sharing one time grid.

    ``flux`` must be shaped ``(n_light_curves, n_times)``. The returned value is
    a list with one ``MatchedFilterMatch`` list per input light curve. Template
    resampling and normalization are performed once, then each template is
    correlated against the flux batch. Set ``batch_size`` to limit peak memory
    when searching very large batches.
    """

    templates, template_metadata = _resolve_templates(templates, template_library_path)
    if match_window_rel_height < 0:
        raise ValueError("match_window_rel_height must be non-negative")
    prominence_sigma, min_score_sigma = _resolve_threshold_sigmas(
        template_metadata,
        prominence_sigma,
        min_score_sigma,
    )

    time_array = _as_float_array(time, "time")
    flux_matrix = _as_float_matrix(flux, "flux")
    if flux_matrix.shape[1] != len(time_array):
        raise ValueError("flux rows must have the same length as time")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive")
    target_cadence = _median_cadence(time_array, "time")

    search_flux = flux_matrix - np.mean(flux_matrix, axis=1, keepdims=True)
    prepared_templates = _prepare_templates(templates, target_cadence)
    matches_by_curve: list[list[MatchedFilterMatch]] = [
        [] for _ in range(search_flux.shape[0])
    ]
    if batch_size is None:
        batch_size = search_flux.shape[0]

    for template in prepared_templates:
        for batch_start in range(0, search_flux.shape[0], batch_size):
            batch_end = min(batch_start + batch_size, search_flux.shape[0])
            detection_time, detection_flux, pad_samples = (
                _pad_search_matrix_for_template(
                    time_array,
                    search_flux[batch_start:batch_end],
                    template.length,
                    target_cadence,
                )
            )
            matched_filter_outputs = signal.correlate(
                detection_flux,
                template.flux[np.newaxis, :],
                mode="valid",
                method="auto",
            )

            for batch_curve_index, matched_filter_output in enumerate(
                matched_filter_outputs
            ):
                curve_index = batch_start + batch_curve_index
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
                if len(peak_indices):
                    _, _, left_ips, right_ips = signal.peak_widths(
                        matched_filter_output,
                        peak_indices,
                        rel_height=match_window_rel_height,
                    )
                else:
                    left_ips = right_ips = np.asarray([], dtype=float)

                for peak_index, peak_prominence_value, left_ip, right_ip in zip(
                    peak_indices,
                    prominences,
                    left_ips,
                    right_ips,
                ):
                    detection_start_index = int(peak_index)
                    detection_center_index = detection_start_index + template.center
                    center_index = detection_center_index - pad_samples
                    if not _is_observed_peak(center_index, len(time_array)):
                        continue
                    start_index, end_index, start_time, end_time = (
                        _match_window_from_correlation_width(
                            detection_time,
                            left_ip=float(left_ip),
                            right_ip=float(right_ip),
                            template_center=template.center,
                            pad_samples=pad_samples,
                        )
                    )
                    start_index, end_index, start_time, end_time = (
                        _clip_match_window_to_observation(
                            start_index=start_index,
                            end_index=end_index,
                            start_time=start_time,
                            end_time=end_time,
                            observation_time=time_array,
                        )
                    )
                    matches_by_curve[curve_index].append(
                        MatchedFilterMatch(
                            template_index=template.index,
                            peak_index=center_index,
                            start_index=start_index,
                            end_index=end_index,
                            peak_time=float(detection_time[detection_center_index]),
                            start_time=start_time,
                            end_time=end_time,
                            score=float(matched_filter_output[peak_index]),
                            prominence=float(peak_prominence_value),
                        )
                    )

    return [
        _finalize_matches(
            matches,
            dedupe_overlaps=dedupe_overlaps,
            max_matches=max_matches,
        )
        for matches in matches_by_curve
    ]
