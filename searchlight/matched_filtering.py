"""Matched-filter searches over a bank of light-curve templates."""

from __future__ import annotations

import json
import logging
import os
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import signal

from .paths import CACHE_PATH, DATA_PATH

DEFAULT_TEMPLATE_LIBRARY_PATH = DATA_PATH / "matched_filter_templates.npz"
DEFAULT_JAX_COMPILATION_CACHE_PATH = CACHE_PATH / "jax_compilation_cache"
logger = logging.getLogger(__name__)
DEFAULT_PROMINENCE_SIGMA = 3.0
DEFAULT_MIN_SCORE_SIGMA = 4.0
DEFAULT_MATCH_WINDOW_REL_HEIGHT = 0.5
DEFAULT_MIN_CENTER_FLUX_SIGMA = 4.0
DEFAULT_TEMPLATE_BATCH_SIZE = 512
_UNSET = object()
_CORRELATION_BACKENDS = {"auto", "cpu", "jax"}
_JAX_CORRELATE_VALID_TEMPLATE_MATRIX = None
_JAX_CORRELATE_VALID_MATRIX = None
_JAX_TEMPLATE_BANK_ENVELOPE = None
_JAX_TEMPLATE_BANK_ENVELOPE_MATRIX = None
_PREPARED_ENVELOPE_TEMPLATE_CACHE = {}
_JAX_COMPILATION_CACHE_CONFIGURED = False


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
    polarity_weights: np.ndarray | None
    polarity_sign: float
    length: int
    center: int
    event_half_width: float | None = None


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


def _pad_polarity_signal(
    flux: np.ndarray,
    pad_samples: int,
) -> np.ndarray:
    if pad_samples == 0:
        return flux
    return np.pad(
        flux,
        (pad_samples, pad_samples),
        mode="constant",
        constant_values=0.0,
    )


def _pad_polarity_matrix(
    flux: np.ndarray,
    pad_samples: int,
) -> np.ndarray:
    if pad_samples == 0:
        return flux
    return np.pad(
        flux,
        ((0, 0), (pad_samples, pad_samples)),
        mode="constant",
        constant_values=0.0,
    )


def _robust_scatter(values: np.ndarray) -> float:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad > 0:
        return 1.4826 * mad
    std = np.std(values)
    return float(std) if std > 0 else 1.0


def _robust_mad_scatter(values: np.ndarray) -> float:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return float(1.4826 * mad) if mad > 0 else 0.0


def _minimum_center_flux(values: np.ndarray) -> float:
    noise_floor = DEFAULT_MIN_CENTER_FLUX_SIGMA * _robust_mad_scatter(values)
    if len(values) >= 100:
        return noise_floor
    max_positive_excursion = float(np.max(values))
    if max_positive_excursion <= 0:
        return noise_floor
    return min(noise_floor, 0.5 * max_positive_excursion)


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


def _template_polarity(
    template_flux: np.ndarray,
    center: int,
) -> tuple[np.ndarray | None, float]:
    baseline = float(np.median(template_flux))
    center_delta = float(template_flux[center] - baseline)
    if center_delta == 0 or not np.isfinite(center_delta):
        return None, 0.0

    polarity_sign = float(np.sign(center_delta))
    signed_excess = polarity_sign * (template_flux - baseline)
    positive = signed_excess > 0
    if not positive[center]:
        return None, polarity_sign

    start = center
    while start > 0 and positive[start - 1]:
        start -= 1
    end = center + 1
    while end < len(template_flux) and positive[end]:
        end += 1

    weights = np.zeros_like(template_flux, dtype=float)
    weights[start:end] = signed_excess[start:end]
    if np.sum(weights) <= 0:
        return None, polarity_sign
    return weights, polarity_sign


def _prepare_templates(
    templates: Iterable[MatchedFilterTemplate],
    target_cadence: float,
    *,
    allow_negative_peaks: bool,
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
        template_center = int(np.argmin(np.abs(raw_template_time)))
        polarity_weights, polarity_sign = _template_polarity(
            raw_template_flux,
            template_center,
        )
        if not allow_negative_peaks and (
            polarity_sign <= 0 or polarity_weights is None
        ):
            continue
        template_flux = _normalise_signal(
            raw_template_flux,
            f"templates[{template_index}].flux",
        )
        prepared_templates.append(
            _PreparedMatchedFilterTemplate(
                index=template_index,
                flux=template_flux,
                polarity_weights=polarity_weights,
                polarity_sign=polarity_sign,
                length=template_len,
                center=template_center,
            )
        )
    return prepared_templates


def _prepare_envelope_templates(
    templates: Iterable[MatchedFilterTemplate],
    target_cadence: float,
    *,
    allow_negative_peaks: bool,
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
        template_len = len(raw_template_flux)
        template_center = int(np.argmin(np.abs(raw_template_time)))
        polarity_weights, polarity_sign = _template_polarity(
            raw_template_flux,
            template_center,
        )
        if not allow_negative_peaks and (
            polarity_sign <= 0 or polarity_weights is None
        ):
            continue
        template_flux = _normalise_signal(
            raw_template_flux,
            f"templates[{template_index}].flux",
        )
        prepared_templates.append(
            _PreparedMatchedFilterTemplate(
                index=template_index,
                flux=template_flux,
                polarity_weights=polarity_weights,
                polarity_sign=polarity_sign,
                length=template_len,
                center=template_center,
                event_half_width=template.event_half_width,
            )
        )
    return prepared_templates


def _prepared_envelope_template_cache_key(
    templates: Iterable[MatchedFilterTemplate],
    target_cadence: float,
    *,
    allow_negative_peaks: bool,
):
    try:
        template_ref = weakref.ref(templates)
    except TypeError:
        return None
    return (
        id(templates),
        round(float(target_cadence), 12),
        bool(allow_negative_peaks),
    ), template_ref


def _cached_prepare_envelope_templates(
    templates: Iterable[MatchedFilterTemplate],
    target_cadence: float,
    *,
    allow_negative_peaks: bool,
) -> list[_PreparedMatchedFilterTemplate]:
    cache_identity = _prepared_envelope_template_cache_key(
        templates,
        target_cadence,
        allow_negative_peaks=allow_negative_peaks,
    )
    if cache_identity is None:
        return _prepare_envelope_templates(
            templates,
            target_cadence,
            allow_negative_peaks=allow_negative_peaks,
        )

    cache_key, template_ref = cache_identity
    cached = _PREPARED_ENVELOPE_TEMPLATE_CACHE.get(cache_key)
    if cached is not None:
        cached_ref, prepared_templates = cached
        if cached_ref() is templates:
            return prepared_templates

    prepared_templates = _prepare_envelope_templates(
        templates,
        target_cadence,
        allow_negative_peaks=allow_negative_peaks,
    )
    _PREPARED_ENVELOPE_TEMPLATE_CACHE[cache_key] = (
        template_ref,
        prepared_templates,
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


def _passes_template_polarity(
    polarity_flux: np.ndarray,
    detection_start_index: int,
    template: _PreparedMatchedFilterTemplate,
    min_center_flux: float,
) -> bool:
    if template.polarity_weights is None:
        return True

    center_flux = polarity_flux[detection_start_index + template.center]
    if template.polarity_sign * center_flux < min_center_flux:
        return False

    segment = polarity_flux[
        detection_start_index : detection_start_index + template.length
    ]
    polarity_score = template.polarity_sign * float(
        np.dot(segment, template.polarity_weights)
    )
    return polarity_score > 0


def _configure_jax_compilation_cache(jax_module):
    global _JAX_COMPILATION_CACHE_CONFIGURED
    if _JAX_COMPILATION_CACHE_CONFIGURED:
        return

    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if cache_dir is None:
        cache_dir = str(DEFAULT_JAX_COMPILATION_CACHE_PATH)
        os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", cache_dir)

    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning(
            "Could not create JAX compilation cache directory %s: %s",
            cache_dir,
            exc,
        )
        _JAX_COMPILATION_CACHE_CONFIGURED = True
        return
    jax_module.config.update("jax_enable_compilation_cache", True)
    jax_module.config.update("jax_compilation_cache_dir", cache_dir)
    jax_module.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax_module.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    _JAX_COMPILATION_CACHE_CONFIGURED = True


def _jax_gpu_available() -> bool:
    try:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        import jax

        _configure_jax_compilation_cache(jax)
        return any(device.platform == "gpu" for device in jax.devices())
    except Exception:
        return False


def _resolve_backend(backend: str) -> str:
    if backend not in _CORRELATION_BACKENDS:
        raise ValueError("backend must be 'auto', 'cpu', or 'jax'")
    if backend == "auto":
        return "jax" if _jax_gpu_available() else "cpu"
    return backend


def _correlate_valid_1d(
    detection_flux: np.ndarray,
    template_flux: np.ndarray,
    *,
    backend: str,
) -> np.ndarray:
    if backend == "cpu":
        return signal.correlate(
            detection_flux,
            template_flux,
            mode="valid",
            method="auto",
        )

    template_flux = np.asarray(template_flux, dtype=float)
    return _correlate_valid_template_matrix(
        detection_flux,
        template_flux[np.newaxis, :],
        backend=backend,
    )[0]


def _jax_correlate_valid_template_matrix_device(
    detection_flux: np.ndarray,
    template_flux: np.ndarray,
):
    """Return a device-resident template-bank correlation matrix."""

    global _JAX_CORRELATE_VALID_TEMPLATE_MATRIX
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax

    _configure_jax_compilation_cache(jax)
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import lax

    if _JAX_CORRELATE_VALID_TEMPLATE_MATRIX is None:

        @jax.jit
        def correlate_valid_template_matrix(signal_1d, template_bank):
            lhs = signal_1d[None, :, None]
            rhs = jnp.swapaxes(template_bank, 0, 1)[:, None, :]
            result = lax.conv_general_dilated(
                lhs,
                rhs,
                window_strides=(1,),
                padding="VALID",
                dimension_numbers=("NWC", "WIO", "NWC"),
            )
            return result[0].T

        _JAX_CORRELATE_VALID_TEMPLATE_MATRIX = correlate_valid_template_matrix

    return _JAX_CORRELATE_VALID_TEMPLATE_MATRIX(
        jnp.asarray(detection_flux, dtype=jnp.float64),
        jnp.asarray(template_flux, dtype=jnp.float64),
    )


def _correlate_valid_template_matrix(
    detection_flux: np.ndarray,
    template_flux: np.ndarray,
    *,
    backend: str,
) -> np.ndarray:
    """Correlate one signal against a same-length bank of templates.

    Returns an array shaped ``(n_templates, n_valid_positions)``.
    """

    template_flux = np.asarray(template_flux, dtype=float)
    if template_flux.ndim != 2:
        raise ValueError("template_flux must be two-dimensional")
    if template_flux.shape[1] > len(detection_flux):
        raise ValueError("template rows cannot be longer than detection_flux")

    if backend == "cpu":
        return np.vstack(
            [
                signal.correlate(
                    detection_flux,
                    row,
                    mode="valid",
                    method="auto",
                )
                for row in template_flux
            ]
        )

    import jax

    _configure_jax_compilation_cache(jax)
    result = _jax_correlate_valid_template_matrix_device(
        detection_flux,
        template_flux,
    )
    return np.asarray(jax.device_get(result), dtype=float)


def _jax_template_bank_envelope(
    detection_flux: np.ndarray,
    template_flux: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return best score and in-batch template index at each valid offset."""

    global _JAX_TEMPLATE_BANK_ENVELOPE
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax

    _configure_jax_compilation_cache(jax)
    import jax.numpy as jnp
    from jax import lax

    if _JAX_TEMPLATE_BANK_ENVELOPE is None:

        @jax.jit
        def template_bank_envelope(signal_1d, template_bank):
            lhs = signal_1d[None, :, None]
            filters = jnp.swapaxes(template_bank, 0, 1)[:, None, :]
            scores = lax.conv_general_dilated(
                lhs,
                filters,
                window_strides=(1,),
                padding="VALID",
                dimension_numbers=("NWC", "WIO", "NWC"),
            )[0]
            return jnp.max(scores, axis=1), jnp.argmax(scores, axis=1)

        _JAX_TEMPLATE_BANK_ENVELOPE = template_bank_envelope

    scores, template_indices = _JAX_TEMPLATE_BANK_ENVELOPE(
        jnp.asarray(detection_flux, dtype=jnp.float32),
        jnp.asarray(template_flux, dtype=jnp.float32),
    )
    scores, template_indices = jax.device_get((scores, template_indices))
    return (
        np.asarray(scores, dtype=float),
        np.asarray(template_indices, dtype=int),
    )


def _jax_template_bank_envelope_matrix(
    detection_flux: np.ndarray,
    template_flux: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return best score and in-batch template index for each curve/offset."""

    global _JAX_TEMPLATE_BANK_ENVELOPE_MATRIX
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax

    _configure_jax_compilation_cache(jax)
    import jax.numpy as jnp
    from jax import lax

    if _JAX_TEMPLATE_BANK_ENVELOPE_MATRIX is None:

        @jax.jit
        def template_bank_envelope_matrix(signal_matrix, template_bank):
            lhs = signal_matrix[:, :, None]
            filters = jnp.swapaxes(template_bank, 0, 1)[:, None, :]
            scores = lax.conv_general_dilated(
                lhs,
                filters,
                window_strides=(1,),
                padding="VALID",
                dimension_numbers=("NWC", "WIO", "NWC"),
            )
            return jnp.max(scores, axis=2), jnp.argmax(scores, axis=2)

        _JAX_TEMPLATE_BANK_ENVELOPE_MATRIX = template_bank_envelope_matrix

    scores, template_indices = _JAX_TEMPLATE_BANK_ENVELOPE_MATRIX(
        jnp.asarray(detection_flux, dtype=jnp.float32),
        jnp.asarray(template_flux, dtype=jnp.float32),
    )
    scores, template_indices = jax.device_get((scores, template_indices))
    return (
        np.asarray(scores, dtype=float),
        np.asarray(template_indices, dtype=int),
    )


def _template_bank_envelope(
    detection_flux: np.ndarray,
    template_flux: np.ndarray,
    *,
    backend: str,
) -> tuple[np.ndarray, np.ndarray]:
    if backend == "jax":
        return _jax_template_bank_envelope(detection_flux, template_flux)

    matched_filter_outputs = _correlate_valid_template_matrix(
        detection_flux,
        template_flux,
        backend=backend,
    )
    return (
        np.max(matched_filter_outputs, axis=0),
        np.argmax(matched_filter_outputs, axis=0),
    )


def _template_bank_envelope_matrix(
    detection_flux: np.ndarray,
    template_flux: np.ndarray,
    *,
    backend: str,
) -> tuple[np.ndarray, np.ndarray]:
    if backend == "jax":
        return _jax_template_bank_envelope_matrix(detection_flux, template_flux)

    curve_scores = []
    curve_indices = []
    for curve_flux in detection_flux:
        scores, indices = _template_bank_envelope(
            curve_flux,
            template_flux,
            backend=backend,
        )
        curve_scores.append(scores)
        curve_indices.append(indices)
    return np.vstack(curve_scores), np.vstack(curve_indices)


def _correlate_valid_matrix(
    detection_flux: np.ndarray,
    template_flux: np.ndarray,
    *,
    backend: str,
) -> np.ndarray:
    if backend == "cpu":
        return signal.correlate(
            detection_flux,
            template_flux[np.newaxis, :],
            mode="valid",
            method="auto",
        )

    global _JAX_CORRELATE_VALID_MATRIX
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax

    _configure_jax_compilation_cache(jax)
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import lax

    if _JAX_CORRELATE_VALID_MATRIX is None:

        @jax.jit
        def correlate_valid_matrix(signal_matrix, template_1d):
            lhs = signal_matrix[:, :, None]
            rhs = template_1d[:, None, None]
            result = lax.conv_general_dilated(
                lhs,
                rhs,
                window_strides=(1,),
                padding="VALID",
                dimension_numbers=("NWC", "WIO", "NWC"),
            )
            return result[:, :, 0]

        _JAX_CORRELATE_VALID_MATRIX = correlate_valid_matrix

    result = _JAX_CORRELATE_VALID_MATRIX(
        jnp.asarray(detection_flux, dtype=jnp.float64),
        jnp.asarray(template_flux, dtype=jnp.float64),
    )
    return np.asarray(jax.device_get(result), dtype=float)


def _detect_matched_filter_peaks_envelope(
    time_array: np.ndarray,
    search_flux: np.ndarray,
    polarity_flux: np.ndarray,
    min_center_flux: float,
    prepared_templates: list[_PreparedMatchedFilterTemplate],
    *,
    target_cadence: float,
    prominence: float | None,
    prominence_sigma: float,
    height: float | None,
    min_score_sigma: float | None,
    distance: int | None,
    dedupe_overlaps: bool,
    max_matches: int | None,
    template_batch_size: int,
    backend: str,
) -> list[MatchedFilterMatch]:
    matches: list[MatchedFilterMatch] = []
    templates_by_length: dict[int, list[_PreparedMatchedFilterTemplate]] = {}
    for template in prepared_templates:
        templates_by_length.setdefault(template.length, []).append(template)

    for template_length, template_group in templates_by_length.items():
        detection_time, detection_flux, pad_samples = _pad_search_signal_for_template(
            time_array,
            search_flux,
            template_length,
            target_cadence,
        )
        detection_polarity_flux = _pad_polarity_signal(polarity_flux, pad_samples)
        output_len = len(detection_time) - template_length + 1
        if output_len < 1:
            continue

        envelope_scores = np.full(output_len, -np.inf, dtype=float)
        envelope_template_indices = np.full(output_len, -1, dtype=int)
        for batch_start in range(0, len(template_group), template_batch_size):
            batch_templates = template_group[
                batch_start : batch_start + template_batch_size
            ]
            template_flux = np.asarray(
                [template.flux for template in batch_templates],
                dtype=np.float32 if backend == "jax" else float,
            )
            batch_scores, batch_template_indices = _template_bank_envelope(
                detection_flux,
                template_flux,
                backend=backend,
            )
            improved = batch_scores > envelope_scores
            envelope_scores[improved] = batch_scores[improved]
            envelope_template_indices[improved] = (
                batch_start + batch_template_indices[improved]
            )

        finite_envelope = np.isfinite(envelope_scores)
        if not np.any(finite_envelope):
            continue

        envelope_for_threshold = envelope_scores[finite_envelope]
        peak_prominence = (
            prominence
            if prominence is not None
            else _default_prominence(envelope_for_threshold, prominence_sigma)
        )
        peak_height = (
            height
            if height is not None or min_score_sigma is None
            else _default_height(envelope_for_threshold, min_score_sigma)
        )
        peak_indices, properties = signal.find_peaks(
            envelope_scores,
            prominence=peak_prominence,
            height=peak_height,
            distance=distance,
        )
        if len(peak_indices) == 0:
            continue

        peak_prominences = properties.get(
            "prominences",
            np.full(len(peak_indices), np.nan),
        )
        for peak_index, peak_prominence_value in zip(
            peak_indices,
            peak_prominences,
        ):
            template_group_index = int(envelope_template_indices[peak_index])
            if template_group_index < 0:
                continue
            template = template_group[template_group_index]
            detection_center_index = int(peak_index) + template.center
            center_index = detection_center_index - pad_samples
            if not _is_observed_peak(center_index, len(time_array)):
                continue
            if (
                template.polarity_sign
                * detection_polarity_flux[detection_center_index]
                < min_center_flux
            ):
                continue

            if (
                template.event_half_width is not None
                and np.isfinite(template.event_half_width)
                and template.event_half_width > 0
            ):
                half_width = float(template.event_half_width)
            else:
                half_width = 0.5 * (template.length - 1) * target_cadence
            peak_time = float(detection_time[detection_center_index])
            start_time = max(float(time_array[0]), peak_time - half_width)
            end_time = min(float(time_array[-1]), peak_time + half_width)
            start_index = int(np.searchsorted(time_array, start_time, side="left"))
            end_index = int(np.searchsorted(time_array, end_time, side="right") - 1)

            matches.append(
                MatchedFilterMatch(
                    template_index=template.index,
                    peak_index=center_index,
                    start_index=max(0, start_index),
                    end_index=min(len(time_array) - 1, end_index),
                    peak_time=peak_time,
                    start_time=start_time,
                    end_time=end_time,
                    score=float(envelope_scores[peak_index]),
                    prominence=float(peak_prominence_value),
                )
            )

    return _finalize_matches(
        matches,
        dedupe_overlaps=dedupe_overlaps,
        max_matches=max_matches,
    )


def _detect_matched_filter_peaks_envelope_matrix(
    time_array: np.ndarray,
    search_flux: np.ndarray,
    polarity_flux: np.ndarray,
    min_center_flux: np.ndarray,
    prepared_templates: list[_PreparedMatchedFilterTemplate],
    *,
    target_cadence: float,
    prominence: float | None,
    prominence_sigma: float,
    height: float | None,
    min_score_sigma: float | None,
    distance: int | None,
    dedupe_overlaps: bool,
    max_matches: int | None,
    template_batch_size: int,
    backend: str,
    progress_callback=None,
) -> list[list[MatchedFilterMatch]]:
    curve_matches: list[list[MatchedFilterMatch]] = [
        [] for _curve_index in range(search_flux.shape[0])
    ]
    templates_by_length: dict[int, list[_PreparedMatchedFilterTemplate]] = {}
    for template in prepared_templates:
        templates_by_length.setdefault(template.length, []).append(template)

    total_template_batches = sum(
        (len(template_group) + template_batch_size - 1) // template_batch_size
        for template_group in templates_by_length.values()
    )
    completed_template_batches = 0
    filter_start_time = time.monotonic()
    for template_length, template_group in templates_by_length.items():
        detection_time, detection_flux, pad_samples = _pad_search_matrix_for_template(
            time_array,
            search_flux,
            template_length,
            target_cadence,
        )
        detection_polarity_flux = _pad_polarity_matrix(polarity_flux, pad_samples)
        output_len = len(detection_time) - template_length + 1
        if output_len < 1:
            continue

        envelope_scores = np.full(
            (search_flux.shape[0], output_len),
            -np.inf,
            dtype=float,
        )
        envelope_template_indices = np.full(
            (search_flux.shape[0], output_len),
            -1,
            dtype=int,
        )
        for batch_start in range(0, len(template_group), template_batch_size):
            batch_templates = template_group[
                batch_start : batch_start + template_batch_size
            ]
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "template_batch_start",
                        "completed_template_batches": completed_template_batches,
                        "total_template_batches": total_template_batches,
                        "template_length": template_length,
                        "template_batch_start": batch_start,
                        "template_batch_size": len(batch_templates),
                        "template_group_size": len(template_group),
                        "curve_count": search_flux.shape[0],
                        "sample_count": search_flux.shape[1],
                        "output_len": output_len,
                        "elapsed_seconds": time.monotonic() - filter_start_time,
                    }
                )
            template_batch_start_time = time.monotonic()
            template_flux = np.asarray(
                [template.flux for template in batch_templates],
                dtype=np.float32 if backend == "jax" else float,
            )
            batch_scores, batch_template_indices = _template_bank_envelope_matrix(
                detection_flux,
                template_flux,
                backend=backend,
            )
            improved = batch_scores > envelope_scores
            envelope_scores[improved] = batch_scores[improved]
            envelope_template_indices[improved] = (
                batch_start + batch_template_indices[improved]
            )
            completed_template_batches += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "template_batch_complete",
                        "completed_template_batches": completed_template_batches,
                        "total_template_batches": total_template_batches,
                        "template_length": template_length,
                        "template_batch_start": batch_start,
                        "template_batch_size": len(batch_templates),
                        "template_group_size": len(template_group),
                        "curve_count": search_flux.shape[0],
                        "sample_count": search_flux.shape[1],
                        "output_len": output_len,
                        "template_batch_seconds": (
                            time.monotonic() - template_batch_start_time
                        ),
                        "elapsed_seconds": time.monotonic() - filter_start_time,
                    }
                )

        for curve_index in range(search_flux.shape[0]):
            curve_envelope_scores = envelope_scores[curve_index]
            finite_envelope = np.isfinite(curve_envelope_scores)
            if not np.any(finite_envelope):
                continue

            envelope_for_threshold = curve_envelope_scores[finite_envelope]
            peak_prominence = (
                prominence
                if prominence is not None
                else _default_prominence(envelope_for_threshold, prominence_sigma)
            )
            peak_height = (
                height
                if height is not None or min_score_sigma is None
                else _default_height(envelope_for_threshold, min_score_sigma)
            )
            peak_indices, properties = signal.find_peaks(
                curve_envelope_scores,
                prominence=peak_prominence,
                height=peak_height,
                distance=distance,
            )
            if len(peak_indices) == 0:
                continue

            peak_prominences = properties.get(
                "prominences",
                np.full(len(peak_indices), np.nan),
            )
            for peak_index, peak_prominence_value in zip(
                peak_indices,
                peak_prominences,
            ):
                template_group_index = int(
                    envelope_template_indices[curve_index, peak_index]
                )
                if template_group_index < 0:
                    continue
                template = template_group[template_group_index]
                detection_center_index = int(peak_index) + template.center
                center_index = detection_center_index - pad_samples
                if not _is_observed_peak(center_index, len(time_array)):
                    continue
                if (
                    template.polarity_sign
                    * detection_polarity_flux[curve_index, detection_center_index]
                    < min_center_flux[curve_index]
                ):
                    continue

                if (
                    template.event_half_width is not None
                    and np.isfinite(template.event_half_width)
                    and template.event_half_width > 0
                ):
                    half_width = float(template.event_half_width)
                else:
                    half_width = 0.5 * (template.length - 1) * target_cadence
                peak_time = float(detection_time[detection_center_index])
                start_time = max(float(time_array[0]), peak_time - half_width)
                end_time = min(float(time_array[-1]), peak_time + half_width)
                start_index = int(np.searchsorted(time_array, start_time, side="left"))
                end_index = int(np.searchsorted(time_array, end_time, side="right") - 1)

                curve_matches[curve_index].append(
                    MatchedFilterMatch(
                        template_index=template.index,
                        peak_index=center_index,
                        start_index=max(0, start_index),
                        end_index=min(len(time_array) - 1, end_index),
                        peak_time=peak_time,
                        start_time=start_time,
                        end_time=end_time,
                        score=float(curve_envelope_scores[peak_index]),
                        prominence=float(peak_prominence_value),
                    )
                )

    return [
        _finalize_matches(
            matches,
            dedupe_overlaps=dedupe_overlaps,
            max_matches=max_matches,
        )
        for matches in curve_matches
    ]


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
    backend: str = "auto",
    match_window_rel_height: float = DEFAULT_MATCH_WINDOW_REL_HEIGHT,
    allow_negative_peaks: bool = False,
    progress_callback=None,
) -> list[MatchedFilterMatch] | list[list[MatchedFilterMatch]]:
    """Find template-like peaks in a light curve.

    The input flux and each template are mean-subtracted. Templates with their
    own time grid are resampled to the median light-curve cadence. Templates are
    then scaled to unit energy before a valid cross-correlation is computed.
    For one-dimensional inputs, the detector builds a best-template envelope:
    at each valid offset it keeps the highest correlation score across the
    template bank and the template that produced it. Peaks are then proposed
    once on that envelope with :func:`scipy.signal.find_peaks`.

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
        threshold is estimated from the matched-filter envelope for
        one-dimensional inputs.
    prominence_sigma : float
        Robust-sigma multiplier used when ``prominence`` is omitted. If omitted
        and the template library contains calibrated thresholds, the calibrated
        value is used.
    height : optional
        Direct matched-filter envelope score threshold passed to
        ``scipy.signal.find_peaks`` for one-dimensional inputs.
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
        Number of templates per envelope correlation batch when ``flux`` is
        one-dimensional, or number of light curves per correlation batch when
        ``flux`` is two-dimensional.
    backend : {"auto", "cpu", "jax"}
        Correlation backend. ``"auto"`` uses JAX when a GPU is visible to JAX
        and otherwise uses SciPy. ``"jax"`` forces JAX for the matched-filter
        envelope correlations.
    match_window_rel_height : float
        Relative height passed to ``scipy.signal.peak_widths`` by the
        two-dimensional exact detector. The one-dimensional envelope detector
        reports windows from template event widths when available, or the full
        template span otherwise.
    allow_negative_peaks : bool
        If false, only clearly positive-centered templates are searched and
        candidate centers must be significant positive flux excursions. Set true
        to also search negative-centered templates, allowing negative observed
        peaks with matching template polarity.
    """

    templates, template_metadata = _resolve_templates(templates, template_library_path)
    if match_window_rel_height < 0:
        raise ValueError("match_window_rel_height must be non-negative")

    prominence_sigma, min_score_sigma = _resolve_threshold_sigmas(
        template_metadata,
        prominence_sigma,
        min_score_sigma,
    )
    resolved_backend = _resolve_backend(backend)

    time_array = _as_float_array(time, "time")
    flux_array = np.asarray(flux)
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive")
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
            backend=resolved_backend,
            match_window_rel_height=match_window_rel_height,
            allow_negative_peaks=allow_negative_peaks,
            progress_callback=progress_callback,
        )

    flux_array = _as_float_array(flux_array, "flux")
    if len(time_array) != len(flux_array):
        raise ValueError("time and flux must have the same length")
    target_cadence = _median_cadence(time_array, "time")

    search_flux = flux_array - np.mean(flux_array)
    polarity_flux = flux_array - np.median(flux_array)
    min_center_flux = _minimum_center_flux(polarity_flux)
    prepared_templates = _prepare_envelope_templates(
        templates,
        target_cadence,
        allow_negative_peaks=allow_negative_peaks,
    )
    template_batch_size = batch_size or DEFAULT_TEMPLATE_BATCH_SIZE
    return _detect_matched_filter_peaks_envelope(
        time_array,
        search_flux,
        polarity_flux,
        min_center_flux,
        prepared_templates,
        target_cadence=target_cadence,
        prominence=prominence,
        prominence_sigma=prominence_sigma,
        height=height,
        min_score_sigma=min_score_sigma,
        distance=distance,
        dedupe_overlaps=dedupe_overlaps,
        max_matches=max_matches,
        template_batch_size=template_batch_size,
        backend=resolved_backend,
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
    backend: str = "auto",
    match_window_rel_height: float = DEFAULT_MATCH_WINDOW_REL_HEIGHT,
    allow_negative_peaks: bool = False,
    progress_callback=None,
) -> list[list[MatchedFilterMatch]]:
    """Find template-like peaks for many light curves sharing one time grid.

    ``flux`` must be shaped ``(n_light_curves, n_times)``. The returned value is
    a list with one ``MatchedFilterMatch`` list per input light curve. Template
    resampling and normalization are performed once, then each light curve is
    searched using the same best-template envelope detector as
    :func:`detect_matched_filter_peaks`. Set ``batch_size`` to cap the number of
    templates in each envelope correlation batch. ``backend`` chooses the
    correlation implementation; ``"auto"`` uses JAX only when a GPU is visible.
    If ``allow_negative_peaks`` is false, only clearly positive-centered
    templates are searched.
    """

    templates, template_metadata = _resolve_templates(templates, template_library_path)
    if match_window_rel_height < 0:
        raise ValueError("match_window_rel_height must be non-negative")
    prominence_sigma, min_score_sigma = _resolve_threshold_sigmas(
        template_metadata,
        prominence_sigma,
        min_score_sigma,
    )
    resolved_backend = _resolve_backend(backend)

    time_array = _as_float_array(time, "time")
    flux_matrix = _as_float_matrix(flux, "flux")
    if flux_matrix.shape[1] != len(time_array):
        raise ValueError("flux rows must have the same length as time")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive")
    target_cadence = _median_cadence(time_array, "time")

    search_flux = flux_matrix - np.mean(flux_matrix, axis=1, keepdims=True)
    polarity_flux = flux_matrix - np.median(flux_matrix, axis=1, keepdims=True)
    min_center_flux = np.apply_along_axis(_minimum_center_flux, 1, polarity_flux)
    prepared_templates = _cached_prepare_envelope_templates(
        templates,
        target_cadence,
        allow_negative_peaks=allow_negative_peaks,
    )
    template_batch_size = batch_size or DEFAULT_TEMPLATE_BATCH_SIZE

    return _detect_matched_filter_peaks_envelope_matrix(
        time_array,
        search_flux,
        polarity_flux,
        min_center_flux,
        prepared_templates,
        target_cadence=target_cadence,
        prominence=prominence,
        prominence_sigma=prominence_sigma,
        height=height,
        min_score_sigma=min_score_sigma,
        distance=distance,
        dedupe_overlaps=dedupe_overlaps,
        max_matches=max_matches,
        template_batch_size=template_batch_size,
        backend=resolved_backend,
        progress_callback=progress_callback,
    )
