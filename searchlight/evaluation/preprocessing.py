"""Explicit local/SPOC loading, provenance and gap-preserving preprocessing."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter


@dataclass
class LightCurve:
    time: np.ndarray  # JD/TDB days for SPOC; days for synthetic inputs
    flux: np.ndarray
    flux_err: np.ndarray
    quality: np.ndarray
    exposure_seconds: np.ndarray
    metadata: dict = field(default_factory=dict)

    def validate(self):
        arrays = [np.asarray(getattr(self, name)) for name in
                  ("time", "flux", "flux_err", "quality", "exposure_seconds")]
        if any(array.ndim != 1 or array.shape != arrays[0].shape for array in arrays):
            raise ValueError("Light-curve columns must be equally sized one-dimensional arrays")
        if not len(arrays[0]):
            raise ValueError("Empty light curve")
        return self

    def subset(self, mask):
        return LightCurve(*(np.asarray(getattr(self, name))[mask] for name in
                            ("time", "flux", "flux_err", "quality", "exposure_seconds")),
                          metadata=dict(self.metadata))

    def write(self, path):
        self.validate()
        np.savez_compressed(path, **{name: np.asarray(getattr(self, name)) for name in
                                    ("time", "flux", "flux_err", "quality", "exposure_seconds")},
                            metadata=np.array(json.dumps(self.metadata, allow_nan=False)))


@dataclass(frozen=True)
class PreprocessingConfig:
    flux_product: str = "PDCSAP_FLUX"
    quality_bitmask: int = -1  # reject any flagged cadence; 0 retains all
    trim_start_hours: float = 0.0
    normalize: bool = True
    detrend_window_days: float | None = None
    gap_factor: float = 1.5
    min_segment_points: int = 3
    exposure_seconds: float | None = None

    def validate(self):
        if self.flux_product not in {"SAP_FLUX", "PDCSAP_FLUX"}:
            raise ValueError("flux_product must be SAP_FLUX or PDCSAP_FLUX")
        if not isinstance(self.quality_bitmask, int) or self.quality_bitmask < -1:
            raise ValueError("quality_bitmask must be -1 or a nonnegative integer")
        if not np.isfinite(self.trim_start_hours) or self.trim_start_hours < 0:
            raise ValueError("trim_start_hours must be nonnegative")
        if not np.isfinite(self.gap_factor) or self.gap_factor <= 1:
            raise ValueError("gap_factor must exceed one")
        if self.min_segment_points < 2:
            raise ValueError("min_segment_points must be at least two")
        for value in (self.detrend_window_days, self.exposure_seconds):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError("Window and exposure overrides must be finite and positive")
        return self


def read_lightcurve(path, config=None):
    explicit_config = config is not None
    config = (config or PreprocessingConfig()).validate()
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            time, flux = np.array(data["time"], float), np.array(data["flux"], float)
            positive = np.diff(np.sort(time[np.isfinite(time)]))
            positive = positive[positive > 0]
            cadence = float(np.median(positive)) * 86400 if len(positive) else np.nan
            metadata = json.loads(str(data["metadata"].item())) if "metadata" in data else {"time_unit": "day"}
            if explicit_config and metadata.get("flux_product", config.flux_product) != config.flux_product:
                raise ValueError("Cached curve flux product differs from preprocessing configuration")
            return LightCurve(time, flux,
                np.array(data["flux_err"], float) if "flux_err" in data else np.full(len(time), np.nan),
                np.array(data["quality"], np.int64) if "quality" in data else np.zeros(len(time), np.int64),
                np.full(len(time), config.exposure_seconds) if config.exposure_seconds else
                np.array(data["exposure_seconds"], float) if "exposure_seconds" in data else np.full(len(time), cadence),
                metadata).validate()
    from astropy.io import fits
    with fits.open(path, memmap=False) as hdus:
        names = set(hdus[1].columns.names or [])
        if {"TIME", "SAP_FLUX", "PDCSAP_FLUX"} <= names:
            header = hdus[0].header.copy()
            header.update(hdus[1].header)
            data = hdus[1].data
            ref = float(header.get("BJDREFI", 2457000)) + float(header.get("BJDREFF", 0))
            metadata = {key: header[key] for key in
                        ("TICID", "SECTOR", "TESSMAG", "CROWDSAP", "FLFRCSAP", "CAMERA", "CCD", "RA_OBJ", "DEC_OBJ")
                        if key in header and isinstance(header[key], (str, int, float, bool))}
            metadata.update(time_unit="day", time_format="jd", time_scale=header.get("TIMESYS", "TDB"),
                            flux_product=config.flux_product, source=str(path.resolve()),
                            exposure_convention="configured override, INT_TIME * NUM_FRM, or TIMEDEL approximation")
            integration = float(header.get("INT_TIME", np.nan)) * float(header.get("NUM_FRM", np.nan))
            exposure = config.exposure_seconds or (integration if np.isfinite(integration) and integration > 0
                                                   else float(header.get("TIMEDEL", np.nan)) * 86400)
            if not np.isfinite(exposure) or exposure <= 0:
                raise ValueError("SPOC product needs TIMEDEL or an exposure_seconds override")
            metadata["exposure_source"] = ("configured override" if config.exposure_seconds else
                "INT_TIME * NUM_FRM" if np.isfinite(integration) and integration > 0 else "TIMEDEL sampling-duration approximation")
            metadata["exposure_seconds"] = float(exposure)
            return LightCurve(np.asarray(data["TIME"], float) + ref,
                              np.asarray(data[config.flux_product], float),
                              np.asarray(data[config.flux_product + "_ERR"], float),
                              np.asarray(data["QUALITY"], np.int64),
                              np.full(len(data), exposure), metadata).validate()
    # Native simulator FITS tables carry astropy Time columns.
    from astropy.timeseries import TimeSeries
    curve = TimeSeries.read(path, astropy_native=True)
    time = np.asarray(curve.time.jd, float)
    exposure = config.exposure_seconds or float(np.median(np.diff(time))) * 86400
    metadata = {str(k): v for k, v in curve.meta.items()
                if isinstance(v, (str, int, float, bool)) and (not isinstance(v, float) or np.isfinite(v))}
    metadata.update(time_unit="day", time_format="jd", source=str(path.resolve()))
    return LightCurve(time, np.asarray(curve["flux"], float),
                      np.asarray(curve["flux_err"], float) if "flux_err" in curve.colnames else np.full(len(time), np.nan),
                      np.zeros(len(time), np.int64), np.full(len(time), exposure), metadata).validate()


def contiguous_slices(time, *, gap_factor=1.5):
    time = np.asarray(time, float)
    if len(time) < 2:
        return [slice(0, len(time))]
    differences = np.diff(time)
    if not np.all(np.isfinite(time)) or not np.all(differences > 0):
        raise ValueError("Times must be finite and strictly increasing")
    cadence = float(np.median(differences))
    edges = np.r_[0, np.flatnonzero(differences > gap_factor * cadence) + 1, len(time)]
    return [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def preprocess(curve, config=None):
    config = (config or PreprocessingConfig()).validate()
    curve.validate()
    finite = np.isfinite(curve.time) & np.isfinite(curve.flux)
    finite &= np.isfinite(curve.exposure_seconds) & (curve.exposure_seconds > 0)
    quality_ok = (np.asarray(curve.quality, np.int64) & config.quality_bitmask) == 0
    mask = finite & quality_ok
    if np.any(np.isfinite(curve.time)):
        mask &= curve.time >= np.nanmin(curve.time) + config.trim_start_hours / 24
    cleaned = curve.subset(mask)
    if len(cleaned.time) < 2:
        raise ValueError("Fewer than two usable cadences after preprocessing")
    order = np.argsort(cleaned.time, kind="stable")
    cleaned = cleaned.subset(order)
    unique = np.r_[True, np.diff(cleaned.time) > 0]
    duplicates = int((~unique).sum())
    cleaned = cleaned.subset(unique)
    if len(cleaned.time) < 2:
        raise ValueError("Fewer than two usable cadences after preprocessing")
    if config.normalize:
        baseline = float(np.median(cleaned.flux))
        if not np.isfinite(baseline) or baseline <= 0:
            raise ValueError("Normalization requires a positive finite median flux")
        cleaned.flux = cleaned.flux / baseline
        cleaned.flux_err = cleaned.flux_err / baseline
    segments = []
    retained = np.zeros(len(cleaned.time), bool)
    for span in contiguous_slices(cleaned.time, gap_factor=config.gap_factor):
        section = cleaned.subset(span)
        if len(section.time) < config.min_segment_points:
            continue
        retained[span] = True
        cadence = float(np.median(np.diff(section.time)))
        if config.detrend_window_days is not None:
            width = min(len(section.time) if len(section.time) % 2 else len(section.time) - 1,
                        max(3, int(round(config.detrend_window_days / cadence)) | 1))
            trend = median_filter(section.flux, size=width, mode="nearest")
            if np.any(trend <= 0):
                raise ValueError("Detrending produced a nonpositive baseline")
            section.flux /= trend
            section.flux_err /= trend
            cleaned.flux[span] = section.flux
            cleaned.flux_err[span] = section.flux_err
        # Resample barycentric jitter inside each uninterrupted run only.
        grid = np.linspace(section.time[0], section.time[-1], len(section.time))
        segments.append(LightCurve(grid, np.interp(grid, section.time, section.flux),
                                  np.interp(grid, section.time, section.flux_err),
                                  np.zeros(len(grid), np.int64),
                                  np.interp(grid, section.time, section.exposure_seconds), dict(section.metadata)))
    if not segments:
        raise ValueError("No contiguous segment meets min_segment_points")
    short_rows = int((~retained).sum())
    cleaned = cleaned.subset(retained)
    report = {"config": asdict(config), "input_rows": len(curve.time), "usable_rows": len(cleaned.time),
              "removed_nonfinite_or_invalid_exposure": int((~finite).sum()),
              "flagged_rows": int((~quality_ok).sum()), "duplicate_rows_removed": duplicates,
              "short_segment_rows_removed": short_rows,
              "segment_lengths": [len(s.time) for s in segments],
              "live_time_days": float(sum(s.exposure_seconds.sum() for s in segments) / 86400),
              "interpolation": "linear within contiguous runs; no interpolation across gaps"}
    return cleaned, segments, report


def noise_scale(flux):
    residual = np.diff(np.asarray(flux, float))
    scale = float(np.median(np.abs(residual - np.median(residual))) / (0.6744897501960817 * np.sqrt(2)))
    return float(max(scale, np.finfo(float).eps))
