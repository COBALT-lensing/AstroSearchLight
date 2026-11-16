"""Reproducible component and cascade evaluation on simulations or SPOC data."""

from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import numpy as np

from .metrics import binary_metrics, select_threshold
from .scoring import score_cnn_paths, score_matched_filter
from .preprocessing import LightCurve, PreprocessingConfig, read_lightcurve, preprocess, noise_scale
from .matching import truth_events, match_events, event_observability
from .uncertainty import group_bootstrap, zero_event_upper_bound, stratified_recovery, recovery_intervals

_GPU_PROGRESS = False


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def template_fingerprint(templates):
    digest = hashlib.sha256()
    for template in templates:
        for values in (template.time, template.flux):
            digest.update(np.asarray(values, dtype="<f8").tobytes())
        digest.update(str(template.event_half_width).encode())
    return digest.hexdigest()


def load_arrays(source):
    curve = read_lightcurve(source)
    return curve.time, curve.flux


def _partial_auc(metrics, max_fpr):
    curve = metrics["curve"]
    if metrics["roc_auc"] is None or max_fpr <= 0:
        return None
    fpr, recall = np.asarray(curve["fpr"]), np.asarray(curve["recall"])
    keep = fpr < max_fpr
    x = np.r_[fpr[keep], max_fpr]
    y = np.r_[recall[keep], np.interp(max_fpr, fpr, recall)]
    area = float(np.trapezoid(y, x))
    return {"max_fpr": max_fpr, "area": area, "area_divided_by_max_fpr": area / max_fpr}


def _summary(rows, component, threshold, repeats, seed, max_fpr):
    known = [row for row in rows if row["truth"] is not None]
    if not known:
        return None
    metrics = binary_metrics([r["truth"] for r in known], [r[component]["score"] for r in known], threshold=threshold)
    metrics["uncertainty"] = group_bootstrap(known, component, threshold, repeats=repeats, seed=seed)
    metrics["partial_roc_auc"] = _partial_auc(metrics, max_fpr)
    negative_groups = {row["group_id"] for row in known if row["truth"] == 0}
    false_groups = {row["group_id"] for row in known if row["truth"] == 0 and row[component]["accepted"]}
    metrics["negative_groups"] = len(negative_groups)
    metrics["false_positive_groups"] = len(false_groups)
    metrics["zero_false_positive_group_upper_bound_95"] = (
        zero_event_upper_bound(len(negative_groups)) if negative_groups and not false_groups else None)
    return metrics


def _prepare_and_score(job):
    sample, settings, preprocessing, templates, backend, normalize_scores, positive_class = job
    if preprocessing is None:
        # Retain the original adapter seam for backwards-compatible local callers.
        time, flux = load_arrays(sample.source)
        exposure = float(np.median(np.diff(time))) * 86400
        raw = LightCurve(time, flux, np.full(len(time), np.nan), np.zeros(len(time), np.int64), np.full(len(time), exposure))
    else:
        raw = read_lightcurve(sample.source, settings)
    clean, segments, report = preprocess(raw, settings)
    segment_scores = []
    segment_iterator = enumerate(segments)
    if backend == "cuda_fft" and _GPU_PROGRESS:
        from tqdm.auto import tqdm
        segment_iterator = tqdm(segment_iterator, total=len(segments), desc="GPU matched-filter segments", position=1, leave=False)
    for index, segment in segment_iterator:
        result = asdict(score_matched_filter(sample.sample_id, segment.time, segment.flux, templates, backend=backend))
        scale = noise_scale(segment.flux) if normalize_scores else 1.
        result.update(raw_score=result["score"], noise_scale=scale, segment_index=index)
        result["score"] /= scale
        result["score_kind"] = "maximum_noise_normalized_correlation" if normalize_scores else result["score_kind"]
        segment_scores.append(result)
    best = max(segment_scores, key=lambda item: item["score"])
    row = {"sample_id": sample.sample_id, "group_id": sample.group_id,
                 "split": sample.split, "label": sample.label,
                 "suite": sample.suite,
                 "truth": None if sample.label is None else int(sample.label == positive_class),
                 "matched_filter": best, "parameters": sample.parameters}
    return row, clean, segments, report


def _detect_sample(job):
    from ..matched_filtering import detect_matched_filter_peaks
    segments, templates, threshold, normalize_scores, backend = job
    detections = []
    if threshold is not None:
        for segment_index, segment in enumerate(segments):
            scale = noise_scale(segment.flux) if normalize_scores else 1.
            if backend == "cuda_fft":
                from .cuda_filter import engine
                found = engine(templates).detect(segment.time, segment.flux, height=threshold * scale)
            else:
                found = detect_matched_filter_peaks(segment.time, segment.flux, templates,
                    height=threshold * scale, prominence=0, min_score_sigma=None,
                    dedupe_overlaps=True, backend=backend)
            detections.extend({**asdict(match), "segment_index": segment_index,
                "index_reference": "segment-local after within-run resampling"} for match in found)
    return detections


_DETECTOR_CONTEXT = None


def _initialize_detector(template_source, settings, preprocessing, backend, normalize_scores, positive_class):
    from .parallel import initialize_worker
    initialize_worker()
    if isinstance(template_source, (str, Path)):
        from ..matched_filtering import load_template_library
        templates = load_template_library(template_source)
    else:
        templates = template_source
    global _DETECTOR_CONTEXT
    _DETECTOR_CONTEXT = (settings, preprocessing, templates, backend, normalize_scores, positive_class)


def _prepare_with_context(sample):
    return _prepare_and_score((sample, *_DETECTOR_CONTEXT))


def _detect_with_context(job):
    segments, threshold = job
    _, _, templates, backend, normalize_scores, _ = _DETECTOR_CONTEXT
    return _detect_sample((segments, templates, threshold, normalize_scores, backend))


def evaluate(manifest, templates, output, *, positive_class="lensing", max_fpr=0.05,
             cnn=None, provenance=None, preprocessing=None, normalize_scores=False,
             backend="cpu", bootstrap_repeats=200, seed=42, pipeline_order="matched_filter_then_cnn",
             fixed_thresholds=None, min_event_samples=3, min_event_coverage=0.5,
             compare_backend=False, progress=False, workers=1, template_path=None):
    """Tune on tuning rows, then evaluate test rows and explicit injection truth.

    Unknown labels remain unlabelled candidate-rate observations. A supplied
    fixed threshold allows scoring a real-data set without labelled tuning rows.
    Event detection uses production center-flux/polarity cuts. Cascade ranking
    sweeps the minimum standardized margin of both components; operating point
    zero requires both independently tuned component thresholds to pass.
    """
    global _GPU_PROGRESS
    _GPU_PROGRESS = progress
    manifest.validate()
    template_metadata = getattr(templates, "metadata", None)
    templates = list(templates)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation run: {output}")
    settings = PreprocessingConfig(**(preprocessing or {})).validate()
    if pipeline_order not in {"matched_filter_then_cnn", "cnn_then_matched_filter"}:
        raise ValueError("Invalid pipeline_order")
    if not 0 <= max_fpr <= 1 or bootstrap_repeats < 0 or min_event_samples < 1 or not 0 <= min_event_coverage <= 1:
        raise ValueError("Invalid evaluation settings")
    fixed_thresholds = dict(fixed_thresholds or {})
    if set(fixed_thresholds) - {"matched_filter", "cnn"}:
        raise ValueError("Unknown fixed-threshold component")
    selected = [sample for sample in manifest.samples if sample.split in {"tuning", "test"}]
    if not selected:
        raise ValueError("No tuning/test samples")
    if cnn is not None and not hasattr(cnn, "classify_curves") and any(not s.source.endswith((".fits", ".fits.gz")) for s in selected):
        raise ValueError("Legacy CNN evaluation requires FITS sources; use EvaluationCNN for shared preprocessed curves")
    rows, prepared, preprocessing_reports = [], {}, {}
    from .parallel import process_map, worker_count
    workers = worker_count(workers, len(selected))
    if backend != "cpu":
        workers = 1  # A single process owns an accelerator-backed detector.
    if workers > 1:
        # A production bank can be hundreds of MB: load once per worker rather
        # than pickle it for every light curve.
        initargs = (str(template_path) if template_path else templates, settings, preprocessing,
                    backend, normalize_scores, positive_class)
        scored = process_map(_prepare_with_context, selected, workers=workers,
            desc="Preprocess / matched filter", progress=progress,
            initializer=_initialize_detector, initargs=initargs)
    else:
        jobs = [(sample, settings, preprocessing, templates, backend, normalize_scores, positive_class) for sample in selected]
        scored = process_map(_prepare_and_score, jobs, workers=1,
            desc="Preprocess / matched filter", progress=progress)
    for sample, (row, clean, segments, report) in zip(selected, scored):
        rows.append(row)
        prepared[sample.sample_id] = (clean, segments)
        preprocessing_reports[sample.sample_id] = report
    del scored
    components = ["matched_filter"]
    if cnn is not None:
        cnn.progress = progress
        predictions = (cnn.classify_curves(selected, {key: value[0] for key, value in prepared.items()}, positive_class=positive_class)
                       if hasattr(cnn, "classify_curves") else score_cnn_paths(cnn, selected, positive_class=positive_class))
        cnn_results = {record.sample_id: asdict(record) for record in predictions}
        if set(cnn_results) != {s.sample_id for s in selected}:
            raise ValueError("CNN outputs do not match manifest sample IDs")
        for row in rows:
            row["cnn"] = cnn_results[row["sample_id"]]
        components.append("cnn")
    thresholds = {}
    for component in components:
        tuning = [row for row in rows if row["split"] == "tuning" and row["truth"] is not None]
        if component in fixed_thresholds:
            threshold = fixed_thresholds[component]
            if threshold is not None and not np.isfinite(threshold):
                raise ValueError("Fixed thresholds must be finite or null (reject all)")
        else:
            threshold = select_threshold([row["truth"] for row in tuning], [row[component]["score"] for row in tuning], max_fpr=max_fpr)
        thresholds[component] = threshold
        for row in rows:
            row[component]["accepted"] = threshold is not None and row[component]["score"] >= threshold
    if cnn is not None:
        tuning_values = [row["matched_filter"]["score"] for row in rows if row["split"] == "tuning"]
        margin_scale = max(float(np.std(tuning_values)) if tuning_values else 1., np.finfo(float).eps)
        # Preserve continuous ranking even when tuning selects a reject-all gate.
        # Its finite ranking anchor uses tuning scores only; operational rejection
        # stays explicit through a null combined threshold.
        anchors = {}
        for name in components:
            values = [r[name]["score"] for r in rows if r["split"] == "tuning"]
            anchors[name] = thresholds[name] if thresholds[name] is not None else max(values, default=0.)
        for row in rows:
            score = min((row["matched_filter"]["score"] - anchors["matched_filter"]) / margin_scale,
                        row["cnn"]["score"] - anchors["cnn"])
            row["combined"] = {"score": float(score), "score_kind": "minimum_component_threshold_margin",
                                "accepted": all(row[name]["accepted"] for name in components)}
        thresholds["combined"] = 0. if all(thresholds[name] is not None for name in components) else None
        components.append("combined")
    metrics = {name: {} for name in components}
    metrics_by_suite = {}
    candidate_rates, attrition = {}, {}
    for split in ("tuning", "test"):
        subset = [row for row in rows if row["split"] == split]
        if not subset:
            continue
        candidate_rates[split] = {}
        for component in components:
            summary = _summary(subset, component, thresholds[component], bootstrap_repeats, seed, max_fpr)
            if summary is not None:
                metrics[component][split] = summary
            for suite in sorted({row["suite"] for row in subset}):
                suite_summary = _summary([row for row in subset if row["suite"] == suite], component,
                    thresholds[component], bootstrap_repeats, seed, max_fpr)
                if suite_summary is not None:
                    metrics_by_suite.setdefault(suite, {}).setdefault(component, {})[split] = suite_summary
            unknown = [row for row in subset if row["truth"] is None]
            candidate_rates[split][component] = {"unlabelled_rows": len(unknown),
                "candidates": sum(row[component]["accepted"] for row in unknown),
                "candidate_fraction": sum(row[component]["accepted"] for row in unknown) / len(unknown) if unknown else None}
        if cnn is not None:
            first, second = ("matched_filter", "cnn") if pipeline_order == "matched_filter_then_cnn" else ("cnn", "matched_filter")
            survivors = [row for row in subset if row[first]["accepted"]]
            known_survivors = [row for row in survivors if row["truth"] is not None]
            attrition[split] = {"order": [first, second], "input_rows": len(subset),
                "after_first": len(survivors), "after_second": sum(row["combined"]["accepted"] for row in subset),
                "second_stage_conditional_metrics": binary_metrics(
                    [r["truth"] for r in known_survivors], [r[second]["score"] for r in known_survivors],
                    threshold=thresholds[second]) if known_survivors else None,
                "note": "Both components scored independently for comparison; counts model production gating"}
    event_rows, event_details = [], []
    by_id = {row["sample_id"]: row for row in rows}
    if workers > 1:
        detection_jobs = [(prepared[s.sample_id][1], thresholds["matched_filter"]) for s in selected]
        all_detections = process_map(_detect_with_context, detection_jobs, workers=workers,
            desc="Locate events", progress=progress, initializer=_initialize_detector, initargs=initargs)
    else:
        detection_jobs = [(prepared[s.sample_id][1], templates, thresholds["matched_filter"], normalize_scores, backend)
                          for s in selected]
        all_detections = process_map(_detect_sample, detection_jobs, workers=1, desc="Locate events", progress=progress)
    for sample, detections in zip(selected, all_detections):
        clean, segments = prepared[sample.sample_id]
        cadence = float(np.median(clean.exposure_seconds)) / 86400
        events = truth_events(sample.parameters, cadence)
        if events and (sample.label != positive_class or manifest.metadata.get("time_unit") != "day"):
            raise ValueError("Event truth requires a positive label and explicit day time units")
        assignment = match_events(events, detections)
        combined_accept = by_id[sample.sample_id].get("cnn", {}).get("accepted", True)
        recovered_indices = {pair["event_index"]: pair for pair in assignment["matches"]}
        details = []
        for index, event in enumerate(events):
            observed = event_observability(event, segments, min_samples=min_event_samples, min_coverage=min_event_coverage)
            recovered = index in recovered_indices
            detail = {**event, **observed, "sample_id": sample.sample_id, "group_id": sample.group_id,
                "split": sample.split, "suite": sample.suite, "event_index": index,
                "recovered": recovered, "combined_recovered": recovered and combined_accept,
                "timing_error_days": recovered_indices[index]["timing_error_days"] if recovered else None,
                "failure_stage": None if recovered and combined_accept else
                    "cnn" if recovered else "coverage" if not observed["observable"] else "matched_filter",
                "noise": noise_scale(clean.flux), "cadence_seconds": cadence * 86400,
                "magnitude": clean.metadata.get("TESSMAG", sample.parameters.get("magnitude")),
                "crowding": clean.metadata.get("CROWDSAP", sample.parameters.get("crowding"))}
            details.append(detail)
            event_details.append(detail)
        event_rows.append({"sample_id": sample.sample_id, "split": sample.split,
            "truth_event_time": events[0]["time"] if len(events) == 1 else None,
            "recovered": any(detail["recovered"] for detail in details) if events else None,
            "timing_error_days": details[0]["timing_error_days"] if len(details) == 1 else None,
            "unmatched_detections": len(assignment["unmatched_detections"]),
            "combined_unmatched_detections": len(assignment["unmatched_detections"]) if combined_accept else 0,
            "detections": detections, "events": details, "assignment": assignment,
            "live_time_days": preprocessing_reports[sample.sample_id]["live_time_days"],
            "known_negative": sample.label is not None and sample.label != positive_class,
            "unlabelled": sample.label is None})
    detections_by_sample = {row["sample_id"]: row["detections"] for row in event_rows}
    parents = {sample.sample_id: sample.parent_id for sample in selected}
    for detail in event_details:
        control = detections_by_sample.get(parents[detail["sample_id"]], [])
        detail["control_has_candidate"] = any(abs(detection["peak_time"] - detail["time"]) <= detail["tolerance_days"]
                                              for detection in control)
        detail["incremental_recovered"] = detail["recovered"] and not detail["control_has_candidate"]
    recovery, strata = {}, {}
    for split in ("tuning", "test"):
        details = [row for row in event_details if row["split"] == split]
        rows_for_split = [row for row in event_rows if row["split"] == split]
        observed = [row for row in details if row["observable"]]
        negatives = [row for row in rows_for_split if row["known_negative"]]
        negative_days = sum(row["live_time_days"] for row in negatives)
        unknowns = [row for row in rows_for_split if row["unlabelled"]]
        unknown_days = sum(row["live_time_days"] for row in unknowns)
        recovery[split] = {"injected_events": len(details), "recovered": sum(row["recovered"] for row in details),
            "uncertainty": recovery_intervals(details, repeats=bootstrap_repeats, seed=seed),
            "overall_completeness": sum(row["recovered"] for row in details) / len(details) if details else None,
            "observable_events": len(observed), "observable_recovered": sum(row["recovered"] for row in observed),
            "observable_completeness": sum(row["recovered"] for row in observed) / len(observed) if observed else None,
            "combined_recovered": sum(row["combined_recovered"] for row in details) if cnn else None,
            "combined_completeness": sum(row["combined_recovered"] for row in details) / len(details) if details and cnn else None,
            "combined_observable_completeness": sum(row["combined_recovered"] for row in observed) / len(observed) if observed and cnn else None,
            "control_coincident_events": sum(row["control_has_candidate"] for row in details),
            "incremental_recovered": sum(row["incremental_recovered"] for row in details),
            "false_detections_per_negative_observing_day": sum(row["unmatched_detections"] for row in negatives) / negative_days if negative_days else None,
            "unlabelled_candidates_per_observing_day": sum(len(row["detections"]) for row in unknowns) / unknown_days if unknown_days else None,
            "negative_observing_days": negative_days,
            "failure_stages": {stage: sum(row["failure_stage"] == stage for row in details)
                               for stage in ("coverage", "matched_filter", "cnn")}}
        strata[split] = {column: stratified_recovery(details, column=column, repeats=bootstrap_repeats, seed=seed)
                         for column in ("amplitude", "duration_days", "magnitude", "noise", "cadence_seconds", "crowding", "coverage_fraction")}
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    config = {"schema_version": 2, "created_utc": datetime.now(timezone.utc).isoformat(),
        "positive_class": positive_class, "evaluation_unit": "target_sector", "seed": seed,
        "threshold_selection": {"split": "tuning", "objective": "maximum recall at empirical FPR budget", "max_fpr": max_fpr,
                                "fixed_threshold_components": list(fixed_thresholds)},
        "thresholds": thresholds, "components": components, "backend": backend,
        "combined_ranking_anchors": anchors if cnn is not None else None,
        "template_sha256": template_fingerprint(templates), "git_revision": revision, "git_dirty": dirty,
        "template_library_metadata": template_metadata,
        "dataset_metadata": manifest.metadata,
        "python": platform.python_version(), "pipeline_order": pipeline_order,
        "preprocessing": asdict(settings), "normalize_scores": normalize_scores,
        "workers": workers, "bootstrap_repeats": bootstrap_repeats, "min_event_samples": min_event_samples, "min_event_coverage": min_event_coverage,
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib")},
        "source_hashes": {s.source: file_hash(s.source) for s in selected},
        "evaluation_code_hashes": {str(path.relative_to(Path(__file__).parents[1])): file_hash(path)
            for path in [*Path(__file__).parent.glob("*.py"), Path(__file__).parents[1] / "matched_filtering.py"]},
        "excluded_sample_ids": [s.sample_id for s in manifest.samples if s.split == "train"],
        "provenance": provenance or {},
        "event_detector": {"height": thresholds["matched_filter"], "prominence": 0, "min_score_sigma": None,
            "dedupe_overlaps": True, "allow_negative_peaks": False, "max_matches": None, "distance": None,
            "center_flux_sigma": 4.0, "center_flux_and_positive_template_cuts": "production defaults"},
        "limitations": ["Dataset independence and label provenance require an external audit",
            "Bootstrap intervals condition on frozen thresholds; do not include model/tuning uncertainty",
            "Delivered light-curve injections do not measure upstream SPOC attenuation",
            "Normalized correlation uses a robust white-noise scale; correlated noise is calibrated empirically",
            "Combined ROC sweeps joint margins; conditional fixed-gate performance is in attrition.json"]}
    if cnn is not None:
        config["cnn"] = {"workers": getattr(cnn, "workers", 1), "checkpoint_sha256": file_hash(cnn.MODEL_SAVE_PATH), "classes": list(cnn.classes),
            "class_mapping_sha256": file_hash(cnn.CLASS_SAVE_PATH),
            "model_source_sha256": file_hash(Path(__file__).parents[1] / "models/simulens/v1.py"),
            "split_sizes_days": [None if np.isinf(x) else float(x) for x in cnn.split_sizes],
            "max_segments_per_item": cnn.max_segments_per_item,
            "rendering": getattr(cnn, "rendering", "legacy model defaults"),
            "shared_preprocessing": hasattr(cnn, "classify_curves")}
    output.mkdir(parents=True)
    manifest.write(output / "manifest.json")
    for name, value in {"run": config, "scores": rows, "metrics": metrics, "events": event_rows,
                        "event_details": event_details, "preprocessing": preprocessing_reports,
                        "metrics_by_suite": metrics_by_suite,
                        "candidate_rates": candidate_rates, "attrition": attrition,
                        "event_summary": recovery, "stratification": strata}.items():
        write_json(output / f"{name}.json", value)
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["component", "split", "n", "roc_auc", "average_precision", "recall", "precision", "false_positive_rate"])
        writer.writeheader()
        for component, splits in metrics.items():
            for split, values in splits.items():
                writer.writerow({"component": component, "split": split,
                                 **{key: values[key] for key in writer.fieldnames[2:]}})
    if compare_backend:
        from .validation import compare_backends
        write_json(output / "backend_comparison.json", compare_backends(selected, prepared, templates))
    from .reporting import save_figures
    save_figures(output)
    write_json(output / "complete.json", {"status": "complete"})
    return metrics
