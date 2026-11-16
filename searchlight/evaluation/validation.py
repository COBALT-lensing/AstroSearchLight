"""Representative CPU/JAX comparisons using production correlation and detection."""

from dataclasses import asdict
import numpy as np


def compare_backends(samples, prepared, templates, *, max_curves=5, rtol=2e-4, atol=1e-6):
    from .scoring import score_matched_filter
    from ..matched_filtering import detect_matched_filter_peaks
    try:
        import jax
        devices = [str(device) for device in jax.devices()]
    except (ImportError, RuntimeError) as exc:
        return {"status": "unavailable", "reason": str(exc)}
    rows = []
    for sample in samples[:max_curves]:
        _, segments = prepared[sample.sample_id]
        for index, segment in enumerate(segments):
            scores, detections = {}, {}
            for backend in ("cpu", "jax"):
                scores[backend] = asdict(score_matched_filter(sample.sample_id, segment.time, segment.flux, templates, backend=backend))
                detections[backend] = [asdict(match) for match in detect_matched_filter_peaks(
                    segment.time, segment.flux, templates, backend=backend)]
            score_agreement = bool(np.isclose(scores["cpu"]["score"], scores["jax"]["score"], rtol=rtol, atol=atol))
            peak_agreement = [d["peak_index"] for d in detections["cpu"]] == [d["peak_index"] for d in detections["jax"]]
            rows.append({"sample_id": sample.sample_id, "segment_index": index,
                         "scores": scores, "detection_counts": {key: len(value) for key, value in detections.items()},
                         "score_agreement": score_agreement, "peak_agreement": peak_agreement})
    return {"status": "passed" if all(row["score_agreement"] and row["peak_agreement"] for row in rows) else "differences",
            "devices": devices, "rtol": rtol, "atol": atol, "comparisons": rows}
