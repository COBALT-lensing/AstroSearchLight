"""Maximum-cardinality, minimum-offset event assignment and observability."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_events(events, detections):
    """Match each event/detection at most once using per-event time tolerances."""
    if not events:
        return {"matches": [], "missed_events": [], "unmatched_detections": list(range(len(detections)))}
    for event in events:
        if not np.isfinite(event["time"]) or not np.isfinite(event["tolerance_days"]) or event["tolerance_days"] <= 0:
            raise ValueError("Event times and positive tolerances must be finite")
    n, m = len(events), len(detections)
    # A miss costs more than every possible valid timing offset combined.
    costs = np.full((n, m + n), float(n + 1))
    for i, event in enumerate(events):
        for j, detection in enumerate(detections):
            difference = abs(float(detection["peak_time"]) - event["time"])
            if not np.isfinite(difference):
                raise ValueError("Detection times must be finite")
            costs[i, j] = difference / event["tolerance_days"] if difference <= event["tolerance_days"] else (n + 1)**2
    row_indices, column_indices = linear_sum_assignment(costs)
    matches = [{"event_index": int(i), "detection_index": int(j),
                "timing_error_days": float(detections[j]["peak_time"] - events[i]["time"])}
               for i, j in zip(row_indices, column_indices) if j < m and costs[i, j] <= 1]
    return {"matches": matches,
            "missed_events": sorted(set(range(n)) - {pair["event_index"] for pair in matches}),
            "unmatched_detections": sorted(set(range(m)) - {pair["detection_index"] for pair in matches})}


def event_observability(event, segments, *, min_samples=3, min_coverage=0.5):
    width = float(event.get("duration_days", event["tolerance_days"]))
    if not np.isfinite(width) or width <= 0 or min_samples < 1 or not 0 <= min_coverage <= 1:
        raise ValueError("Invalid observability settings")
    low, high = event["time"] - width, event["time"] + width
    overlaps = []
    for segment in segments:
        half_exposure = segment.exposure_seconds / 172800
        lengths = np.maximum(0, np.minimum(segment.time + half_exposure, high)
                             - np.maximum(segment.time - half_exposure, low))
        overlaps.extend(lengths[lengths > 0].tolist())
    coverage = min(1., sum(overlaps) / (2 * width))
    return {"observed_samples": len(overlaps), "coverage_fraction": coverage,
            "observable": len(overlaps) >= min_samples and coverage >= min_coverage}


def truth_events(parameters, cadence_days):
    if "events" in parameters:
        events = [dict(event) for event in parameters["events"]]
    elif "event_time" in parameters:
        events = [{"time": parameters["event_time"],
                   "tolerance_days": parameters.get("match_tolerance_days", 2 * cadence_days),
                   "duration_days": parameters.get("duration_days", 2 * cadence_days),
                   "amplitude": parameters.get("amplitude")}]
    else:
        return []
    for event in events:
        event.setdefault("tolerance_days", max(2 * cadence_days, float(event.get("duration_days", 0))))
        match_events([event], [])
    return events
