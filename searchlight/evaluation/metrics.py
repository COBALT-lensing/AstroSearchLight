"""Binary ranking metrics with explicit ties and JSON-safe undefined values."""

import numpy as np


def binary_metrics(truth, scores, *, threshold):
    y = np.asarray(truth)
    score = np.asarray(scores, dtype=float)
    if y.ndim != 1 or score.shape != y.shape or not len(y):
        raise ValueError("Expected equally sized nonempty truth and score vectors")
    if not np.all(np.isin(y, [0, 1])) or not np.all(np.isfinite(score)):
        raise ValueError("Expected binary truth and finite scores")
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError("threshold must be finite or None (reject all)")
    y = y.astype(bool)
    order = np.argsort(-score, kind="stable")
    ranked = score[order]
    ends = np.r_[np.flatnonzero(np.diff(ranked)), len(y) - 1]
    tp = np.r_[0, np.cumsum(y[order])[ends]]
    fp = np.r_[0, np.cumsum(~y[order])[ends]]
    positives, negatives = int(y.sum()), int((~y).sum())
    tpr = tp / positives if positives else None
    fpr = fp / negatives if negatives else None
    precision = np.divide(tp, tp + fp, out=np.ones(len(tp)), where=(tp + fp) > 0)
    prediction = np.zeros(len(y), dtype=bool) if threshold is None else score >= threshold
    tn, false_positive = int((~y & ~prediction).sum()), int((~y & prediction).sum())
    false_negative, true_positive = int((y & ~prediction).sum()), int((y & prediction).sum())
    matrix = [[tn, false_positive], [false_negative, true_positive]]
    divide = lambda numerator, denominator: numerator / denominator if denominator else None
    return {
        "n": len(y), "positives": positives, "negatives": negatives,
        "threshold": threshold, "threshold_rule": "score >= threshold; null rejects all",
        "confusion_matrix": matrix, "confusion_labels": ["non-lensing", "lensing"],
        "confusion_row_normalized": [
            [divide(tn, negatives), divide(false_positive, negatives)],
            [divide(false_negative, positives), divide(true_positive, positives)],
        ],
        "recall": divide(true_positive, positives), "specificity": divide(tn, negatives),
        "precision": divide(true_positive, true_positive + false_positive),
        "false_positive_rate": divide(false_positive, negatives),
        "roc_auc": float(np.trapezoid(tpr, fpr)) if positives and negatives else None,
        "average_precision": float(np.sum(np.diff(tpr) * precision[1:])) if positives else None,
        "curve": {"thresholds": [None] + ranked[ends].tolist(),
                  "fpr": fpr.tolist() if fpr is not None else None,
                  "recall": tpr.tolist() if tpr is not None else None,
                  "precision": precision.tolist()},
    }


def select_threshold(truth, scores, *, max_fpr=0.05):
    """Maximize tuning recall under an empirical FPR budget; prefer tighter ties."""
    if not 0 <= max_fpr <= 1:
        raise ValueError("max_fpr must lie in [0, 1]")
    metrics = binary_metrics(truth, scores, threshold=None)
    if not metrics["positives"] or not metrics["negatives"]:
        raise ValueError("Threshold tuning requires both positive and negative samples")
    curve = metrics["curve"]
    eligible = np.flatnonzero(np.asarray(curve["fpr"]) <= max_fpr)
    best = eligible[np.argmax(np.asarray(curve["recall"])[eligible])]
    return curve["thresholds"][best]
