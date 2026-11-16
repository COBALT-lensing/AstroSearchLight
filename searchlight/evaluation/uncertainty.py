"""Grouped uncertainty and event-recovery summaries."""

import numpy as np
from scipy.stats import beta

from .metrics import binary_metrics


def proportion_interval(successes, total, *, confidence=0.95):
    if total < 0 or not 0 <= successes <= total or not 0 < confidence < 1:
        raise ValueError("Invalid count or confidence")
    if not total:
        return [None, None]
    alpha = 1 - confidence
    low = 0. if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    high = 1. if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return [low, high]


def zero_event_upper_bound(trials, *, confidence=0.95):
    if trials <= 0 or not 0 < confidence < 1:
        raise ValueError("Positive independent trial count and confidence required")
    return float(-np.expm1(np.log1p(-confidence) / trials))



def group_bootstrap(rows, component, threshold, *, repeats=500, seed=42, confidence=0.95):
    if repeats < 0 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap settings")
    groups = sorted({row["group_id"] for row in rows})
    names = ("roc_auc", "average_precision", "recall", "precision", "false_positive_rate")
    if len(groups) < 2 or not repeats:
        return {"method": "group bootstrap", "groups": len(groups), "repeats": repeats,
                "status": "insufficient independent groups" if len(groups) < 2 else "disabled",
                "intervals": {name: [None, None] for name in names}}
    by_group = {group: [row for row in rows if row["group_id"] == group] for group in groups}
    rng = np.random.default_rng(seed)
    values = {name: [] for name in names}
    roc_grid = np.linspace(0, 1, 101)
    roc_values = []
    for _ in range(repeats):
        sampled = [row for group in rng.choice(groups, len(groups), replace=True) for row in by_group[group]]
        metrics = binary_metrics([row["truth"] for row in sampled], [row[component]["score"] for row in sampled], threshold=threshold)
        for name in names:
            if metrics[name] is not None:
                values[name].append(metrics[name])
        if metrics["roc_auc"] is not None:
            roc_values.append(np.interp(roc_grid, metrics["curve"]["fpr"], metrics["curve"]["recall"]))
    alpha = (1 - confidence) / 2
    return {"method": "target/system group percentile bootstrap; threshold fixed", "groups": len(groups),
            "repeats": repeats, "confidence": confidence,
            "valid_replicates": {name: len(series) for name, series in values.items()},
            "roc_band": {"fpr": roc_grid.tolist(),
                         "lower": np.quantile(roc_values, alpha, axis=0).tolist(),
                         "upper": np.quantile(roc_values, 1 - alpha, axis=0).tolist()} if roc_values else None,
            "intervals": {name: np.quantile(series, [alpha, 1 - alpha]).tolist() if series else [None, None]
                          for name, series in values.items()}}


def stratified_recovery(events, *, column="amplitude", bins=5, repeats=500, seed=42):
    valid = [event for event in events if event.get(column) is not None and np.isfinite(event[column])]
    if not valid:
        return []
    values = np.array([event[column] for event in valid], float)
    edges = np.linspace(values.min(), values.max(), bins + 1)
    if edges[0] == edges[-1]:
        edges = np.array([edges[0] - 1e-12, edges[0] + 1e-12])
    rows = []
    rng = np.random.default_rng(seed)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        subset = [event for event in valid if low <= event[column] and
                  (event[column] <= high if index == len(edges) - 2 else event[column] < high)]
        groups = sorted({event["group_id"] for event in subset})
        grouped = {group: np.zeros(5) for group in groups}
        for event in subset:
            grouped[event["group_id"]] += [1, int(event["recovered"]), int(event["observable"]),
                int(event["observable"] and event["recovered"]), int(event.get("combined_recovered", event["recovered"]))]
        totals = np.array(list(grouped.values()))
        fractions, observable_fractions, combined_fractions = [], [], []
        for _ in range(repeats if len(groups) >= 2 else 0):
            statistics = totals[rng.integers(0, len(groups), len(groups))].sum(axis=0)
            fractions.append(statistics[1] / statistics[0])
            combined_fractions.append(statistics[4] / statistics[0])
            if statistics[2]:
                observable_fractions.append(statistics[3] / statistics[2])
        successes = sum(event["recovered"] for event in subset)
        observable = [event for event in subset if event["observable"]]
        rows.append({"low": float(low), "high": float(high), "events": len(subset), "groups": len(groups),
            "recovered": successes, "fraction": successes / len(subset) if subset else None,
            "observable_events": len(observable),
            "observable_fraction": sum(e["recovered"] for e in observable) / len(observable) if observable else None,
            "combined_fraction": sum(e.get("combined_recovered", e["recovered"]) for e in subset) / len(subset) if subset else None,
            "interval": np.quantile(fractions, [0.025, 0.975]).tolist() if fractions else [None, None],
            "observable_interval": np.quantile(observable_fractions, [0.025, 0.975]).tolist() if observable_fractions else [None, None],
            "combined_interval": np.quantile(combined_fractions, [0.025, 0.975]).tolist() if combined_fractions else [None, None]})
    return rows


def recovery_intervals(events, *, repeats=500, seed=42):
    groups = sorted({event["group_id"] for event in events})
    values = {"overall_completeness": [], "observable_completeness": [], "combined_completeness": []}
    rng = np.random.default_rng(seed)
    by_group = {group: [event for event in events if event["group_id"] == group] for group in groups}
    for _ in range(repeats if len(groups) >= 2 else 0):
        subset = [event for group in rng.choice(groups, len(groups), replace=True) for event in by_group[group]]
        visible = [event for event in subset if event["observable"]]
        values["overall_completeness"].append(sum(event["recovered"] for event in subset) / len(subset))
        values["combined_completeness"].append(sum(event["combined_recovered"] for event in subset) / len(subset))
        if visible:
            values["observable_completeness"].append(sum(event["recovered"] for event in visible) / len(visible))
    return {"method": "target/system group bootstrap", "groups": len(groups), "repeats": repeats,
            "status": "insufficient independent groups" if len(groups) < 2 else "computed",
            "intervals": {key: np.quantile(series, [0.025, 0.975]).tolist() if series else [None, None]
                          for key, series in values.items()}}
