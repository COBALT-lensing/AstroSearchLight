"""Static figures shared by saved reports and the evaluation notebook."""

import json
from pathlib import Path

import numpy as np


def read_run(directory):
    directory = Path(directory)
    return {name: json.loads((directory / f"{name}.json").read_text())
            for name in ("run", "manifest", "scores", "metrics", "events", "event_details",
                         "event_summary", "stratification", "attrition", "candidate_rates", "backend_comparison")
            if (directory / f"{name}.json").exists()}


def plot_classification(results, *, split="test"):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for component, splits in results["metrics"].items():
        if split not in splits:
            continue
        metrics = splits[split]
        curve = metrics["curve"]
        if curve["fpr"] is not None and curve["recall"] is not None:
            line, = axes[0].plot(curve["fpr"], curve["recall"],
                                label=f'{component}: AUC {metrics["roc_auc"]:.3f}')
            band = metrics.get("uncertainty", {}).get("roc_band")
            if band:
                axes[0].fill_between(band["fpr"], band["lower"], band["upper"], alpha=.12, color=line.get_color())
        if curve["recall"] is not None:
            axes[1].step(curve["recall"], curve["precision"], where="pre",
                         label=f'{component}: AP {metrics["average_precision"]:.3f}')
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC — {split}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title=f"Precision–recall — {split}")
    for axis in axes:
        axis.set(xlim=(0, 1), ylim=(0, 1.02))
        if axis.get_legend_handles_labels()[0]:
            axis.legend()
    figure.tight_layout()
    return figure


def plot_confusion(results, *, split="test", normalized=False):
    import matplotlib.pyplot as plt

    components = [component for component, splits in results["metrics"].items() if split in splits]
    if not components:
        figure, axis = plt.subplots()
        axis.text(.5, .5, "No labelled samples in this split", ha="center")
        return figure
    figure, axes = plt.subplots(1, len(components), figsize=(5 * len(components), 4), squeeze=False)
    for axis, component in zip(axes[0], components):
        metrics = results["metrics"][component][split]
        key = "confusion_row_normalized" if normalized else "confusion_matrix"
        matrix = np.array(metrics[key], dtype=float)
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if normalized else None)
        for (i, j), value in np.ndenumerate(matrix):
            label = "undefined" if not np.isfinite(value) else f"{value:.2f}" if normalized else f"{value:g}"
            axis.text(j, i, label, ha="center", va="center")
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=metrics["confusion_labels"],
                 yticklabels=metrics["confusion_labels"], xlabel="Predicted", ylabel="True",
                 title=f"{component} — {split}\nfrozen operating threshold")
    figure.tight_layout()
    return figure


def recovery_table(results, *, split="test", bins=5):
    if "event_details" in results:
        from .uncertainty import stratified_recovery
        rows = stratified_recovery([event for event in results["event_details"] if event["split"] == split], repeats=0, bins=bins)
        return [{"amplitude_low": row["low"], "amplitude_high": row["high"],
                 "injected": row["events"], "recovered": row["recovered"], "fraction": row["fraction"]} for row in rows]
    scores = {row["sample_id"]: row for row in results["scores"]}
    pairs = [(scores[event["sample_id"]]["parameters"].get("amplitude"), event["recovered"])
             for event in results["events"] if event["split"] == split and event["recovered"] is not None]
    pairs = [(amplitude, recovered) for amplitude, recovered in pairs if amplitude is not None]
    if not pairs:
        return []
    amplitudes = np.array([pair[0] for pair in pairs])
    recovered = np.array([pair[1] for pair in pairs])
    edges = np.linspace(amplitudes.min(), amplitudes.max(), bins + 1)
    if edges[0] == edges[-1]:
        edges = np.array([edges[0] - 1e-12, edges[0] + 1e-12])
    rows = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (amplitudes >= low) & ((amplitudes <= high) if index == len(edges) - 2 else (amplitudes < high))
        total = int(selected.sum())
        successes = int(recovered[selected].sum())
        rows.append({"amplitude_low": float(low), "amplitude_high": float(high),
                     "injected": total, "recovered": successes,
                     "fraction": successes / total if total else None})
    return rows


def plot_recovery(results, *, split="test"):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4))
    for row in recovery_table(results, split=split):
        if row["fraction"] is None:
            continue
        x = (row["amplitude_low"] + row["amplitude_high"]) / 2
        axis.plot(x, row["fraction"], "o", color="tab:blue")
        axis.annotate(f'{row["recovered"]}/{row["injected"]}', (x, row["fraction"]), xytext=(0, 8), textcoords="offset points", ha="center")
    axis.set(xlabel="Injected amplitude (flux units)", ylabel="Event recovery fraction",
             ylim=(-0.05, 1.2), title=f"Matched-filter event recovery — {split}")
    figure.tight_layout()
    return figure


def save_figures(directory):
    import matplotlib.pyplot as plt

    directory = Path(directory)
    results = read_run(directory)
    functions = {"roc_precision_recall": plot_classification,
                 "confusion_counts": plot_confusion,
                 "confusion_normalized": lambda data: plot_confusion(data, normalized=True),
                 "injection_recovery": plot_recovery,
                 "roc_low_fpr": plot_low_fpr, "completeness_strata": plot_stratification,
                 "pipeline_attrition": plot_attrition, "timing_errors": plot_timing}
    for name, plot in functions.items():
        figure = plot(results)
        for extension in ("png", "pdf"):
            figure.savefig(directory / f"{name}.{extension}", dpi=150, bbox_inches="tight")
        plt.close(figure)
    (directory / "recovery.json").write_text(json.dumps(recovery_table(results), indent=2) + "\n")
    lines = ["# Evaluation report", "", f'Created: {results["run"]["created_utc"]}', "",
             "Thresholds are frozen from tuning data or supplied explicitly. See run.json for provenance.", "",
             "| Component | Test n | ROC AUC | AP | Recall | False positive rate |",
             "| --- | --- | --- | --- | --- | --- |"]
    for component, splits in results["metrics"].items():
        if "test" in splits:
            row = splits["test"]
            lines.append(f'| {component} | {row["n"]} | {row["roc_auc"]} | {row["average_precision"]} | {row["recall"]} | {row["false_positive_rate"]} |')
    lines += ["", "![ROC and precision–recall](roc_precision_recall.png)", "",
              "![Confusion matrices](confusion_counts.png)", "",
              "![Completeness](completeness_strata.png)", "", "## Limits", ""]
    lines += [f"- {limitation}" for limitation in results["run"]["limitations"]]
    (directory / "report.md").write_text("\n".join(lines) + "\n")


def plot_low_fpr(results, *, split="test", max_fpr=None):
    import matplotlib.pyplot as plt
    max_fpr = max_fpr or max(.01, results["run"]["threshold_selection"]["max_fpr"])
    figure, axis = plt.subplots(figsize=(7, 4))
    for component, splits in results["metrics"].items():
        if split not in splits:
            continue
        curve = splits[split]["curve"]
        if curve["fpr"] is not None and curve["recall"] is not None:
            axis.plot(curve["fpr"], curve["recall"], label=component)
    axis.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, max_fpr),
             ylim=(0, 1.02), title=f"Low-FPR ROC — {split}")
    if axis.get_legend_handles_labels()[0]:
        axis.legend()
    figure.tight_layout()
    return figure


def plot_stratification(results, *, split="test"):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    columns = results.get("stratification", {}).get(split, {})
    figure, axes = plt.subplots(2, 4, figsize=(16, 7))
    for axis, column in zip(axes.flat, ["amplitude", "duration_days", "magnitude", "noise", "cadence_seconds", "crowding", "coverage_fraction"]):
        bins = columns.get(column, [])
        points = [row for row in bins if row["fraction"] is not None]
        if points:
            x = [(row["low"] + row["high"]) / 2 for row in points]
            y = [row["fraction"] for row in points]
            axis.plot(x, y, "o-", label="All injected events")
            for position, fraction, row in zip(x, y, points):
                low, high = row["interval"]
                if low is not None:
                    axis.vlines(position, low, high, color="tab:blue")
                axis.annotate(f'n={row["events"]}', (position, fraction), xytext=(0, 6),
                              textcoords="offset points", ha="center", fontsize=8)
            observed = [(position, row["observable_fraction"]) for position, row in zip(x, points)
                        if row["observable_fraction"] is not None]
            if observed:
                axis.plot(*zip(*observed), "x--", label="Observable events")
            if "cnn" in results["run"]["components"] and any(row.get("combined_fraction") is not None for row in points):
                axis.plot(x, [row.get("combined_fraction") for row in points], "s:", label="Cascade, all events")
        else:
            axis.text(.5, .5, "No data", transform=axis.transAxes, ha="center")
        axis.set(xlabel=column, ylabel="Recovery fraction", ylim=(-.05, 1.15))
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower right")
    figure.suptitle(f"Completeness and target-bootstrap 95% intervals — {split}")
    figure.tight_layout()
    return figure


def plot_attrition(results, *, split="test"):
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 4))
    row = results.get("attrition", {}).get(split)
    if row:
        labels = ["Input", f'After {row["order"][0]}', f'After {row["order"][1]}']
        axis.bar(labels, [row["input_rows"], row["after_first"], row["after_second"]])
        axis.set(ylabel="Light curves", title=f"Cascade attrition — {split}")
    else:
        axis.text(.5, .5, "No CNN/cascade in this run", transform=axis.transAxes, ha="center")
    figure.tight_layout()
    return figure


def plot_timing(results, *, split="test"):
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 4))
    values = [row["timing_error_days"] * 1440 for row in results.get("event_details", [])
              if row["split"] == split and row["timing_error_days"] is not None]
    if values:
        axis.hist(values, bins=min(20, max(3, len(values))))
    else:
        axis.text(.5, .5, "No matched events", transform=axis.transAxes, ha="center")
    axis.set(xlabel="Detection − injection time (minutes)", ylabel="Matched events", title=f"Timing error — {split}")
    figure.tight_layout()
    return figure
