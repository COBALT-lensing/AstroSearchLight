import uuid
from pathlib import Path

import numpy
from matplotlib import pyplot

from .lightcurves import load

ZOONIVERSE_UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "black-hole-hunters.org")


def flux_to_numpy(flux):
    if hasattr(flux, "value"):
        flux = flux.value
    return numpy.asarray(flux, dtype=float)


def build_candidate_id(path):
    return str(uuid.uuid5(ZOONIVERSE_UUID_NAMESPACE, str(path)))


def build_output_name(path):
    candidate_id = build_candidate_id(path)
    return candidate_id, f"{candidate_id}.png"


def build_output_relative_path(filename):
    stem = filename.removesuffix(".png").replace("-", "")
    return Path(stem[:2]) / stem[2:4] / filename


def render_candidate_plot(path, output_path):
    lc = load(path)
    time_days = numpy.asarray(lc["time"].jd, dtype=float)
    time_days = time_days - time_days[0]
    brightness = flux_to_numpy(lc["flux"])

    finite = numpy.isfinite(time_days) & numpy.isfinite(brightness)
    time_days = time_days[finite]
    brightness = brightness[finite]

    if len(time_days) == 0:
        raise ValueError(f"No finite data points found in {path}")

    median_brightness = numpy.nanmedian(brightness)
    if numpy.isfinite(median_brightness) and median_brightness != 0:
        brightness = brightness / median_brightness

    fig, ax = pyplot.subplots(figsize=(10, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("#f7f4ea")
    ax.set_facecolor("#fffdf8")

    ax.scatter(
        time_days,
        brightness,
        s=9,
        color="#24445c",
        alpha=0.75,
        edgecolors="none",
        zorder=2,
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#9aa8b6")
    ax.spines["bottom"].set_color("#9aa8b6")
    ax.grid(axis="y", color="#d8dfdf", linewidth=0.8, alpha=0.8)

    ax.set_xlabel("Time (days)", color="#23313d")
    ax.set_ylabel("Brightness", color="#23313d")
    ax.tick_params(colors="#4c5b67")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    pyplot.close(fig)


def render_candidate_task(task):
    render_candidate_plot(task["path"], Path(task["output_path"]))
    return task["output_path"]
