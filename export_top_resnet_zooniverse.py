import argparse
import uuid
from pathlib import Path

import numpy
import pandas
from matplotlib import pyplot
from tqdm.contrib.concurrent import process_map

from searchlight.lightcurves import load
from searchlight.review_exports import (
    DEFAULT_RESULTS_PATH,
    load_lensing_results,
)

DEFAULT_OUTPUT_DIR = Path("data/zooniverse/top_resnet_candidates")
DEFAULT_MANIFEST_NAME = "manifest.csv"
ZOONIVERSE_UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "black-hole-hunters.org")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export simple whole-light-curve images for high-scoring lensing "
            "candidates and write a manifest CSV for Zooniverse."
        )
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Only include rows with lensing score strictly above this value.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Directory containing *_resnet.csv result files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where images and the manifest CSV will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of candidates to export.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Optional number of worker processes for rendering.",
    )
    return parser.parse_args()


def flux_to_numpy(flux):
    if hasattr(flux, "value"):
        flux = flux.value
    return numpy.asarray(flux, dtype=float)


def build_output_name(row):
    candidate_id = str(uuid.uuid5(ZOONIVERSE_UUID_NAMESPACE, str(row["path"])))
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


def export_candidates(
    threshold, results_path, output_dir, limit=None, max_workers=None
):
    results = load_lensing_results(results_path=results_path, min_score=threshold)
    if limit is not None:
        results = results.head(limit).copy()

    image_dir = output_dir / "images"
    manifest_path = output_dir / DEFAULT_MANIFEST_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    render_tasks = []
    total = len(results)
    print(f"Exporting {total} candidates to {image_dir}")

    for row in results.to_dict("records"):
        candidate_id, filename = build_output_name(row)
        relative_path = build_output_relative_path(filename)
        output_path = image_dir / relative_path

        if not output_path.exists():
            render_tasks.append(
                {
                    "path": row["path"],
                    "output_path": str(output_path),
                }
            )

        manifest_rows.append(
            {
                "filename": str(relative_path),
                "candidate_id": candidate_id,
                "split_size": "infinite",
                "survey": "TESS",
                "simulens_score": float(row["lensing"]),
            }
        )

    if render_tasks:
        process_map(
            render_candidate_task,
            render_tasks,
            max_workers=max_workers,
            chunksize=10,
            desc="Rendering images",
        )
    else:
        print("All images already exist; nothing to render.")

    manifest = pandas.DataFrame(
        manifest_rows,
        columns=["filename", "candidate_id", "split_size", "survey", "simulens_score"],
    )
    manifest.to_csv(manifest_path, index=False)
    print(f"Wrote manifest to {manifest_path}")


def main():
    args = parse_args()
    export_candidates(
        threshold=args.threshold,
        results_path=args.results_path,
        output_dir=args.output_dir,
        limit=args.limit,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
