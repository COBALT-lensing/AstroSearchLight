import argparse
from pathlib import Path

import pandas
from tqdm.contrib.concurrent import process_map

from searchlight.public_candidate_exports import (
    build_output_name,
    build_output_relative_path,
    render_candidate_task,
)
from searchlight.review_exports import (
    DEFAULT_RESULTS_PATH,
    load_lensing_results,
)

DEFAULT_OUTPUT_DIR = Path("data/zooniverse/top_resnet_candidates")
DEFAULT_MANIFEST_NAME = "manifest.csv"


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
        candidate_id, filename = build_output_name(row["path"])
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
