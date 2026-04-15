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
MAX_MANIFEST_ROWS = 100_000


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


def build_manifest_paths(manifest_path, chunk_count):
    if chunk_count == 1:
        return [manifest_path]

    suffix = manifest_path.suffix
    stem = manifest_path.stem
    return [
        manifest_path.with_name(f"{stem}_{chunk_index:04d}{suffix}")
        for chunk_index in range(1, chunk_count + 1)
    ]


def write_manifests(manifest_rows, manifest_path, max_manifest_rows=MAX_MANIFEST_ROWS):
    chunk_count = max(1, (len(manifest_rows) + max_manifest_rows - 1) // max_manifest_rows)
    manifest_paths = build_manifest_paths(manifest_path, chunk_count)

    for chunk_index, chunk_path in enumerate(manifest_paths):
        start = chunk_index * max_manifest_rows
        end = start + max_manifest_rows
        manifest = pandas.DataFrame(
            manifest_rows[start:end],
            columns=[
                "filename",
                "candidate_id",
                "split_size",
                "survey",
                "simulens_score",
            ],
        )
        temp_path = chunk_path.with_suffix(f"{chunk_path.suffix}.tmp")
        manifest.to_csv(temp_path, index=False)
        temp_path.replace(chunk_path)
        print(f"Wrote manifest to {chunk_path}")


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

    write_manifests(manifest_rows, manifest_path)

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
