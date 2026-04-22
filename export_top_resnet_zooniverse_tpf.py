import argparse
from pathlib import Path

import pandas
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from searchlight.public_candidate_exports import (
    build_output_name,
    build_output_relative_path,
    render_candidate_tpf_task,
)
from searchlight.review_exports import (
    DEFAULT_RESULTS_PATH,
    load_lensing_results,
)
from searchlight.zooniverse_results import (
    DEFAULT_SUBJECT_METADATA_CACHE_PATH,
    DEFAULT_TASK_KEY,
    load_answer_results_from_csv,
)

DEFAULT_OUTPUT_DIR = Path("data/zooniverse/top_resnet_candidates_tpf")
DEFAULT_MANIFEST_NAME = "manifest.csv"
DEFAULT_TPF_DOWNLOAD_DIR = Path("data/cache/lightkurve")
MAX_MANIFEST_ROWS = 100_000


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export Zooniverse candidate images with whole light curves and "
            "TESS target-pixel heatmap insets for subjects matching a "
            "Zooniverse workflow answer."
        )
    )
    parser.add_argument(
        "--zooniverse-results-csv",
        type=Path,
        required=True,
        help=(
            "Zooniverse aggregation CSV containing subject_id, workflow_id, "
            "data.most_likely, data.num_votes, and data.agreement columns."
        ),
    )
    parser.add_argument(
        "--answer",
        required=True,
        help="Workflow answer string to export, resolved from the task answers.",
    )
    parser.add_argument(
        "--workflow-id",
        type=int,
        default=None,
        help=(
            "Optional workflow ID. Required if the Zooniverse CSV does not "
            "contain exactly one workflow_id value."
        ),
    )
    parser.add_argument(
        "--task-key",
        default=DEFAULT_TASK_KEY,
        help="Workflow task key whose answers are indexed by data.most_likely.",
    )
    parser.add_argument(
        "--subject-metadata-cache-path",
        type=Path,
        default=DEFAULT_SUBJECT_METADATA_CACHE_PATH,
        help="CSV cache mapping integer Zooniverse subject IDs to candidate IDs.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional extra filter for rows with lensing score strictly above this value.",
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
        "--tpf-download-dir",
        type=Path,
        default=DEFAULT_TPF_DOWNLOAD_DIR,
        help="Directory used by lightkurve to cache downloaded target pixel files.",
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
    chunk_count = max(
        1, (len(manifest_rows) + max_manifest_rows - 1) // max_manifest_rows
    )
    manifest_paths = build_manifest_paths(manifest_path, chunk_count)

    for chunk_index, chunk_path in enumerate(manifest_paths):
        start = chunk_index * max_manifest_rows
        end = start + max_manifest_rows
        manifest = pandas.DataFrame(
            manifest_rows[start:end],
            columns=[
                "filename",
                "candidate_id",
                "subject_id",
                "most_likely_answer",
                "num_votes",
                "agreement",
                "split_size",
                "survey",
                "simulens_score",
            ],
        )
        temp_path = chunk_path.with_suffix(f"{chunk_path.suffix}.tmp")
        manifest.to_csv(temp_path, index=False)
        temp_path.replace(chunk_path)
        print(f"Wrote manifest to {chunk_path}")


def export_candidates_with_tpf(
    zooniverse_results_csv,
    answer,
    results_path,
    output_dir,
    tpf_download_dir,
    workflow_id=None,
    task_key=DEFAULT_TASK_KEY,
    subject_metadata_cache_path=DEFAULT_SUBJECT_METADATA_CACHE_PATH,
    threshold=None,
    limit=None,
    max_workers=None,
):
    results = load_lensing_results(results_path=results_path, min_score=threshold)
    results, workflow_id, _answer_labels = load_answer_results_from_csv(
        model_results=results,
        zooniverse_results_csv=zooniverse_results_csv,
        answer=answer,
        workflow_id=workflow_id,
        task_key=task_key,
        subject_metadata_cache_path=subject_metadata_cache_path,
    )
    if limit is not None:
        results = results.head(limit).copy()

    image_dir = output_dir
    manifest_path = output_dir / DEFAULT_MANIFEST_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    tpf_download_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    render_tasks = []
    total = len(results)
    print(
        f"Exporting {total} candidates to {image_dir} "
        f"for workflow {workflow_id} answer {answer!r}"
    )

    for row in tqdm(
        results.to_dict("records"),
        desc="Preparing candidates",
        unit="candidate",
    ):
        candidate_id, filename = build_output_name(row["path"])
        relative_path = build_output_relative_path(filename)
        output_path = image_dir / relative_path

        if not output_path.exists():
            render_tasks.append(
                {
                    "path": row["path"],
                    "output_path": str(output_path),
                    "tpf_download_dir": str(tpf_download_dir),
                }
            )

        manifest_rows.append(
            {
                "filename": str(relative_path),
                "candidate_id": candidate_id,
                "subject_id": row["subject_id"],
                "most_likely_answer": row["most_likely_answer"],
                "num_votes": row["data.num_votes"],
                "agreement": row["data.agreement"],
                "split_size": "infinite",
                "survey": "TESS",
                "simulens_score": float(row["lensing"]),
            }
        )

    write_manifests(manifest_rows, manifest_path)

    if render_tasks:
        process_map(
            render_candidate_tpf_task,
            render_tasks,
            max_workers=max_workers,
            chunksize=1,
            desc="Rendering images",
        )
    else:
        print("All images already exist; nothing to render.")


def main():
    args = parse_args()
    export_candidates_with_tpf(
        zooniverse_results_csv=args.zooniverse_results_csv,
        answer=args.answer,
        results_path=args.results_path,
        output_dir=args.output_dir,
        tpf_download_dir=args.tpf_download_dir,
        workflow_id=args.workflow_id,
        task_key=args.task_key,
        subject_metadata_cache_path=args.subject_metadata_cache_path,
        threshold=args.threshold,
        limit=args.limit,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
