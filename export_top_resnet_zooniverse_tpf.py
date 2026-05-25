import argparse
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas
from tqdm.auto import tqdm

warnings.filterwarnings(
    "ignore",
    message=r"Warning: the tpfmodel submodule is not available.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*cadences will be ignored due to the quality mask.*",
)

from searchlight.public_candidate_exports import (
    build_output_name,
    build_output_relative_path,
    render_candidate_tpf_task,
    DEFAULT_TPF_DOWNLOAD_DIR,
)
from searchlight.review_exports import (
    DEFAULT_RESULTS_PATH,
    load_lensing_results,
)
from searchlight.zooniverse_results import (
    DEFAULT_MIN_AGREEMENT,
    DEFAULT_MIN_CLASSIFICATIONS,
    DEFAULT_SUBJECT_METADATA_CACHE_PATH,
    DEFAULT_TASK_KEY,
    load_answer_results_from_csv,
)

DEFAULT_OUTPUT_DIR = Path("data/zooniverse/top_resnet_candidates_tpf")
DEFAULT_MANIFEST_NAME = "manifest.csv"
DEFAULT_RENDER_FAILURES_NAME = "render_failures.csv"

MAX_MANIFEST_ROWS = 100_000
MANIFEST_COLUMNS = [
    "filename",
    "candidate_id",
    "subject_id",
    "most_likely_answer",
    "num_votes",
    "agreement",
    "split_size",
    "survey",
    "simulens_score",
]
RENDER_FAILURE_COLUMNS = MANIFEST_COLUMNS + ["error_type", "error"]


class RenderInterrupted(Exception):
    def __init__(self, render_results):
        super().__init__("Rendering interrupted")
        self.render_results = render_results


def quiet_render_worker_output():
    """Keep noisy library progress messages from trampling tqdm output."""
    for logger_name in (
        "lightkurve",
        "lightkurve.utils",
        "searchlight.public_candidate_exports",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


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
        "--min-classifications",
        type=int,
        default=DEFAULT_MIN_CLASSIFICATIONS,
        help="Minimum number of Zooniverse classifications required for a row.",
    )
    parser.add_argument(
        "--min-agreement",
        type=float,
        default=DEFAULT_MIN_AGREEMENT,
        help="Minimum Zooniverse agreement required for a row.",
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
    expected_paths = set(manifest_paths)
    stale_paths = {
        manifest_path,
        *manifest_path.parent.glob(f"{manifest_path.stem}_*{manifest_path.suffix}"),
    }

    for stale_path in stale_paths - expected_paths:
        stale_path.unlink(missing_ok=True)

    for chunk_index, chunk_path in enumerate(manifest_paths):
        start = chunk_index * max_manifest_rows
        end = start + max_manifest_rows
        manifest = pandas.DataFrame(
            manifest_rows[start:end],
            columns=MANIFEST_COLUMNS,
        )
        temp_path = chunk_path.with_suffix(f"{chunk_path.suffix}.tmp")
        manifest.to_csv(temp_path, index=False)
        temp_path.replace(chunk_path)
        print(f"Wrote manifest to {chunk_path}")


def render_candidate_tpf_task_safely(task):
    manifest_row = task.get("manifest_row", {})
    quiet_render_worker_output()
    try:
        render_candidate_tpf_task(task)
    except Exception as exc:
        return {
            "ok": False,
            "manifest_row": manifest_row,
            "candidate_id": task.get("candidate_id"),
            "path": task.get("path"),
            "output_path": task.get("output_path"),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    return {
        "ok": True,
        "manifest_row": manifest_row,
        "candidate_id": task.get("candidate_id"),
        "path": task.get("path"),
        "output_path": task.get("output_path"),
        "error_type": "",
        "error": "",
    }


def write_render_failure_summary(render_results, failure_path):
    failures = [result for result in render_results if not result["ok"]]
    if failures:
        write_render_failures(failures, failure_path)
    print_render_failure_summary(failures, failure_path)


def write_render_failures(failures, failure_path):
    failure_rows = [
        {
            **failure.get("manifest_row", {}),
            "error_type": failure["error_type"],
            "error": failure["error"],
        }
        for failure in failures
    ]
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    pandas.DataFrame(failure_rows, columns=RENDER_FAILURE_COLUMNS).to_csv(
        failure_path,
        index=False,
    )


def print_render_failure_summary(failures, failure_path, interrupted=False):
    if not failures:
        if interrupted:
            print("Rendering interrupted; no failures were recorded before stopping.")
        else:
            print("Rendered all missing images successfully.")
        return

    prefix = "Rendering interrupted; recorded" if interrupted else "Failed to render"
    print(f"{prefix} {len(failures)} failures; wrote details to {failure_path}")
    for failure in failures:
        print(
            f"- {failure['candidate_id']}: {failure['error_type']}: "
            f"{failure['error']}"
        )


def remove_failed_manifest_rows(manifest_rows, render_results):
    failed_candidate_ids = {
        result["candidate_id"] for result in render_results if not result["ok"]
    }
    if not failed_candidate_ids:
        return manifest_rows

    return [
        row for row in manifest_rows if row["candidate_id"] not in failed_candidate_ids
    ]


def collect_render_results(render_tasks, failure_path, max_workers=None):
    failure_path.unlink(missing_ok=True)
    render_results = []
    executor = ProcessPoolExecutor(max_workers=max_workers)
    future_tasks = {
        executor.submit(render_candidate_tpf_task_safely, task): task
        for task in render_tasks
    }

    try:
        with tqdm(
            total=len(render_tasks),
            desc="Rendering images",
            unit="image",
        ) as progress:
            for future in as_completed(future_tasks):
                task = future_tasks[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "ok": False,
                        "manifest_row": task.get("manifest_row", {}),
                        "candidate_id": task.get("candidate_id"),
                        "path": task.get("path"),
                        "output_path": task.get("output_path"),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }

                render_results.append(result)
                failures = [result for result in render_results if not result["ok"]]
                if failures:
                    write_render_failures(failures, failure_path)
                progress.update()
    except KeyboardInterrupt:
        for future in future_tasks:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        failures = [result for result in render_results if not result["ok"]]
        if failures:
            write_render_failures(failures, failure_path)
        print_render_failure_summary(failures, failure_path, interrupted=True)
        raise RenderInterrupted(render_results) from None
    else:
        executor.shutdown()
        return render_results


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
    min_classifications=DEFAULT_MIN_CLASSIFICATIONS,
    min_agreement=DEFAULT_MIN_AGREEMENT,
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
        min_classifications=min_classifications,
        min_agreement=min_agreement,
        subject_metadata_cache_path=subject_metadata_cache_path,
    )
    if limit is not None:
        results = results.head(limit).copy()

    image_dir = output_dir
    manifest_path = output_dir / DEFAULT_MANIFEST_NAME
    failure_path = output_dir / DEFAULT_RENDER_FAILURES_NAME
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
        manifest_row = {
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

        if not output_path.exists():
            render_tasks.append(
                {
                    "path": row["path"],
                    "output_path": str(output_path),
                    "tpf_download_dir": str(tpf_download_dir),
                    "candidate_id": candidate_id,
                    "manifest_row": manifest_row,
                }
            )

        manifest_rows.append(manifest_row)

    failure_path.unlink(missing_ok=True)

    if render_tasks:
        try:
            render_results = collect_render_results(
                render_tasks,
                failure_path,
                max_workers=max_workers,
            )
        except RenderInterrupted as exc:
            render_results = exc.render_results
            manifest_rows = remove_failed_manifest_rows(manifest_rows, render_results)
            write_manifests(manifest_rows, manifest_path)
            raise SystemExit(130) from None

        manifest_rows = remove_failed_manifest_rows(manifest_rows, render_results)
        write_manifests(manifest_rows, manifest_path)
        write_render_failure_summary(render_results, failure_path)
    else:
        write_manifests(manifest_rows, manifest_path)
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
        min_classifications=args.min_classifications,
        min_agreement=args.min_agreement,
        limit=args.limit,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
