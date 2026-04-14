import argparse
from pathlib import Path

from tqdm.contrib.concurrent import process_map

from searchlight.public_candidate_exports import (
    build_output_name,
    render_candidate_task,
)
from searchlight.review_exports import (
    DEFAULT_RESULTS_PATH,
    DEFAULT_SUPPRESSION_PATH,
    load_lensing_results,
    load_suppression_list,
)

DEFAULT_OUTPUT_DIR = Path("data/zooniverse/top_resnet_examples")
DEFAULT_POSITIVE_LIMIT = 100


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export public example plots for volunteer training, using "
            "suppressed candidates as negatives and the top non-suppressed "
            "candidates as positives."
        )
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Directory containing *_resnet.csv result files.",
    )
    parser.add_argument(
        "--suppression-path",
        type=Path,
        default=DEFAULT_SUPPRESSION_PATH,
        help="Suppression list used to define negative examples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where images and the manifest CSV will be written.",
    )
    parser.add_argument(
        "--positive-limit",
        type=int,
        default=DEFAULT_POSITIVE_LIMIT,
        help="Number of highest-scoring non-suppressed examples to export.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Optional number of worker processes for rendering.",
    )
    return parser.parse_args()


def build_render_tasks(results, suppressed_paths, output_dir, positive_limit):
    suppressed_paths = set(suppressed_paths)
    negatives = results[results["path"].isin(suppressed_paths)].copy()
    positives = (
        results[~results["path"].isin(suppressed_paths)].head(positive_limit).copy()
    )
    render_tasks = []
    for row in positives.to_dict("records"):
        candidate_id, filename = build_output_name(row["path"])
        output_path = output_dir / "positive" / filename

        if not output_path.exists():
            render_tasks.append(
                {
                    "path": row["path"],
                    "output_path": str(output_path),
                }
            )

    for row in negatives.to_dict("records"):
        candidate_id, filename = build_output_name(row["path"])
        output_path = output_dir / "negative" / filename

        if not output_path.exists():
            render_tasks.append(
                {
                    "path": row["path"],
                    "output_path": str(output_path),
                }
            )

    return render_tasks, len(positives), len(negatives)


def export_examples(
    results_path, suppression_path, output_dir, positive_limit, max_workers=None
):
    results = load_lensing_results(results_path=results_path)
    suppressed_paths = load_suppression_list(suppression_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    render_tasks, positive_count, negative_count = build_render_tasks(
        results, suppressed_paths, output_dir, positive_limit
    )

    print(
        f"Exporting {positive_count + negative_count} examples to {output_dir} "
        f"({positive_count} positive, {negative_count} negative)"
    )

    if render_tasks:
        process_map(
            render_candidate_task,
            render_tasks,
            max_workers=max_workers,
            chunksize=10,
            desc="Rendering example images",
        )
    else:
        print("All example images already exist; nothing to render.")


def main():
    args = parse_args()
    export_examples(
        results_path=args.results_path,
        suppression_path=args.suppression_path,
        output_dir=args.output_dir,
        positive_limit=args.positive_limit,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
