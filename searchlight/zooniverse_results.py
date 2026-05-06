from pathlib import Path

import pandas
from panoptes_client import Subject, Workflow
from tqdm.auto import tqdm

from .public_candidate_exports import build_candidate_id

DEFAULT_SUBJECT_METADATA_CACHE_PATH = Path("data/zooniverse/subject_metadata_cache.csv")
DEFAULT_TASK_KEY = "T0"
REQUIRED_ZOONIVERSE_COLUMNS = {
    "subject_id",
    "data.most_likely",
    "data.num_votes",
    "data.agreement",
}


def normalize_identifier(value):
    if pandas.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def normalize_subject_id(value):
    text = normalize_identifier(value)
    if text is None:
        return None
    try:
        return str(int(text))
    except ValueError:
        return str(int(float(text)))


def normalize_answer_index(value):
    text = normalize_identifier(value)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return int(float(text))


def add_candidate_ids(results):
    results = results.copy()
    results["candidate_id"] = results["path"].map(build_candidate_id)
    return results


def resolve_workflow_id(zooniverse_results, workflow_id=None):
    if workflow_id is not None:
        return int(workflow_id)
    if "workflow_id" not in zooniverse_results.columns:
        raise ValueError(
            "Set workflow_id because the CSV does not contain a workflow_id column"
        )

    workflow_ids = zooniverse_results["workflow_id"].dropna().astype(int).unique()
    if len(workflow_ids) != 1:
        raise ValueError(
            f"Set workflow_id explicitly; found workflow ids: {workflow_ids}"
        )
    return int(workflow_ids[0])


def extract_answer_label(answer):
    if isinstance(answer, str):
        return answer
    if not isinstance(answer, dict):
        return str(answer)

    for key in ("label", "text", "value"):
        value = normalize_identifier(answer.get(key))
        if value is not None:
            return value
    return str(answer)


def load_workflow_answer_labels(workflow_id, task_key=DEFAULT_TASK_KEY):
    workflow = Workflow.find(int(workflow_id))
    task = workflow.tasks.get(task_key)
    if task is None:
        raise KeyError(f"Workflow {workflow_id} has no task {task_key!r}")

    answers = task.get("answers")
    if answers is None:
        raise KeyError(f"Workflow {workflow_id} task {task_key!r} has no answers")

    labels = {
        index: extract_answer_label(answer) for index, answer in enumerate(answers)
    }
    if not labels:
        raise ValueError(
            f"Workflow {workflow_id} task {task_key!r} has no answer labels"
        )
    return labels


def load_subject_metadata_cache(cache_path=DEFAULT_SUBJECT_METADATA_CACHE_PATH):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return pandas.DataFrame(columns=["subject_id", "candidate_id"])

    cache = pandas.read_csv(
        cache_path,
        dtype={"subject_id": "string", "candidate_id": "string"},
    )
    cache["subject_id"] = cache["subject_id"].map(normalize_subject_id)
    cache["candidate_id"] = cache["candidate_id"].map(normalize_identifier)
    return cache.dropna(subset=["subject_id", "candidate_id"])


def fetch_subject_candidate_id(subject_id):
    subject = Subject.find(int(subject_id))
    metadata = dict(subject.metadata or {})
    candidate_id = normalize_identifier(metadata.get("candidate_id"))
    if candidate_id is None:
        raise KeyError(f"Zooniverse subject {subject_id} has no candidate_id metadata")
    return candidate_id


def load_subject_candidate_ids(
    subject_ids,
    cache_path=DEFAULT_SUBJECT_METADATA_CACHE_PATH,
):
    subject_ids = [normalize_subject_id(subject_id) for subject_id in subject_ids]
    subject_ids = sorted(
        {subject_id for subject_id in subject_ids if subject_id is not None}
    )

    cache = load_subject_metadata_cache(cache_path)
    cached = dict(cache[["subject_id", "candidate_id"]].values)
    missing_subject_ids = [
        subject_id for subject_id in subject_ids if subject_id not in cached
    ]

    new_rows = []
    for subject_id in tqdm(
        missing_subject_ids,
        desc="Fetching Zooniverse subjects",
        unit="subject",
    ):
        candidate_id = fetch_subject_candidate_id(subject_id)
        cached[subject_id] = candidate_id
        new_rows.append({"subject_id": subject_id, "candidate_id": candidate_id})

    if new_rows:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        updated_cache = pandas.concat(
            [cache, pandas.DataFrame(new_rows)],
            ignore_index=True,
        )
        updated_cache = updated_cache.drop_duplicates(
            subset="subject_id",
            keep="last",
        )
        updated_cache.sort_values("subject_id").to_csv(cache_path, index=False)
        print(f"Cached {len(updated_cache)} subject metadata rows at {cache_path}")

    return cached


def load_zooniverse_results(results_csv):
    results_csv = Path(results_csv)
    if not results_csv.exists():
        raise FileNotFoundError(f"CSV file does not exist: {results_csv}")

    zooniverse = pandas.read_csv(results_csv, dtype={"subject_id": "string"})
    missing_columns = REQUIRED_ZOONIVERSE_COLUMNS - set(zooniverse.columns)
    if missing_columns:
        raise KeyError(
            f"Missing required columns from {results_csv}: {sorted(missing_columns)}"
        )

    zooniverse["subject_id"] = zooniverse["subject_id"].map(normalize_subject_id)
    zooniverse["data.most_likely"] = zooniverse["data.most_likely"].map(
        normalize_answer_index
    )
    zooniverse["data.num_votes"] = pandas.to_numeric(
        zooniverse["data.num_votes"],
        errors="coerce",
    )
    zooniverse["data.agreement"] = pandas.to_numeric(
        zooniverse["data.agreement"],
        errors="coerce",
    )
    return zooniverse


def annotate_most_likely_answers(
    zooniverse,
    workflow_id=None,
    task_key=DEFAULT_TASK_KEY,
):
    workflow_id = resolve_workflow_id(zooniverse, workflow_id=workflow_id)
    answer_labels = load_workflow_answer_labels(workflow_id, task_key=task_key)

    zooniverse = zooniverse.copy()
    zooniverse["most_likely_answer"] = zooniverse["data.most_likely"].map(answer_labels)
    print(f"Workflow {workflow_id} {task_key} answers: {answer_labels}")
    return zooniverse, workflow_id, answer_labels


def select_zooniverse_answer_results(
    model_results,
    zooniverse,
    answer,
    min_classifications,
    min_agreement,
    subject_metadata_cache_path=DEFAULT_SUBJECT_METADATA_CACHE_PATH,
):
    model_results = add_candidate_ids(model_results)
    candidate_path_map = dict(model_results[["candidate_id", "path"]].values)

    selected = zooniverse[
        (zooniverse["most_likely_answer"].str.casefold() == str(answer).casefold())
        & (zooniverse["data.num_votes"] >= min_classifications)
        & (zooniverse["data.agreement"] >= min_agreement)
    ].copy()

    subject_candidate_map = load_subject_candidate_ids(
        selected["subject_id"],
        cache_path=subject_metadata_cache_path,
    )
    selected["candidate_id"] = selected["subject_id"].map(subject_candidate_map)
    selected["path"] = selected["candidate_id"].map(candidate_path_map)
    missing_paths = selected["path"].isna()

    print(f"Selected {len(selected)} rows where most_likely_answer == {answer!r}")
    print(f"Mapped {len(selected) - missing_paths.sum()} selected subjects to paths")
    if missing_paths.any():
        print(
            f"Could not map {missing_paths.sum()} selected candidate_ids "
            "to local result paths"
        )

    selected = selected[~missing_paths].copy()
    selected = selected.merge(
        model_results[["path", "lensing"]],
        on="path",
        how="left",
    )
    selected.sort_values("lensing", inplace=True, ascending=False, na_position="last")
    selected.reset_index(drop=True, inplace=True)
    return selected


def select_zooniverse_subject_results(
    model_results,
    subject_ids,
    subject_metadata_cache_path=DEFAULT_SUBJECT_METADATA_CACHE_PATH,
):
    model_results = add_candidate_ids(model_results)
    candidate_path_map = dict(model_results[["candidate_id", "path"]].values)

    if isinstance(subject_ids, (str, int, float)):
        subject_ids = [subject_ids]

    rows = []
    seen_subject_ids = set()
    for index, subject_id in enumerate(subject_ids):
        normalized_subject_id = normalize_subject_id(subject_id)
        if normalized_subject_id is None or normalized_subject_id in seen_subject_ids:
            continue
        seen_subject_ids.add(normalized_subject_id)
        rows.append({"subject_order": index, "subject_id": normalized_subject_id})

    selected = pandas.DataFrame(rows)
    if selected.empty:
        return pandas.DataFrame(
            columns=[
                "subject_order",
                "subject_id",
                "candidate_id",
                "path",
                "data.num_votes",
                "data.agreement",
                "most_likely_answer",
                "lensing",
            ]
        )

    subject_candidate_map = load_subject_candidate_ids(
        selected["subject_id"],
        cache_path=subject_metadata_cache_path,
    )
    selected["candidate_id"] = selected["subject_id"].map(subject_candidate_map)
    selected["path"] = selected["candidate_id"].map(candidate_path_map)
    selected["data.num_votes"] = pandas.NA
    selected["data.agreement"] = pandas.NA
    selected["most_likely_answer"] = pandas.NA

    missing_paths = selected["path"].isna()
    print(f"Mapped {len(selected) - missing_paths.sum()} subjects to local paths")
    if missing_paths.any():
        print(
            f"Could not map {missing_paths.sum()} subject candidate_ids "
            "to local result paths"
        )

    selected = selected[~missing_paths].copy()
    selected = selected.merge(
        model_results[["path", "lensing"]],
        on="path",
        how="left",
    )
    selected.sort_values("subject_order", inplace=True)
    selected.reset_index(drop=True, inplace=True)
    return selected


def load_answer_results_from_csv(
    model_results,
    zooniverse_results_csv,
    answer,
    workflow_id=None,
    task_key=DEFAULT_TASK_KEY,
    subject_metadata_cache_path=DEFAULT_SUBJECT_METADATA_CACHE_PATH,
):
    zooniverse = load_zooniverse_results(zooniverse_results_csv)
    zooniverse, workflow_id, answer_labels = annotate_most_likely_answers(
        zooniverse,
        workflow_id=workflow_id,
        task_key=task_key,
    )
    selected = select_zooniverse_answer_results(
        model_results,
        zooniverse,
        answer,
        subject_metadata_cache_path=subject_metadata_cache_path,
    )
    return selected, workflow_id, answer_labels
