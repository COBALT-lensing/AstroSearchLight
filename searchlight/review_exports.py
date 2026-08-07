import json
from pathlib import Path
from urllib.parse import quote

import pandas

from .lightcurves import load
from .paths import DATA_PATH

DEFAULT_RESULTS_PATH = DATA_PATH / "results" / "simulens1.0_tess_spoc"
DEFAULT_RESULT_GLOB = "*_resnet.csv"
DEFAULT_RESULT_CACHE_PATH = DEFAULT_RESULTS_PATH / "top_resnet_results_cache.pkl"
DEFAULT_RESULT_CACHE_META_PATH = (
    DEFAULT_RESULTS_PATH / "top_resnet_results_cache_meta.json"
)
DEFAULT_METADATA_CACHE_PATH = DEFAULT_RESULTS_PATH / "top_resnet_metadata_cache.pkl"
DEFAULT_SUPPRESSION_PATH = DEFAULT_RESULTS_PATH / "top_resnet_suppressions.txt"
METADATA_COLUMNS = ["path", "tic", "sector", "simbad_url"]

RESULT_COLUMNS = ["path", "label", "lensing"]
RESULT_DTYPES = {
    "path": "string",
    "label": "category",
    "lensing": "float32",
}


def result_file_signature(paths):
    return [
        {
            "path": str(path),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(paths)
    ]


def load_result_frames(
    paths,
    columns=RESULT_COLUMNS,
    dtypes=RESULT_DTYPES,
    cache_path=DEFAULT_RESULT_CACHE_PATH,
    cache_meta_path=DEFAULT_RESULT_CACHE_META_PATH,
):
    signature = result_file_signature(paths)
    if cache_path.exists() and cache_meta_path.exists():
        cached_signature = json.loads(cache_meta_path.read_text())
        if cached_signature == signature:
            print(f"Loading cached results from {cache_path}")
            return pandas.read_pickle(cache_path)

    print(f"Reading {len(paths)} CSV files from disk...")
    frame = pandas.concat(
        [pandas.read_csv(path, usecols=columns, dtype=dtypes) for path in paths],
        ignore_index=True,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(cache_path)
    cache_meta_path.write_text(json.dumps(signature, indent=2))
    print(f"Cached merged results to {cache_path}")
    return frame


def load_lensing_results(
    results_path=DEFAULT_RESULTS_PATH,
    result_glob=DEFAULT_RESULT_GLOB,
    min_score=None,
    cache_path=DEFAULT_RESULT_CACHE_PATH,
    cache_meta_path=DEFAULT_RESULT_CACHE_META_PATH,
):
    result_paths = sorted(results_path.glob(result_glob))
    if not result_paths:
        raise FileNotFoundError(
            f"No result files matching {result_glob!r} were found in {results_path}"
        )

    results = load_result_frames(
        result_paths,
        cache_path=cache_path,
        cache_meta_path=cache_meta_path,
    )
    results = results[results["label"] == "lensing"].copy()

    if min_score is not None:
        results = results[results["lensing"] > min_score].copy()

    results.sort_values("lensing", inplace=True, ascending=False)
    results.reset_index(drop=True, inplace=True)
    return results


def load_metadata_cache(cache_path=DEFAULT_METADATA_CACHE_PATH):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return pandas.DataFrame(columns=METADATA_COLUMNS)

    try:
        return pandas.read_pickle(cache_path)
    except Exception as exc:
        backup_path = backup_unreadable_cache(cache_path)
        print(
            f"Ignoring unreadable metadata cache {cache_path}: {exc!r}. "
            f"Moved it to {backup_path}."
        )
        return pandas.DataFrame(columns=METADATA_COLUMNS)


def backup_unreadable_cache(cache_path):
    backup_path = cache_path.with_name(
        f"{cache_path.stem}.unreadable{cache_path.suffix}"
    )
    index = 1
    while backup_path.exists():
        backup_path = cache_path.with_name(
            f"{cache_path.stem}.unreadable-{index}{cache_path.suffix}"
        )
        index += 1

    cache_path.rename(backup_path)
    return backup_path


def extract_path_metadata(path):
    lc = load(path)
    ticid = lc.meta.get("TICID") or lc.meta.get("TARGETID")
    sector = lc.meta.get("SECTOR")
    ra = lc.meta.get("RA")
    dec = lc.meta.get("DEC")
    simbad_url = None
    if ra is not None and dec is not None:
        simbad_url = (
            "https://simbad.cds.unistra.fr/simbad/sim-coo?Coord="
            + quote(f"{ra} {dec}")
        )
    return {
        "path": path,
        "tic": ticid,
        "sector": sector,
        "simbad_url": simbad_url,
    }


def enrich_results_with_metadata(results, cache_path=DEFAULT_METADATA_CACHE_PATH):
    metadata_cache = load_metadata_cache(cache_path)
    cached_paths = set(metadata_cache["path"]) if not metadata_cache.empty else set()
    unique_paths = pandas.Index(results["path"].dropna().unique())
    missing_paths = [path for path in unique_paths if path not in cached_paths]

    if missing_paths:
        print(f"Loading metadata for {len(missing_paths)} new paths...")
        new_rows = [extract_path_metadata(path) for path in missing_paths]
        metadata_cache = pandas.concat(
            [metadata_cache, pandas.DataFrame(new_rows)],
            ignore_index=True,
        )
        metadata_cache = metadata_cache.drop_duplicates(subset="path", keep="last")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_cache.to_pickle(cache_path)
        print(f"Updated metadata cache: {cache_path}")

    return results.merge(
        metadata_cache[METADATA_COLUMNS],
        on="path",
        how="left",
    )


def load_suppression_list(path=DEFAULT_SUPPRESSION_PATH):
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    }
