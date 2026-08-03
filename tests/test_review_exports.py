from types import SimpleNamespace

import pandas as pd
import pytest

from searchlight import review_exports


def test_result_file_signature_is_sorted_and_includes_file_stats(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    b.write_text("b")
    a.write_text("a")

    signature = review_exports.result_file_signature([b, a])

    assert [item["path"] for item in signature] == [str(a), str(b)]
    assert [item["size"] for item in signature] == [1, 1]
    assert all("mtime_ns" in item for item in signature)


def test_load_result_frames_reads_and_caches_csvs(tmp_path, monkeypatch):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text("path,label,lensing\np1,lensing,0.9\n")
    second.write_text("path,label,lensing\np2,not-lensing,0.1\n")
    cache_path = tmp_path / "cache" / "results.pkl"
    cache_meta_path = tmp_path / "cache" / "results.json"

    frame = review_exports.load_result_frames(
        [second, first],
        cache_path=cache_path,
        cache_meta_path=cache_meta_path,
    )

    assert list(frame["path"]) == ["p2", "p1"]
    assert cache_path.exists()
    assert cache_meta_path.exists()

    monkeypatch.setattr(
        review_exports.pandas,
        "read_csv",
        lambda path: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    cached = review_exports.load_result_frames(
        [second, first],
        cache_path=cache_path,
        cache_meta_path=cache_meta_path,
    )
    assert list(cached["path"]) == ["p2", "p1"]


def test_load_lensing_results_filters_sorts_and_errors(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "one_resnet.csv").write_text(
        "path,label,lensing\n"
        "low,lensing,0.2\n"
        "high,lensing,0.9\n"
        "no,not-lensing,1.0\n"
    )

    frame = review_exports.load_lensing_results(
        results_path=results_dir,
        min_score=0.5,
        cache_path=tmp_path / "cache.pkl",
        cache_meta_path=tmp_path / "cache.json",
    )

    assert list(frame["path"]) == ["high"]

    with pytest.raises(FileNotFoundError, match="No result files"):
        review_exports.load_lensing_results(results_path=tmp_path / "missing")


def test_load_metadata_cache_returns_empty_or_backs_up_unreadable_cache(tmp_path):
    missing = review_exports.load_metadata_cache(tmp_path / "missing.pkl")
    assert list(missing.columns) == review_exports.METADATA_COLUMNS

    cache_path = tmp_path / "metadata.pkl"
    cache_path.write_text("not a pickle")
    loaded = review_exports.load_metadata_cache(cache_path)

    assert loaded.empty
    assert not cache_path.exists()
    assert (tmp_path / "metadata.unreadable.pkl").exists()


def test_backup_unreadable_cache_chooses_available_suffix(tmp_path):
    cache_path = tmp_path / "metadata.pkl"
    cache_path.write_text("bad")
    (tmp_path / "metadata.unreadable.pkl").write_text("old")

    backup_path = review_exports.backup_unreadable_cache(cache_path)

    assert backup_path == tmp_path / "metadata.unreadable-1.pkl"
    assert backup_path.read_text() == "bad"


def test_extract_path_metadata_builds_simbad_url(monkeypatch):
    lc = SimpleNamespace(meta={"TICID": 123, "SECTOR": 4, "RA": 10.5, "DEC": -20.25})
    monkeypatch.setattr(review_exports, "load", lambda path: lc)

    metadata = review_exports.extract_path_metadata("path.fits")

    assert metadata["path"] == "path.fits"
    assert metadata["tic"] == 123
    assert metadata["sector"] == 4
    assert metadata["simbad_url"].endswith("Coord=10.5%20-20.25")


def test_enrich_results_with_metadata_uses_cache_and_adds_missing(monkeypatch, tmp_path):
    results = pd.DataFrame({"path": ["cached", "new", "new"], "score": [1, 2, 3]})
    cache_path = tmp_path / "metadata.pkl"
    pd.DataFrame(
        [
            {
                "path": "cached",
                "tic": 1,
                "sector": 2,
                "simbad_url": "cached-url",
            }
        ]
    ).to_pickle(cache_path)
    monkeypatch.setattr(
        review_exports,
        "extract_path_metadata",
        lambda path: {
            "path": path,
            "tic": 10,
            "sector": 20,
            "simbad_url": "new-url",
        },
    )

    enriched = review_exports.enrich_results_with_metadata(results, cache_path)

    assert list(enriched["tic"]) == [1, 10, 10]
    cache = pd.read_pickle(cache_path)
    assert set(cache["path"]) == {"cached", "new"}


def test_load_suppression_list_ignores_missing_and_blank_lines(tmp_path):
    assert review_exports.load_suppression_list(tmp_path / "missing.txt") == set()

    path = tmp_path / "suppressions.txt"
    path.write_text("\n a \n\nb\n")
    assert review_exports.load_suppression_list(path) == {"a", "b"}
