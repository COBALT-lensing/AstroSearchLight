import importlib

import searchlight.utils as utils


def test_cache_returns_cached_value_for_hashable_args(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "FUNC_CACHE_PATH", str(tmp_path))
    monkeypatch.setattr(utils, "CACHE_ENABLED", True)
    calls = []

    @utils.cache
    def compute(value):
        calls.append(value)
        return {"value": value}

    assert compute("a") == {"value": "a"}
    assert compute("a") == {"value": "a"}
    assert calls == ["a"]


def test_cache_ignores_unhashable_args_when_building_key(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "FUNC_CACHE_PATH", str(tmp_path))
    monkeypatch.setattr(utils, "CACHE_ENABLED", True)
    calls = []

    @utils.cache
    def compute(values):
        calls.append(list(values))
        return len(calls)

    assert compute(["a"]) == 1
    assert compute(["b"]) == 1
    assert calls == [["a"]]


def test_cache_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "FUNC_CACHE_PATH", str(tmp_path))
    monkeypatch.setattr(utils, "CACHE_ENABLED", False)
    calls = []

    @utils.cache
    def compute(value):
        calls.append(value)
        return len(calls)

    assert compute("a") == 1
    assert compute("a") == 2


def test_cache_enabled_reads_environment_at_import(monkeypatch):
    monkeypatch.setenv("CACHE_ENABLED", "false")
    reloaded = importlib.reload(utils)

    try:
        assert reloaded.CACHE_ENABLED is False
    finally:
        monkeypatch.delenv("CACHE_ENABLED", raising=False)
        importlib.reload(utils)
