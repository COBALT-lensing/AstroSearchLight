"""Keep all versioned artifact reads/writes isolated from the user's library."""
import pytest

from searchlight import paths


@pytest.fixture(autouse=True)
def artifact_store(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_PATH", tmp_path / "artifacts")
