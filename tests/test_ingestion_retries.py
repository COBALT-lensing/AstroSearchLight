"""Transient MAST failures retry without publishing partial or corrupt products."""
import errno
import http.client
import io
import json
import socket
import ssl
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import pytest

from searchlight.evaluation import ingestion

SOURCE = "mast:TESS/product/example.fits"


def dns_failure(code=socket.EAI_AGAIN):
    return URLError(socket.gaierror(code, "Temporary failure in name resolution"))


def test_temporary_dns_failure_retries_then_caches_verified_bytes(tmp_path, monkeypatch):
    failure = dns_failure()
    download = Mock(side_effect=[failure, failure, io.BytesIO(b"complete FITS bytes")])
    sleep = Mock()
    monkeypatch.setattr(ingestion.urllib.request, "urlopen", download)
    monkeypatch.setattr(ingestion.time, "sleep", sleep)
    path = ingestion.cache_product(SOURCE, tmp_path)
    assert path.read_bytes() == b"complete FITS bytes"
    assert download.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [1., 2.]
    assert all(call.kwargs["timeout"] == 120 for call in download.call_args_list)
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata == {"source": SOURCE, "sha256": ingestion.sha256(path)}
    assert not list(tmp_path.glob("*.partial"))
    assert ingestion.cache_product(SOURCE, tmp_path) == path
    assert download.call_count == 3


def test_failed_body_is_discarded_before_retry_and_success_starts_fresh(tmp_path, monkeypatch):
    class BrokenResponse(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise http.client.IncompleteRead(b"", 20)
            return super().read(size)
    download = Mock(side_effect=[BrokenResponse(b"truncated prefix"), io.BytesIO(b"good")])
    monkeypatch.setattr(ingestion.urllib.request, "urlopen", download)
    def sleep(delay):
        assert not list(tmp_path.iterdir())
    monkeypatch.setattr(ingestion.time, "sleep", sleep)
    path = ingestion.cache_product(SOURCE, tmp_path)
    assert path.read_bytes() == b"good"
    assert download.call_count == 2
    assert sorted(file.suffix for file in tmp_path.iterdir()) == [".fits", ".json"]


def test_retry_exhaustion_raises_original_error_without_cache_artifacts(tmp_path, monkeypatch):
    failure = dns_failure()
    download = Mock(side_effect=failure)
    sleep = Mock()
    monkeypatch.setattr(ingestion.urllib.request, "urlopen", download)
    monkeypatch.setattr(ingestion.time, "sleep", sleep)
    with pytest.raises(URLError) as caught:
        ingestion.cache_product(SOURCE, tmp_path)
    assert caught.value is failure
    assert download.call_count == 5
    assert [call.args[0] for call in sleep.call_args_list] == [1., 2., 4., 8.]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", [
    HTTPError("https://example.org", 404, "Not found", {}, None),
    HTTPError("https://example.org", 403, "Forbidden", {}, None),
    dns_failure(socket.EAI_NONAME),
    URLError(ssl.SSLCertVerificationError("certificate verify failed")),
    OSError(errno.ENOSPC, "No space left on device"),
])
def test_permanent_failures_are_not_retried(tmp_path, monkeypatch, failure):
    download = Mock(side_effect=failure)
    sleep = Mock()
    monkeypatch.setattr(ingestion.urllib.request, "urlopen", download)
    monkeypatch.setattr(ingestion.time, "sleep", sleep)
    with pytest.raises(type(failure)):
        ingestion.cache_product(SOURCE, tmp_path)
    download.assert_called_once()
    sleep.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", [
    HTTPError("https://example.org", 429, "Too many requests", {}, None),
    HTTPError("https://example.org", 503, "Unavailable", {}, None),
    URLError(TimeoutError("request timed out")),
    TimeoutError("socket timed out"),
    ConnectionResetError("connection reset"),
    http.client.RemoteDisconnected("remote server closed connection"),
])
def test_other_transient_network_failures_are_retried(tmp_path, monkeypatch, failure):
    download = Mock(side_effect=[failure, io.BytesIO(b"recovered")])
    monkeypatch.setattr(ingestion.urllib.request, "urlopen", download)
    monkeypatch.setattr(ingestion.time, "sleep", Mock())
    assert ingestion.cache_product(SOURCE, tmp_path).read_bytes() == b"recovered"
    assert download.call_count == 2


def test_local_file_errors_are_not_retried(tmp_path, monkeypatch):
    download = Mock()
    sleep = Mock()
    monkeypatch.setattr(ingestion.urllib.request, "urlopen", download)
    monkeypatch.setattr(ingestion.time, "sleep", sleep)
    with pytest.raises(FileNotFoundError):
        ingestion.cache_product(tmp_path / "missing.fits", tmp_path / "cache")
    download.assert_not_called()
    sleep.assert_not_called()
    assert not list((tmp_path / "cache").iterdir())
