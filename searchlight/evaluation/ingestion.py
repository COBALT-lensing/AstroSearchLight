"""Cache exact SPOC files and prepare manifests without inventing labels."""

from dataclasses import replace
import errno
import hashlib
import http.client
import json
import logging
import math
from pathlib import Path
import shutil
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from .datasets import EvaluationManifest, EvaluationSample
from .preprocessing import PreprocessingConfig, read_lightcurve


logger = logging.getLogger(__name__)
DEFAULT_DOWNLOAD_ATTEMPTS = 5
DEFAULT_DOWNLOAD_RETRY_DELAY_SECONDS = 1.0
_RETRYABLE_HTTP_STATUS = {408, 429, 500, 502, 503, 504}
_RETRYABLE_NETWORK_ERRNOS = {errno.EAGAIN, errno.ETIMEDOUT, errno.ECONNRESET,
    errno.ECONNABORTED, errno.ECONNREFUSED, errno.ENETUNREACH, errno.EHOSTUNREACH,
    errno.ENETDOWN, errno.EPIPE}


def _retryable_download_error(error):
    if isinstance(error, urllib.error.HTTPError):
        return error.code in _RETRYABLE_HTTP_STATUS
    if isinstance(error, urllib.error.URLError):
        return _retryable_download_error(error.reason)
    if isinstance(error, socket.gaierror):
        return error.errno == socket.EAI_AGAIN  # Temporary DNS failure; not an unknown hostname.
    return (isinstance(error, (TimeoutError, ConnectionError, http.client.IncompleteRead))
            or isinstance(error, OSError) and error.errno in _RETRYABLE_NETWORK_ERRNOS)


def _download_product(url, temporary, *, max_attempts, retry_delay_seconds):
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            return
        except (urllib.error.URLError, OSError, http.client.HTTPException) as error:
            temporary.unlink(missing_ok=True)
            if isinstance(error, urllib.error.HTTPError):
                error.close()
            if attempt + 1 == max_attempts or not _retryable_download_error(error):
                raise
            delay = min(30., retry_delay_seconds * 2 ** attempt)
            logger.warning("Product download failed (%d/%d): %s; retrying in %gs",
                           attempt + 1, max_attempts, error, delay)
            time.sleep(delay)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_product(source, directory, *, max_attempts=DEFAULT_DOWNLOAD_ATTEMPTS,
                  retry_delay_seconds=DEFAULT_DOWNLOAD_RETRY_DELAY_SECONDS):
    """Verify cached product bytes; retry transient MAST transfer failures.

    Downloads get five attempts by default, with 1/2/4/8-second backoff. A failed
    transfer is discarded before retrying; local copies and corrupt cache hits
    are never retried. The final failure retains its original exception.
    """
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    if not math.isfinite(retry_delay_seconds) or retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must be finite and nonnegative")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(str(source).encode()).hexdigest()
    path, record = directory / f"{key}.fits", directory / f"{key}.json"
    if path.exists() and record.exists():
        metadata = json.loads(record.read_text())
        if metadata["source"] != str(source) or metadata["sha256"] != sha256(path):
            raise ValueError(f"Corrupt product cache: {path}")
        return path
    temporary = path.with_suffix(".partial")
    try:
        if str(source).startswith("mast:"):
            url = "https://mast.stsci.edu/api/v0.1/Download/file?uri=" + urllib.parse.quote(str(source), safe="")
            _download_product(url, temporary, max_attempts=max_attempts,
                              retry_delay_seconds=retry_delay_seconds)
        else:
            shutil.copyfile(source, temporary)
        checksum = sha256(temporary)
        temporary.replace(path)
        record.write_text(json.dumps({"source": str(source), "sha256": checksum}, indent=2) + "\n")
    finally:
        temporary.unlink(missing_ok=True)
    return path


def discover_spoc(tic, *, sector=None, cadence_seconds=None):
    from lightkurve import search_lightcurve
    kwargs = {"mission": "TESS", "author": "SPOC"}
    if sector is not None:
        kwargs["sector"] = sector
    if cadence_seconds is not None:
        kwargs["exptime"] = cadence_seconds
    result = search_lightcurve(f"TIC {tic}", **kwargs)
    return sorted(set(str(row["dataURI"] if "dataURI" in result.table.colnames else row["dataURL"])
                      for row in result.table))


def assign_splits(samples, *, seed=42, tuning_fraction=0.3, train_fraction=0., held_out_sectors=()):
    if not 0 <= train_fraction < 1 or not 0 < tuning_fraction < 1 or train_fraction + tuning_fraction >= 1:
        raise ValueError("Split fractions must leave nonempty tuning and test fractions")
    assignments = {}
    def identity(sample):
        return ("tic:" + str(int(str(sample.tic).upper().removeprefix("TIC").strip()))
                if sample.tic is not None else sample.group_id)
    heldout = {identity(s) for s in samples if s.sector in held_out_sectors}
    for sample in samples:
        target = identity(sample)
        if target in assignments:
            continue
        # Target identity determines assignment even when filenames/sectors change.
        key = f"{seed}:{target}".encode()
        value = int(hashlib.sha256(key).hexdigest()[:16], 16) / 2**64
        assignments[target] = ("test" if target in heldout else
            "train" if value < train_fraction else
            "tuning" if value < train_fraction + tuning_fraction else "test")
    return tuple(replace(s, split=assignments[identity(s)]) for s in samples)


def prepare_sources(config, output):
    """Prepare explicit records or discover products for configured TICs."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    settings = PreprocessingConfig(**config.get("preprocessing", {})).validate()
    records = list(config.get("sources", []))
    for target in config.get("targets", []):
        for uri in discover_spoc(target["tic"], sector=target.get("sector"), cadence_seconds=target.get("cadence_seconds")):
            records.append({**target, "source": uri})
    if not records:
        raise ValueError("No sources found")
    samples = []
    from tqdm.auto import tqdm
    for record in tqdm(records, desc="Prepare SPOC products", disable=not config.get("progress", False)):
        source = record["source"]
        path = cache_product(source, output / "products") if str(source).startswith("mast:") else Path(source).resolve()
        curve = read_lightcurve(path, settings)
        tic = record.get("tic", curve.metadata.get("TICID"))
        if tic is None:
            raise ValueError("Real SPOC records require a TIC identity")
        tic = str(int(str(tic).upper().removeprefix("TIC").strip()))
        sample_id = record.get("sample_id", hashlib.sha256(str(source).encode()).hexdigest()[:20])
        samples.append(EvaluationSample(
            sample_id=sample_id, source=str(path.resolve()), suite=record.get("suite", "real_labelled"),
            split=record.get("split", "test"), group_id=f"tic:{tic}", tic=tic,
            sector=record.get("sector", curve.metadata.get("SECTOR")), label=record.get("label"),
            label_provenance=record.get("label_provenance"), flux_product=settings.flux_product,
            cadence_seconds=float(np.nanmedian(curve.exposure_seconds)),
            parameters={"product_source": source, "source_sha256": sha256(path), **record.get("parameters", {})}))
    explicit = ["split" in record for record in records]
    if any(explicit) and not all(explicit):
        raise ValueError("Supply splits for every record or let all splits be assigned")
    if not any(explicit):
        samples = assign_splits(samples, **config.get("splits", {}))
    manifest = EvaluationManifest(tuple(samples), metadata={"time_unit": "day", "preparation": config,
                                                          "labels": "Unlabelled originals remain unknown"})
    manifest.write(output / "manifest.json")
    return manifest


def import_simulations(records, output, *, seed=42, tuning_fraction=0.3):
    """Import explicit system identities, labels and provenance, not directory guesses."""
    samples = []
    for row in records:
        path = Path(row["source"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        samples.append(EvaluationSample(**{**row, "source": str(path), "suite": "simulated",
                                           "split": row.get("split", "test")}))
    if not all("split" in row for row in records):
        if any("split" in row for row in records):
            raise ValueError("Provide either every split or none")
        samples = assign_splits(samples, seed=seed, tuning_fraction=tuning_fraction)
    manifest = EvaluationManifest(tuple(samples), metadata={"time_unit": "day", "seed": seed})
    manifest.write(output)
    return manifest
