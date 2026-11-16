"""Artifact versions, their on-disk locations, and saved CNN inference settings."""

import json
import math
from pathlib import Path
import re

from . import paths

# Change these to select new defaults. Explicit versions and config.json override them.
DEFAULT_SIMULENS_VERSION = "1.0"
DEFAULT_TEMPLATE_VERSION = "1.0"
DEFAULT_DATASET_VERSION = "1.0"
DEFAULT_SPLIT_SIZES = (math.inf, 7.0, 2.0)
DEFAULT_MAX_SEGMENTS_PER_ITEM = 24


def validate_version(version):
    if not isinstance(version, str) or re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", version) is None:
        raise ValueError("Version must be a numeric string such as '1.0' or '2.1'; paths are not accepted")
    return version


def config_path():
    return paths.DATA_PATH / "config.json"


def read_config():
    path = config_path()
    values = json.loads(path.read_text()) if path.exists() else {}
    if not isinstance(values, dict) or set(values) - {"simulens_version", "template_version", "dataset_version"}:
        raise ValueError(f"Invalid artifact configuration: {path}")
    for version in values.values():
        validate_version(version)
    return values


def resolve_version(kind, version=None):
    if version is not None:
        return validate_version(version)
    defaults = {"simulens": DEFAULT_SIMULENS_VERSION, "template": DEFAULT_TEMPLATE_VERSION,
                "dataset": DEFAULT_DATASET_VERSION}
    if kind not in defaults:
        raise ValueError(f"Unknown artifact kind: {kind}")
    return validate_version(read_config().get(f"{kind}_version", defaults[kind]))


def simulens_directory(version=None):
    return paths.DATA_PATH / "models" / "simulens" / resolve_version("simulens", version)


def dataset_directory(version=None):
    return paths.DATA_PATH / "datasets" / "simulens" / resolve_version("dataset", version)


def template_library_path(version=None):
    return paths.DATA_PATH / "templates" / "matched_filter" / resolve_version("template", version) / "templates.npz"


def cnn_inference_settings(directory, *, split_sizes=None, max_segments_per_item=None):
    """Restore saved settings, with explicit overrides and a legacy 1.0 fallback."""
    directory = Path(directory)
    saved = {}
    inference = directory / "inference.json"
    training = directory / "training_run.json"
    if inference.exists():
        saved = json.loads(inference.read_text())
    elif training.exists():
        saved = json.loads(training.read_text())["settings"]
    if not isinstance(saved, dict):
        raise ValueError("Invalid saved CNN inference settings")
    sizes = split_sizes if split_sizes is not None else saved.get("split_sizes", DEFAULT_SPLIT_SIZES)
    sizes = [math.inf if value is None else float(value) for value in sizes]
    if not sizes or any(math.isnan(value) or value <= 0 for value in sizes):
        raise ValueError("CNN segment sizes must be positive and not NaN")
    cap = max_segments_per_item if max_segments_per_item is not None else saved.get(
        "max_segments_per_item", DEFAULT_MAX_SEGMENTS_PER_ITEM)
    if cap is not None and (not isinstance(cap, int) or isinstance(cap, bool) or cap < 1):
        raise ValueError("CNN segment cap must be a positive integer or null")
    return sizes, cap


def initialize_config():
    """Create an empty override file, preserving the defaults defined above."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as stream:
            stream.write("{}\n")
    except FileExistsError:
        read_config()  # Never replace an existing selection.
    return path


def migrate_legacy_templates():
    """Move the legacy bank to version 1.0 without overwriting any existing bank."""
    import hashlib
    source = paths.DATA_PATH / "matched_filter_templates.npz"
    destination = template_library_path("1.0")
    if not source.exists():
        return destination if destination.exists() else None
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        def digest(path):
            with path.open("rb") as stream:
                return hashlib.file_digest(stream, "sha256").digest()
        if digest(source) != digest(destination):
            raise FileExistsError(f"Different template bank already exists: {destination}")
    else:
        # Both paths belong to the same artifact store. Linking creates the new
        # name atomically and fails rather than overwriting another version.
        destination.hardlink_to(source)
    source.unlink()
    return destination


if __name__ == "__main__":
    print(f"Configuration: {initialize_config()}")
    print(f"Legacy template bank: {migrate_legacy_templates()}")
