"""Installed dependency identities, including Git revisions for direct installs."""

import importlib.metadata
import json


def dependency_provenance(*names):
    """Record distribution versions and PEP 610 source identities.

    Include available source metadata for saved results and generation-cache
    identities. Registry wheels have no direct URL or inferred source commit.
    """
    result = {}
    for name in names:
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = {"version": None, "direct_url": None, "commit_id": None}
            continue
        direct_url = json.loads(distribution.read_text("direct_url.json") or "null")
        result[name] = {
            "version": distribution.version,
            "direct_url": direct_url,
            "commit_id": (direct_url or {}).get("vcs_info", {}).get("commit_id"),
        }
    return result
