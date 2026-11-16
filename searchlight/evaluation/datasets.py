"""Versioned JSON manifests and leakage checks for evaluation datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

SPLITS = {"train", "tuning", "test"}
SUITES = {"simulated", "spoc_injection", "real_labelled"}


@dataclass(frozen=True)
class EvaluationSample:
    sample_id: str
    source: str
    suite: str
    split: str
    # Stable identity of the underlying star or simulated physical system.
    group_id: str
    label: str | None = None
    label_provenance: str | None = None
    tic: str | None = None
    sector: int | None = None
    cadence_seconds: float | None = None
    flux_product: str | None = None
    parent_id: str | None = None
    parameters: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationManifest:
    samples: tuple[EvaluationSample, ...]
    metadata: dict = field(default_factory=dict)
    schema_version: int = 1

    def validate(self):
        if self.schema_version != 1:
            raise ValueError("Unsupported manifest schema_version")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")
        ids = {}
        assignments = {}
        tic_groups = {}
        for sample in self.samples:
            for name in ("sample_id", "source", "group_id"):
                if not isinstance(getattr(sample, name), str) or not getattr(sample, name).strip():
                    raise ValueError(f"{name} must be a nonempty string")
            if sample.sample_id in ids:
                raise ValueError(f"Duplicate sample_id: {sample.sample_id}")
            ids[sample.sample_id] = sample
            if sample.split not in SPLITS or sample.suite not in SUITES:
                raise ValueError(f"Invalid split or suite for {sample.sample_id}")
            if sample.label is not None:
                if not isinstance(sample.label, str) or not sample.label.strip():
                    raise ValueError("label must be a nonempty string or None")
                if not isinstance(sample.label_provenance, str) or not sample.label_provenance.strip():
                    raise ValueError(f"Label provenance required for {sample.sample_id}")
            if not isinstance(sample.parameters, dict):
                raise ValueError("parameters must be a dictionary")
            if sample.cadence_seconds is not None:
                if not math.isfinite(sample.cadence_seconds) or sample.cadence_seconds <= 0:
                    raise ValueError("cadence_seconds must be finite and positive")
            # Check both declared grouping and TIC so changing group_id cannot
            # accidentally bypass target-level isolation across benchmark suites.
            keys = [("group", sample.group_id), ("source", str(Path(sample.source).resolve()))]
            if sample.tic is not None:
                tic = str(sample.tic).strip().upper().removeprefix("TIC").strip()
                if not tic.isdecimal() or int(tic) <= 0:
                    raise ValueError(f"Invalid TIC: {sample.tic}")
                canonical_tic = str(int(tic))
                previous_group = tic_groups.setdefault(canonical_tic, sample.group_id)
                if previous_group != sample.group_id:
                    raise ValueError("Identity leakage: all observations of one TIC must share group_id")
                keys.append(("tic", canonical_tic))
            for key in keys:
                previous = assignments.setdefault(key, sample.split)
                if previous != sample.split:
                    raise ValueError(f"Split leakage for {key}: {previous}, {sample.split}")
        for sample in self.samples:
            if sample.parent_id is None:
                continue
            if sample.parent_id not in ids:
                raise ValueError(f"Missing parent: {sample.parent_id}")
            parent = ids[sample.parent_id]
            if parent.sample_id == sample.sample_id:
                raise ValueError("Sample cannot be its own parent")
            if (parent.group_id, parent.split) != (sample.group_id, sample.split):
                raise ValueError("Parent and derivative must share group_id and split")
            visited = {sample.sample_id}
            ancestor = parent
            while True:
                if ancestor.sample_id in visited:
                    raise ValueError("Cycle in sample ancestry")
                visited.add(ancestor.sample_id)
                if ancestor.parent_id is None:
                    break
                if ancestor.parent_id not in ids:
                    raise ValueError(f"Missing parent: {ancestor.parent_id}")
                ancestor = ids[ancestor.parent_id]
        return self

    def write(self, path):
        self.validate()
        payload = json.dumps(asdict(self), indent=2, allow_nan=False) + "\n"
        Path(path).write_text(payload, encoding="utf-8")

    @classmethod
    def read(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["samples"] = tuple(EvaluationSample(**row) for row in payload["samples"])
        return cls(**payload).validate()


def audit_overlap(evaluation, training):
    """Reject shared declared identities or byte-identical files across datasets.

    Physical identity cannot be inferred from differently labelled simulations;
    callers must supply stable system group IDs as well as training membership.
    """
    from .ingestion import sha256
    evaluation.validate()
    training.validate()
    def identities(manifest):
        values = set()
        for sample in manifest.samples:
            values.add(("group", sample.group_id))
            if sample.tic is not None:
                values.add(("tic", str(int(str(sample.tic).upper().removeprefix("TIC").strip()))))
            values.add(("source", str(Path(sample.source).resolve())))
            if Path(sample.source).is_file():
                values.add(("sha256", sha256(sample.source)))
        return values
    overlap = identities(evaluation) & identities(training)
    if overlap:
        raise ValueError(f"Training/evaluation overlap: {sorted(overlap)}")
    return {"status": "no declared identity or byte overlap", "training_rows": len(training.samples),
            "evaluation_rows": len(evaluation.samples)}
