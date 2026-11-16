"""JSON-configured prepare, simulate, inject, score and report workflow."""

import json
from pathlib import Path

from .datasets import EvaluationManifest


def read_config(path):
    """Resolve configured paths relative to the configuration file, never the CWD."""
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    base = path.parent
    for key in ("output", "manifest", "templates", "checkpoint", "classes", "training_manifest", "thresholds_from_run",
                "study_lock"):
        if config.get(key):
            candidate = Path(config[key]).expanduser()
            config[key] = str(candidate if candidate.is_absolute() else base / candidate)
    for record in config.get("sources", []):
        if not str(record["source"]).startswith("mast:"):
            record["source"] = str((base / Path(record["source"]).expanduser()).resolve())
    backgrounds = config.get('backgrounds', {})
    for key in ('catalog', 'exclude_manifest', 'exclude_prior_runs'):
        if backgrounds.get(key):
            backgrounds[key] = str((base / Path(backgrounds[key]).expanduser()).resolve())
    return config


def prepare(config):
    from .ingestion import prepare_sources, import_simulations
    if config.get("kind") == "simulation_import":
        Path(config["output"]).mkdir(parents=True, exist_ok=False)
        return import_simulations(config["sources"], Path(config["output"]) / "manifest.json", **config.get("splits", {}))
    return prepare_sources(config, config["output"])


def simulate(config):
    from .simulations import create_backgrounds
    from .injections import build_injections
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    controls = create_backgrounds(output / "backgrounds", **config.get("backgrounds", {}))
    return build_injections(controls, output / "injections", **config.get("injection", {}))


def inject(config):
    from .injections import build_injections
    return build_injections(EvaluationManifest.read(config["manifest"]), config["output"],
                            preprocessing=config.get("preprocessing"), **config.get("injection", {}))


def score(config):
    if config.get("study_lock"):
        raise ValueError("Historical study locks are no longer supported; use bin/evaluate_pipeline.py")
    from ..matched_filtering import load_template_library
    from .runner import evaluate
    manifest = EvaluationManifest.read(config["manifest"])
    templates = load_template_library(config["templates"])
    options = dict(config.get("evaluation", {}))
    if config.get("thresholds_from_run"):
        previous = json.loads(Path(config["thresholds_from_run"]).read_text())
        from .runner import template_fingerprint, file_hash
        if template_fingerprint(templates) != previous["template_sha256"]:
            raise ValueError("Transferred thresholds require an identical template bank")
        if previous.get("normalize_scores", False) != config.get("normalize_scores", True):
            raise ValueError("Transferred thresholds require the same score normalization")
        if previous["positive_class"] != options.get("positive_class", "lensing"):
            raise ValueError("Transferred thresholds require the same positive class")
        from .preprocessing import PreprocessingConfig
        from dataclasses import asdict
        if previous["preprocessing"] != asdict(PreprocessingConfig(**config.get("preprocessing", {}))):
            raise ValueError("Transferred thresholds require identical preprocessing")
        if "fixed_thresholds" in options:
            raise ValueError("Choose fixed_thresholds or thresholds_from_run")
        options["fixed_thresholds"] = {name: value for name, value in previous["thresholds"].items()
                                      if name == "matched_filter" or name == "cnn" and config.get("checkpoint")}
        if config.get("checkpoint"):
            previous_cnn = previous.get("cnn", {})
            classes = config.get("classes") or str(Path(config["checkpoint"]).with_name("model_classes.json"))
            if (file_hash(config["checkpoint"]) != previous_cnn.get("checkpoint_sha256") or
                    file_hash(classes) != previous_cnn.get("class_mapping_sha256")):
                raise ValueError("Transferred CNN thresholds require identical checkpoint and class mapping")
    if config.get("training_manifest"):
        from .datasets import audit_overlap
        audit = audit_overlap(manifest, EvaluationManifest.read(config["training_manifest"]))
    else:
        audit = {"status": "not supplied; fresh generation or external membership audit required"}
    cnn = None
    if config.get("checkpoint"):
        from .cnn import EvaluationCNN
        cnn = EvaluationCNN(config["checkpoint"], config.get("classes"), **config.get("cnn", {}))
        if config.get("thresholds_from_run"):
            sizes = [None if value == float("inf") else value for value in cnn.split_sizes]
            if sizes != previous_cnn["split_sizes_days"] or cnn.max_segments_per_item != previous_cnn["max_segments_per_item"]:
                raise ValueError("Transferred CNN thresholds require identical segmentation")
            if "rendering" in previous_cnn and previous_cnn["rendering"] != cnn.rendering:
                raise ValueError("Transferred CNN thresholds require identical rendering")
    from .runner import file_hash
    return evaluate(manifest, templates, config["output"], cnn=cnn, template_path=config["templates"],
                    preprocessing=config.get("preprocessing", {}),
                    normalize_scores=config.get("normalize_scores", True),
                    provenance={"experiment": config, "template_file_sha256": file_hash(config["templates"]),
                                "training_overlap_audit": audit}, **options)
