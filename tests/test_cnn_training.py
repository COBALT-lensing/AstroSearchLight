"""Training regressions using a small CPU network and synthetic segment tensors."""

import json

import pytest
import torch

from searchlight.models.simulens import v1
from searchlight.models.simulens import training
from searchlight import utils, config


class TinyNetwork(torch.nn.Module):
    def __init__(self, num_classes, weights=None):
        super().__init__()
        self.linear = torch.nn.Linear(1, 4)
        self.bn = torch.nn.BatchNorm1d(4)
        self.dropout = torch.nn.Dropout(0.4)
        self.classifier = torch.nn.Linear(4, num_classes)

    def forward(self, images, mask):
        features = images[:, 0].flatten(1).mean(1, keepdim=True)
        return self.classifier(self.dropout(self.bn(self.linear(features))))


@pytest.fixture
def setup_training(monkeypatch, tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(v1, "SegmentAwareResNet", TinyNetwork)
    monkeypatch.setattr(training, "_source_provenance", lambda: {"test": True})
    def get_item(self, index):
        _, label = self.samples[index]
        return [torch.full((1, 2, 2), float(index + 1))], label
    monkeypatch.setattr(v1.GroupedLightcurveFolderDataSet, "__getitem__", get_item)
    root = config.dataset_directory("2.0")
    for name in ("lensing", "not-lensing"):
        (root / name).mkdir(parents=True)
        for index in range(5):
            (root / name / f"{index}.fits").write_text(f"{name}-{index}")
    versions = {}
    def trainer(name):
        version = versions.setdefault(name, f"{len(versions) + 1}.0")
        return v1.SimuLens(device="cpu", version=version,
                          pretrained=False, loader_batch_size=2, loader_workers=0)
    yield root, trainer
    torch.set_num_threads(previous_threads)


def test_validation_uses_eval_without_gradients_or_batchnorm_updates(setup_training):
    root, make_trainer = setup_training
    trainer = make_trainer("validation")
    dataset = v1.GroupedLightcurveFolderDataSet(root)
    loader = trainer._build_dataloader(dataset, shuffle=False)
    training._epoch(trainer, loader, training=True)
    state_before = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    seen = []
    handle = trainer.model.register_forward_hook(
        lambda model, args, out: seen.append((model.training, out.requires_grad)))
    first = training._epoch(trainer, loader, training=False)
    second = training._epoch(trainer, loader, training=False)
    handle.remove()
    assert first == second
    assert seen and all(item == (False, False) for item in seen)
    assert all(torch.equal(value, trainer.model.state_dict()[key]) for key, value in state_before.items())
    training._epoch(trainer, loader, training=True)
    assert trainer.model.training


def test_best_checkpoint_is_independent_and_patience_stops(setup_training, monkeypatch):
    root, make_trainer = setup_training
    trainer = make_trainer("best")
    losses = iter([0.8, 0.3, 0.4, 0.5])
    def epoch(model, loader, *, training):
        if training:
            with torch.no_grad():
                next(model.model.parameters()).add_(1)
            # Exercise the optimizer before stepping its scheduler.
            model.optimizer.zero_grad()
            next(model.model.parameters()).sum().backward()
            model.optimizer.step()
        return {"loss": 1.0 if training else next(losses), "accuracy": 0.5}
    monkeypatch.setattr(training, "_epoch", epoch)
    run = trainer.train("2.0", max_epochs=20, patience=2)
    checkpoint = torch.load(trainer.MODEL_SAVE_PATH.parent / "training_checkpoint.pth", weights_only=True)
    best = torch.load(trainer.MODEL_SAVE_PATH, weights_only=True)
    assert run["best_epoch"] == 2
    assert run["stopped_epoch"] == 4
    assert run["stop_reason"] == "patience"
    assert not torch.equal(best["linear.weight"], checkpoint["model"]["linear.weight"])
    assert torch.equal(best["linear.weight"], checkpoint["best_weights"]["linear.weight"])
    assert torch.equal(trainer.model.linear.weight, best["linear.weight"])


def test_resume_matches_uninterrupted_training_and_records_split(setup_training):
    root, make_trainer = setup_training
    full = make_trainer("full")
    full.train("2.0", max_epochs=4, patience=10, seed=17)
    partial = make_trainer("resumed")
    partial.train("2.0", max_epochs=2, patience=10, seed=17,
                  provenance={"simulator_revision": "fixture"})
    resumed = make_trainer("resumed")
    run = resumed.train("2.0", max_epochs=4, patience=10, seed=17,
        resume=True)
    full_checkpoint = torch.load(full.MODEL_SAVE_PATH.parent / "training_checkpoint.pth", weights_only=True)
    resumed_checkpoint = torch.load(resumed.MODEL_SAVE_PATH.parent / "training_checkpoint.pth", weights_only=True)
    assert full_checkpoint["history"] == resumed_checkpoint["history"]
    for key, value in full_checkpoint["model"].items():
        assert torch.equal(value, resumed_checkpoint["model"][key])
    assert full_checkpoint["scheduler"] == resumed_checkpoint["scheduler"]
    assert full_checkpoint["split"] == resumed_checkpoint["split"]
    split = run["split"]
    assert not set(split["train"]) & set(split["validation"])
    assert sorted(split["train"] + split["validation"]) == list(range(10))
    assert all(len(item["sha256"]) == 64 for item in run["manifest"])
    assert run["original_run"]["dataset_provenance"] == {"simulator_revision": "fixture"}
    json.loads((resumed.MODEL_SAVE_PATH.parent / "training_run.json").read_text())


def test_existing_run_and_changed_resume_data_are_rejected(setup_training):
    root, make_trainer = setup_training
    trainer = make_trainer("protected")
    trainer.train("2.0", max_epochs=1)
    weights = trainer.MODEL_SAVE_PATH.read_bytes()
    with pytest.raises(FileExistsError):
        trainer.train("2.0", max_epochs=2)
    (root / "lensing" / "0.fits").write_text("changed")
    with pytest.raises(ValueError, match="Dataset contents"):
        trainer.train("2.0", max_epochs=2,
                      resume=True)
    assert trainer.MODEL_SAVE_PATH.read_bytes() == weights


def test_resume_rejects_changed_settings(setup_training):
    root, make_trainer = setup_training
    trainer = make_trainer("settings")
    trainer.train("2.0", max_epochs=1)
    with pytest.raises(ValueError, match="settings changed"):
        trainer.train("2.0", max_epochs=2, seed=99,
                      resume=True)


def test_empty_class_is_rejected_before_writing_artifacts(setup_training):
    root, make_trainer = setup_training
    (root / "empty").mkdir()
    trainer = make_trainer("empty")
    with pytest.raises(ValueError, match="nonempty classes"):
        trainer.train("2.0", max_epochs=1)
    assert not trainer.MODEL_SAVE_PATH.parent.exists()


def test_plot_cache_invalidates_when_fits_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "FUNC_CACHE_PATH", tmp_path / "cache")
    monkeypatch.setattr(utils, "CACHE_ENABLED", True)
    path = tmp_path / "curve.fits"
    path.write_text("old")
    calls = []
    monkeypatch.setattr(v1, "load", lambda filename: calls.append(filename) or path.read_text())
    monkeypatch.setattr(v1, "plot", lambda curve: curve)
    monkeypatch.setattr(v1, "plot_to_rgb", lambda curve: curve)
    assert v1.load_fits(str(path)) == "old"
    assert v1.load_fits(str(path)) == "old"
    assert len(calls) == 1
    path.write_text("new")
    assert v1.load_fits(str(path)) == "new"
    assert len(calls) == 2


def test_min_delta_counts_cumulative_improvement(setup_training, monkeypatch):
    root, make_trainer = setup_training
    trainer = make_trainer("delta")
    losses = iter([1.0, 0.96, 0.92, 0.91])
    def epoch(model, loader, *, training):
        if training:
            model.optimizer.zero_grad()
            next(model.model.parameters()).sum().backward()
            model.optimizer.step()
        return {"loss": 1.0 if training else next(losses), "accuracy": 0.5}
    monkeypatch.setattr(training, "_epoch", epoch)
    run = trainer.train("2.0", max_epochs=4, patience=2, min_delta=0.05)
    assert run["stopped_epoch"] == 4
    assert run["stop_reason"] == "max_epochs"
    assert run["best_epoch"] == 4


def test_real_fits_rendering_training_and_inference(tmp_path, monkeypatch):
    from astropy.table import QTable
    from astropy.time import Time
    from astropy.timeseries import TimeSeries
    import numpy as np

    monkeypatch.setattr(v1, "SegmentAwareResNet", TinyNetwork)
    monkeypatch.setattr(training, "_source_provenance", lambda: {"test": True})
    monkeypatch.setattr(utils, "FUNC_CACHE_PATH", tmp_path / "cache")
    monkeypatch.setattr(utils, "CACHE_ENABLED", True)
    root = config.dataset_directory("2.0")
    for label, name in enumerate(("lensing", "not-lensing")):
        (root / name).mkdir(parents=True)
        for index in range(4):
            curve = TimeSeries(time=Time(2460000 + np.arange(12) / 12, format="jd"))
            curve["flux"] = 1 + (label + 1) * np.sin(np.arange(12) + index) * 0.01
            QTable({"time": curve.time, "flux": curve["flux"]}).write(root / name / f"{index}.fits")
    trainer = v1.SimuLens(device="cpu", version="2.0", pretrained=False,
                         loader_workers=0, loader_batch_size=2, split_sizes=[np.inf])
    run = trainer.train("2.0", max_epochs=1)
    assert run["best_epoch"] == 1
    inference = v1.SimuLens(device="cpu", version="2.0", pretrained=False,
                           loader_workers=0, split_sizes=[np.inf])
    path = str(root / "lensing" / "0.fits")
    assert inference.classify_paths([path])[path][0] in run["classes"]


def test_cli_requires_explicit_version_and_resumes_without_imagenet(tmp_path, monkeypatch):
    import runpy
    from pathlib import Path
    from searchlight.models import simulens

    seen = {}
    class FakeTrainer:
        LOADER_WORKERS = 0
        def __init__(self, **kwargs):
            seen["init"] = kwargs
            self.MODEL_SAVE_PATH = config.simulens_directory(kwargs["version"]) / "model_weights.pth"
        def train(self, path, **kwargs):
            seen["train"] = (path, kwargs)
            return {"best_epoch": 1}
    monkeypatch.setattr(simulens, "SimuLens", FakeTrainer)
    main = runpy.run_path(str(Path(__file__).parents[1] / "bin/train_cnn.py"))["main"]
    with pytest.raises(SystemExit):
        main([])
    assert not seen
    main(["--dataset-version", "2.0", "--version", "2.0", "--resume"])
    assert seen["init"]["pretrained"] is False
    assert seen["train"][1]["resume"] is True


def test_new_dataset_can_initialize_from_existing_model(setup_training, monkeypatch):
    root, make_trainer = setup_training
    source = make_trainer("source")
    source.train("2.0", max_epochs=1)
    source_weights = torch.load(source.MODEL_SAVE_PATH, weights_only=True)
    (root / "lensing/0.fits").write_text("updated dataset")
    target = make_trainer("fine-tune")
    original_epoch = training._epoch
    seen = []
    def epoch(trainer, loader, *, training):
        if not seen:
            assert all(torch.equal(value, trainer.model.state_dict()[key])
                       for key, value in source_weights.items())
            assert not trainer.optimizer.state  # New run, fresh optimizer.
            seen.append(True)
        return original_epoch(trainer, loader, training=training)
    monkeypatch.setattr(training, "_epoch", epoch)
    run = target.train("2.0", max_epochs=1, init_version=source.version)
    assert seen
    assert run["initialization"]["checkpoint"] == str(source.MODEL_SAVE_PATH)
    assert len(run["initialization"]["sha256"]) == 64
    (source.CLASS_SAVE_PATH).write_text('["not-lensing", "lensing"]')
    rejected = make_trainer("wrong-mapping")
    with pytest.raises(ValueError, match="class mapping"):
        rejected.train("2.0", max_epochs=1, init_version=source.version)
    assert not rejected.MODEL_SAVE_PATH.parent.exists()


def test_overwrite_does_not_leave_old_resume_checkpoint_on_failure(setup_training, monkeypatch):
    root, make_trainer = setup_training
    trainer = make_trainer("overwrite")
    trainer.train("2.0", max_epochs=1)
    def fail(*args, **kwargs):
        raise RuntimeError("failed first epoch")
    monkeypatch.setattr(training, "_epoch", fail)
    with pytest.raises(RuntimeError, match="first epoch"):
        trainer.train("2.0", max_epochs=1, overwrite=True)
    assert not (trainer.MODEL_SAVE_PATH.parent / "training_checkpoint.pth").exists()
    assert not trainer.MODEL_SAVE_PATH.exists()


def test_generated_dataset_provenance_is_loaded_and_verified(setup_training):
    import hashlib
    root, make_trainer = setup_training
    entries = [{"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
               for path in sorted(root.rglob("*.fits"))]
    generated = {"generator": "photometry-batch-v1", "status": "complete", "files": entries}
    (root / "generation.json").write_text(json.dumps(generated))
    trainer = make_trainer("generated")
    run = trainer.train("2.0", max_epochs=1)
    assert run["dataset_provenance"]["generation"] == generated
    (root / "lensing/0.fits").write_text("changed since generation")
    with pytest.raises(ValueError, match="generation manifest"):
        make_trainer("changed-generated").train("2.0", max_epochs=1)
    generated["status"] = "generating"
    (root / "generation.json").write_text(json.dumps(generated))
    with pytest.raises(ValueError, match="incomplete"):
        make_trainer("incomplete-generated").train("2.0", max_epochs=1)


def test_classification_and_evaluation_restore_saved_segment_settings(setup_training, monkeypatch):
    from searchlight.evaluation.cnn import EvaluationCNN
    root, make_trainer = setup_training
    trainer = make_trainer("segments")
    trainer.split_sizes = [float("inf"), 3., .5]
    trainer.max_segments_per_item = 8
    trainer.train("2.0", max_epochs=1)
    saved = json.loads((trainer.MODEL_SAVE_PATH.parent / "inference.json").read_text())
    assert saved == {"split_sizes": [None, 3., .5], "max_segments_per_item": 8}
    reloaded = make_trainer("segments")
    monkeypatch.setattr(v1.GroupedLightcurvePathsDataSet, "__getitem__",
        lambda self, index: ([torch.ones(1, 2, 2)], self.paths[index]))
    assert reloaded.classify_paths(["first", "second"])
    assert reloaded.split_sizes == [float("inf"), 3., .5]
    assert reloaded.max_segments_per_item == 8
    # Multiple workers defer model initialization but must resolve the same settings.
    evaluation = EvaluationCNN(version=trainer.version, workers=2)
    assert evaluation.split_sizes == reloaded.split_sizes
    assert evaluation.max_segments_per_item == reloaded.max_segments_per_item


def test_training_uses_configured_dataset_version_and_records_it(setup_training):
    root, make_trainer = setup_training
    config.initialize_config()
    config.config_path().write_text('{"dataset_version":"2.0"}')
    trainer = make_trainer("default-dataset")
    run = trainer.train(max_epochs=1)
    assert run["dataset_version"] == "2.0"
    assert run["dataset_root"] == str(root)
    assert run["settings"]["dataset_version"] == "2.0"
    assert run["model_version"] == trainer.version
    assert run["model_version"] != run["dataset_version"]


def test_resume_rejects_a_different_dataset_version_even_with_same_contents(setup_training):
    import shutil
    root, make_trainer = setup_training
    trainer = make_trainer("dataset-resume")
    trainer.train("2.0", max_epochs=1)
    shutil.copytree(root, config.dataset_directory("3.0"))
    with pytest.raises(ValueError, match="settings changed"):
        trainer.train("3.0", max_epochs=2, resume=True)


def test_training_rejects_dataset_paths_and_missing_versions(setup_training):
    root, make_trainer = setup_training
    trainer = make_trainer("missing-dataset")
    with pytest.raises(ValueError, match="Version must"):
        trainer.train(root, max_epochs=1)
    with pytest.raises(FileNotFoundError, match="Dataset version 9.0"):
        trainer.train("9.0", max_epochs=1)
    assert not trainer.MODEL_SAVE_PATH.parent.exists()
