"""Reproducible light-curve training and epoch-boundary checkpoint recovery."""

import copy
import hashlib
import json
import math
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from ...evaluation.provenance import dependency_provenance


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reproducibility is scoped to the same software and hardware environment.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _rng_state(generator):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
        "loader": generator.get_state(),
    }


def _restore_rng(state, generator):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         *numpy_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    if state.get("mps") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())
    generator.set_state(state["loader"].cpu())


def _atomic_save(path, value, *, json_file=False):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        if json_file:
            temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        else:
            torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _dataset_manifest(dataset, root):
    manifest = []
    for filename, label in tqdm(dataset.samples, desc="Hashing dataset", leave=False):
        path = Path(filename)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest.append({"path": str(path.relative_to(root)), "label": label,
                         "sha256": digest, "bytes": path.stat().st_size})
    return manifest


def _split_indices(targets, seed):
    """Keep every class represented in both partitions, without splitting curves."""
    generator = torch.Generator().manual_seed(seed)
    train, validation = [], []
    for label in sorted(set(targets)):
        indices = [index for index, target in enumerate(targets) if target == label]
        if len(indices) < 2:
            raise ValueError("Each class needs at least two light curves for train/validation")
        indices = [indices[index] for index in torch.randperm(len(indices), generator=generator).tolist()]
        count = max(1, int(0.2 * len(indices)))
        validation.extend(indices[:count])
        train.extend(indices[count:])
    return {"train": sorted(train), "validation": sorted(validation)}


def _source_provenance():
    root = Path(__file__).resolve().parents[3]
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(root), *args],
                                           stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    source_hashes = {}
    for relative in ("searchlight/models/simulens/v1.py", "searchlight/models/simulens/training.py",
                     "searchlight/lightcurves.py", "searchlight/utils.py", "searchlight/generation.py", "bin/train_cnn.py", "uv.lock"):
        path = root / relative
        if path.is_file():
            with path.open("rb") as stream:
                source_hashes[relative] = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"git_commit": git("rev-parse", "HEAD"),
            "source_sha256": source_hashes,
            "git_status": git("status", "--porcelain"),
            "dependencies": dependency_provenance(
                "searchlight", "torch", "torchvision", "numpy", "matplotlib",
                "astropy", "selflensing", "rustlens", "jaxoplanet")}


def _epoch(trainer, loader, *, training):
    trainer.model.train(training)
    loss_sum, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for inputs, mask, labels in tqdm(loader, desc="Train" if training else "Validation", leave=False):
            inputs = inputs.to(trainer.device, non_blocking=trainer.pin_memory)
            mask = mask.to(trainer.device, non_blocking=trainer.pin_memory)
            labels = labels.to(trainer.device, non_blocking=trainer.pin_memory)
            if training:
                trainer.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=trainer.use_amp):
                outputs = trainer.model(inputs, mask)
                loss = trainer.criterion(outputs, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training/validation loss; checkpoint was not advanced")
            if training:
                trainer.grad_scaler.scale(loss).backward()
                trainer.grad_scaler.step(trainer.optimizer)
                trainer.grad_scaler.update()
            total += labels.size(0)
            loss_sum += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
    return {"loss": loss_sum / total, "accuracy": correct / total}


def train(trainer, path, *, seed, max_epochs, patience, min_delta, resume, overwrite,
          provenance, init_checkpoint=None):
    from .v1 import GroupedLightcurveFolderDataSet, _resolve_split_sizes

    if trainer.loader_batch_size < 1 or trainer.loader_workers < 0 or trainer.max_segments_per_item < 1:
        raise ValueError("batch size and segment cap must be positive; workers must be nonnegative")
    if max_epochs < 1 or patience < 1 or not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError("max_epochs/patience must be positive and min_delta finite and nonnegative")
    if resume and init_checkpoint:
        raise ValueError("resume and init_checkpoint cannot be combined")
    if resume and overwrite:
        raise ValueError("resume and overwrite cannot be combined")
    root = Path(path).expanduser().resolve()
    output = trainer.MODEL_SAVE_PATH.parent
    checkpoint_path = output / "training_checkpoint.pth"
    metadata_path = output / "training_run.json"
    artifacts = [trainer.MODEL_SAVE_PATH, trainer.CLASS_SAVE_PATH, checkpoint_path, metadata_path]
    if not resume and not overwrite and any(item.exists() for item in artifacts):
        raise FileExistsError(f"Training artifacts already exist in {output}; choose a new output directory")
    if resume and Path(resume).expanduser().resolve() != checkpoint_path.resolve():
        raise ValueError("Resume checkpoint must belong to the selected output directory")

    dataset = GroupedLightcurveFolderDataSet(root, transform=trainer.transform,
        split_sizes=trainer.split_sizes, max_segments_per_item=trainer.max_segments_per_item)
    counts = {name: dataset.targets.count(index) for index, name in enumerate(dataset.classes)}
    if len(counts) < 2 or any(count < 2 for count in counts.values()):
        raise ValueError(f"Need at least two nonempty classes with two light curves each: {counts}")
    manifest = _dataset_manifest(dataset, root)
    generated_path = root / "generation.json"
    if generated_path.exists():
        generated = json.loads(generated_path.read_text())
        if generated.get("status") != "complete":
            raise ValueError("Generated dataset is incomplete; finish generation before training")
        expected = {entry["path"]: entry["sha256"] for entry in generated["files"]}
        actual = {entry["path"]: entry["sha256"] for entry in manifest}
        if expected != actual:
            raise ValueError("Generated dataset no longer matches its generation manifest")
        provenance = {"generation": generated, "user": provenance}
    settings = {
        "seed": seed, "patience": patience, "min_delta": min_delta,
        "split_sizes": [None if math.isinf(size) else size for size in _resolve_split_sizes(trainer.split_sizes)],
        "max_segments_per_item": trainer.max_segments_per_item,
        "batch_size": trainer.loader_batch_size, "loader_workers": trainer.loader_workers,
        "loader_prefetch_factor": trainer.loader_prefetch_factor,
        "lr": trainer.lr, "weight_decay": trainer.weight_decay,
        "scheduler_step_size": trainer.scheduler_step_size, "scheduler_gamma": trainer.scheduler_gamma,
        "use_amp": trainer.use_amp, "pretrained": trainer.pretrained,
    }
    initialization = None
    if init_checkpoint:
        initial_path = Path(init_checkpoint).expanduser().resolve()
        initial_classes = json.loads(initial_path.with_name("model_classes.json").read_text())
        if initial_classes != dataset.classes:
            raise ValueError("Initial checkpoint class mapping must match the dataset's ordered classes")
        with initial_path.open("rb") as stream:
            initialization = {"checkpoint": str(initial_path),
                              "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
    settings["init_checkpoint_sha256"] = initialization["sha256"] if initialization else None
    checkpoint = None
    if resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["manifest"] != manifest or checkpoint["classes"] != dataset.classes:
            raise ValueError("Dataset contents or classes changed since checkpoint; start a new run")
        # Initialization weights are irrelevant when all weights are restored.
        saved_settings = {k: v for k, v in checkpoint["settings"].items() if k not in {"pretrained", "init_checkpoint_sha256"}}
        if saved_settings != {k: v for k, v in settings.items() if k not in {"pretrained", "init_checkpoint_sha256"}}:
            raise ValueError("Training settings changed since checkpoint")
        settings = checkpoint["settings"]
        if max_epochs <= checkpoint["epoch"]:
            raise ValueError("max_epochs must exceed the checkpoint epoch (it is a total, not an increment)")
    split = checkpoint["split"] if checkpoint else _split_indices(dataset.targets, seed)
    trainer.classes = dataset.classes
    trainer.train_dataset = torch.utils.data.Subset(dataset, split["train"])
    trainer.test_dataset = torch.utils.data.Subset(dataset, split["validation"])
    _seed(seed)
    if checkpoint or initialization:
        trainer.pretrained = False
    trainer._set_model(len(trainer.classes), reset=True)
    if initialization:
        state = torch.load(initialization["checkpoint"], map_location="cpu", weights_only=True)
        if all(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        trainer._unwrap_model().load_state_dict(state)
    trainer._reset_optimizer_and_scheduler()
    trainer.grad_scaler = torch.amp.GradScaler("cuda", enabled=trainer.use_amp)
    generator = torch.Generator().manual_seed(seed + 1)
    train_loader = trainer._build_dataloader(trainer.train_dataset, shuffle=True, generator=generator)
    validation_loader = trainer._build_dataloader(trainer.test_dataset, shuffle=False, generator=generator)

    history, best_loss, best_epoch, bad_epochs, best_weights, start_epoch = [], math.inf, 0, 0, None, 0
    patience_loss = math.inf
    if checkpoint:
        trainer._unwrap_model().load_state_dict(checkpoint["model"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        trainer.scheduler.load_state_dict(checkpoint["scheduler"])
        trainer.grad_scaler.load_state_dict(checkpoint["scaler"])
        history, best_loss, best_epoch = checkpoint["history"], checkpoint["best_loss"], checkpoint["best_epoch"]
        bad_epochs, best_weights, start_epoch = checkpoint["bad_epochs"], checkpoint["best_weights"], checkpoint["epoch"]
        patience_loss = checkpoint["patience_loss"]
        _restore_rng(checkpoint["rng"], generator)
    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for artifact in artifacts:
            artifact.unlink(missing_ok=True)
    run = {
        "schema_version": 1, "dataset_root": str(root), "classes": trainer.classes,
        "class_counts": counts, "manifest": manifest, "split": split, "settings": settings,
        "source": _source_provenance(), "dataset_provenance": provenance,
        "initialization": initialization,
        "device": str(trainer.device), "max_epochs": max_epochs,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "cuda_runtime": torch.version.cuda},
    }
    if checkpoint:
        run["original_run"] = copy.deepcopy(checkpoint["run"])
    _atomic_save(trainer.CLASS_SAVE_PATH, trainer.classes, json_file=True)
    _atomic_save(metadata_path, run, json_file=True)

    for epoch in range(start_epoch + 1, max_epochs + 1):
        training = _epoch(trainer, train_loader, training=True)
        validation = _epoch(trainer, validation_loader, training=False)
        trainer.scheduler.step()
        history.append({"epoch": epoch, "training": training, "validation": validation})
        # Always retain the actual lowest loss; min_delta only controls patience.
        significant_improvement = validation["loss"] < patience_loss - min_delta
        if significant_improvement:
            patience_loss = validation["loss"]
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch
            best_weights = {key: value.detach().cpu().clone()
                            for key, value in trainer._unwrap_model().state_dict().items()}
            _atomic_save(trainer.MODEL_SAVE_PATH, best_weights)
        bad_epochs = 0 if significant_improvement else bad_epochs + 1
        run.update(history=history, best_epoch=best_epoch, best_validation_loss=best_loss,
                   stopped_epoch=epoch, stop_reason=("patience" if bad_epochs >= patience else
                       "max_epochs" if epoch == max_epochs else None))
        checkpoint = {
            "schema_version": 1, "epoch": epoch, "model": trainer._unwrap_model().state_dict(),
            "optimizer": trainer.optimizer.state_dict(), "scheduler": trainer.scheduler.state_dict(),
            "scaler": trainer.grad_scaler.state_dict(), "best_weights": best_weights,
            "best_loss": best_loss, "best_epoch": best_epoch, "bad_epochs": bad_epochs,
            "patience_loss": patience_loss,
            "history": history, "rng": _rng_state(generator), "manifest": manifest,
            "classes": trainer.classes, "split": split, "settings": settings, "run": run,
        }
        _atomic_save(checkpoint_path, checkpoint)
        _atomic_save(metadata_path, run, json_file=True)
        print(f"Epoch {epoch}: training loss={training['loss']:.6f}, "
              f"validation loss={validation['loss']:.6f}, accuracy={validation['accuracy']:.2%}")
        if bad_epochs >= patience:
            break
    _atomic_save(trainer.MODEL_SAVE_PATH, best_weights)
    trainer._unwrap_model().load_state_dict(best_weights)
    trainer.model.eval()
    return run
