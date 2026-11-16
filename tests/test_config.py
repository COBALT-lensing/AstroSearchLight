"""Version resolution, immutable artifact publication and legacy migration."""
import hashlib
import json

import numpy as np
import pytest

from searchlight import config, paths
from searchlight import matched_filtering as filtering


def test_default_constants_configuration_and_explicit_version_precedence(monkeypatch):
    assert config.simulens_directory().parts[-3:] == ("models", "simulens", "1.0")
    assert config.template_library_path().parts[-4:] == ("templates", "matched_filter", "1.0", "templates.npz")
    config.initialize_config()
    assert json.loads(config.config_path().read_text()) == {}
    monkeypatch.setattr(config, "DEFAULT_SIMULENS_VERSION", "2.0")
    monkeypatch.setattr(config, "DEFAULT_TEMPLATE_VERSION", "3.0")
    assert config.resolve_version("simulens") == "2.0"
    assert config.resolve_version("template") == "3.0"
    config.config_path().write_text('{"simulens_version":"4.0", "template_version":"5.0"}')
    assert config.resolve_version("simulens") == "4.0"
    assert config.resolve_version("template") == "5.0"
    assert config.resolve_version("template", "6.0") == "6.0"
    config.initialize_config()
    assert config.read_config()["simulens_version"] == "4.0"


@pytest.mark.parametrize("version", ["../2.0", "/tmp/model", "model.pth", "", 1.0, "2.0/bank", "1\\2"])
def test_versions_cannot_be_paths(version):
    with pytest.raises(ValueError, match="Version must"):
        config.simulens_directory(version)
    with pytest.raises(ValueError, match="Version must"):
        filtering.load_template_library(version)
    with pytest.raises(ValueError, match="Version must"):
        config.dataset_directory(version)


def test_template_versions_are_selected_and_preserved_independently():
    a = filtering.save_template_library("1.0", flux=np.array([[1., 2., 1.]]), time=np.arange(3.))
    original = a.read_bytes()
    filtering.save_template_library("2.0", flux=np.array([[1., 3., 1.]]), time=np.arange(3.))
    config.initialize_config()
    config.config_path().write_text('{"template_version":"2.0"}')
    assert filtering.load_template_library().flux[0, 1] == 3.
    assert filtering.load_template_library("1.0").flux[0, 1] == 2.
    with pytest.raises(FileExistsError):
        filtering.save_template_library("1.0", flux=np.array([[9., 9., 9.]]), time=np.arange(3.))
    assert a.read_bytes() == original


def test_migration_preserves_bytes_and_is_idempotent():
    paths.DATA_PATH.mkdir(parents=True)
    source = paths.DATA_PATH / "matched_filter_templates.npz"
    filtering._save_template_library_file(source, flux=np.array([[1., 2., 1.]]), time=np.arange(3.))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    target = config.migrate_legacy_templates()
    assert target == config.template_library_path("1.0")
    assert not source.exists()
    assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
    assert config.migrate_legacy_templates() == target
    assert filtering.load_template_library("1.0").flux.shape == (1, 3)


def test_migration_does_not_overwrite_a_different_version_one_bank():
    target = filtering.save_template_library("1.0", flux=np.array([[1., 2., 1.]]), time=np.arange(3.))
    source = paths.DATA_PATH / "matched_filter_templates.npz"
    source.write_bytes(b"different bank")
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        config.migrate_legacy_templates()
    assert target.read_bytes() == before
    assert source.read_bytes() == b"different bank"


def test_saved_segmentation_settings_and_legacy_defaults():
    directory = config.simulens_directory("2.0")
    assert config.cnn_inference_settings(directory) == ([np.inf, 7., 2.], 24)
    directory.mkdir(parents=True)
    (directory / "training_run.json").write_text(json.dumps({"settings": {
        "split_sizes": [None, 3., .5], "max_segments_per_item": 8}}))
    assert config.cnn_inference_settings(directory) == ([np.inf, 3., .5], 8)
    (directory / "inference.json").write_text('{"split_sizes":[5.0],"max_segments_per_item":4}')
    assert config.cnn_inference_settings(directory) == ([5.], 4)
    assert config.cnn_inference_settings(directory, split_sizes=[2.], max_segments_per_item=7) == ([2.], 7)


def test_evaluation_config_records_resolved_versions_for_reproducible_resume(tmp_path):
    from searchlight.evaluation.experiment import read_config
    config.initialize_config()
    config.config_path().write_text('{"simulens_version":"2.0","template_version":"3.0"}')
    experiment = tmp_path / "experiment.json"
    experiment.write_text('{"model_version":null,"template_version":null}')
    selected = read_config(experiment)
    assert selected["model_version"] == "2.0"
    assert selected["template_version"] == "3.0"
    experiment.write_text(json.dumps(selected))
    config.config_path().write_text('{"simulens_version":"4.0","template_version":"5.0"}')
    resumed = read_config(experiment)
    assert resumed["model_version"] == "2.0"
    assert resumed["template_version"] == "3.0"
    assert resumed["checkpoint"] == selected["checkpoint"]
    assert resumed["templates"] == selected["templates"]


def test_dataset_versions_follow_the_same_default_precedence(monkeypatch):
    assert config.dataset_directory().parts[-3:] == ("datasets", "simulens", "1.0")
    monkeypatch.setattr(config, "DEFAULT_DATASET_VERSION", "2.0")
    assert config.dataset_directory() == paths.DATA_PATH / "datasets/simulens/2.0"
    config.initialize_config()
    config.config_path().write_text('{"dataset_version":"3.0"}')
    assert config.dataset_directory() == paths.DATA_PATH / "datasets/simulens/3.0"
    assert config.dataset_directory("4.0") == paths.DATA_PATH / "datasets/simulens/4.0"
