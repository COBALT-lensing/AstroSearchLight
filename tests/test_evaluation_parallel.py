"""Spawn-pool ordering, errors and serial/parallel detector parity."""
import json
import os

import numpy as np
import pytest

from searchlight.evaluation.parallel import process_map, worker_count


def _worker_info(value):
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 1
    return value * value, cpus, os.environ['OMP_NUM_THREADS']


def _fail(value):
    raise ValueError(f'bad task {value}')


def test_spawn_order_affinity_thread_limits_and_parent_environment(monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS', '7')
    rows = process_map(_worker_info, [3, 1, 2], workers=2)
    assert rows == [(9, 1, '1'), (1, 1, '1'), (4, 1, '1')]
    assert os.environ['OMP_NUM_THREADS'] == '7'
    with pytest.raises(ValueError, match='bad task'):
        process_map(_fail, [1, 2], workers=2)
    assert os.environ['OMP_NUM_THREADS'] == '7'
    assert worker_count('auto', 2) <= 2
    with pytest.raises(ValueError):
        worker_count(0)


def test_parallel_matched_filter_preserves_scores_thresholds_and_matches(tmp_path, monkeypatch):
    from searchlight.evaluation.simulations import create_demo, demo_templates
    from searchlight.evaluation.runner import evaluate
    monkeypatch.setattr('searchlight.evaluation.reporting.save_figures', lambda directory: None)
    manifest = create_demo(tmp_path / 'data', tuning_groups=2, test_groups=2)
    from searchlight.matched_filtering import _save_template_library_file as save_template_library
    bank = demo_templates()
    path = tmp_path / 'templates.npz'
    save_template_library(path, flux=np.array([t.flux for t in bank]), time=bank[0].time)
    for workers in [1, 2]:
        evaluate(manifest, demo_templates(), tmp_path / str(workers), preprocessing={},
                 normalize_scores=True, bootstrap_repeats=10, workers=workers, template_path=path)
    for name in ['scores', 'metrics', 'event_details', 'events', 'event_summary', 'preprocessing']:
        one = json.loads((tmp_path / '1' / f'{name}.json').read_text())
        two = json.loads((tmp_path / '2' / f'{name}.json').read_text())
        assert one == two, name
