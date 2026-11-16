"""Regression checks for the focused amplitude experiment."""
import json
from pathlib import Path

import numpy as np
import pytest

from searchlight.evaluation.datasets import EvaluationManifest, EvaluationSample
from searchlight.evaluation.preprocessing import LightCurve
from searchlight.evaluation.simple import amplitude_summary, make_injections


def test_confusion_display_normalizes_truth_rows_and_names_decisions(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    from searchlight.evaluation.simple import render
    components = ['matched_filter', 'cnn', 'combined']
    metric = dict(confusion_matrix=[[9, 1], [630, 10]], recall=10/640,
                  precision=10/11, curve=dict(recall=[0., 1.], precision=[1., .9]))
    summary = dict(rows=[dict(amplitude_ppm=100., recall=.5, lower=.2, upper=.8)],
                   required_recall=.9, measure='test recovery')
    (tmp_path/'metrics.json').write_text(json.dumps({c: {'test': metric} for c in components}))
    (tmp_path/'amplitude_recovery.json').write_text(json.dumps({'by_component': {c: summary for c in components}}))
    (tmp_path/'run.json').write_text('{}')
    figures = []
    monkeypatch.setattr(Figure, 'savefig', lambda self, *a, **kw: figures.append(self))
    render(tmp_path)
    axis = figures[0].axes[1]
    np.testing.assert_allclose(axis.images[0].get_array(), [[.9, .1], [630/640, 10/640]])
    assert [tick.get_text() for tick in axis.get_xticklabels()] == ['Rejected', 'Accepted']
    assert axis.get_ylabel() == 'Benchmark truth'
    assert [label.get_text() for label in axis.texts] == ['90.0%\n(n=9)', '10.0%\n(n=1)',
                                                       '98.4%\n(n=630)', '1.6%\n(n=10)']


def test_amplitude_limit_requires_all_higher_levels_and_ignores_tuning_and_unobservable():
    rows = []
    for amplitude, success in [(100, False), (300, True), (1000, False), (3000, True)]:
        for group in ['a', 'b']:
            rows.append(dict(split='test', observable=True, amplitude=amplitude * 1e-6,
                             group_id=group, combined_recovered=success))
    rows += [dict(split='tuning', observable=True, amplitude=.003, group_id='c', combined_recovered=False),
             dict(split='test', observable=False, amplitude=.003, group_id='d', combined_recovered=False)]
    result = amplitude_summary(rows, [100, 300, 1000, 3000], repeats=20)
    assert result['minimum_tested_amplitude_ppm'] == 3000
    assert result['rows'][-1]['events'] == 2
    assert result['rows'][-1]['lower'] == 1
    assert amplitude_summary(rows, [1000], repeats=20)['minimum_tested_amplitude_ppm'] is None
    assert amplitude_summary([], [100])['rows'][0]['recall'] is None


def test_scaled_injections_preserve_controls_uncertainties_splits_and_actual_peak(tmp_path, monkeypatch):
    from searchlight.evaluation import injections
    # Cheap asymmetric physical-interface stand-in: the flux maximum is after the epoch.
    monkeypatch.setattr(injections, 'sample_physical_system', lambda rng: dict(duration_days=.02, period_days=20))
    monkeypatch.setattr(injections, 'physical_profile', lambda params, center, **kw:
                        lambda t: .01 * np.exp(-.5 * ((t-center-.01)/.008)**2))
    time = np.arange(0, 2, 1/720)
    raw = LightCurve(time, np.full(len(time), 100.), np.full(len(time), .1),
                     np.zeros(len(time), dtype=int), np.full(len(time), 120.))
    raw.flux[0] = np.nan
    path = tmp_path / 'background.npz'
    raw.write(path)
    original = EvaluationSample('original', str(path), 'real_labelled', 'test', 'tic:1', tic='1')
    manifest = make_injections(EvaluationManifest((original,)), tmp_path / 'injections', amplitudes_ppm=[100, 1000])
    control, low, high = manifest.samples
    assert control.label == 'non-lensing'
    assert low.parent_id == high.parent_id == control.sample_id
    assert all(s.split == 'test' and s.group_id == 'tic:1' for s in manifest.samples)
    with np.load(low.source) as a, np.load(high.source) as b:
        assert np.nanmax(a['flux']-raw.flux) / 100 * 1e6 == pytest.approx(100)
        assert np.nanmax(b['flux']-raw.flux) / 100 * 1e6 == pytest.approx(1000)
        np.testing.assert_equal(a['flux_err'], raw.flux_err)
        assert np.isnan(a['flux'][0])
    diagnostic = manifest.metadata['profiles'][0]
    assert diagnostic['peak_time'] - diagnostic['reference_epoch'] == pytest.approx(.01, abs=.0001)
    assert low.parameters['events'][0]['time'] == diagnostic['peak_time']
    assert low.parameters['events'][0]['duration_days'] == pytest.approx(.008*2.355, rel=.02)
    assert (tmp_path / 'injections/manifest.json').exists()


@pytest.mark.parametrize("order", ["matched_filter_then_cnn", "cnn_then_matched_filter"])
def test_reject_all_gate_keeps_continuous_pr_ranking(tmp_path, monkeypatch, order):
    from searchlight.evaluation.simulations import create_demo, demo_templates
    from searchlight.evaluation.runner import evaluate
    from searchlight.evaluation.scoring import cnn_score
    manifest = create_demo(tmp_path / 'data', tuning_groups=2, test_groups=2)

    class CNN:
        MODEL_SAVE_PATH = tmp_path / 'weights'
        CLASS_SAVE_PATH = tmp_path / 'classes'
        classes = ['other', 'lensing']
        split_sizes = [np.inf]
        max_segments_per_item = 1

        def classify_curves(self, samples, curves, *, positive_class):
            return [cnn_score(s.sample_id, self.classes, [0, float(i)], positive_class=positive_class)
                    for i, s in enumerate(samples)]

    CNN.MODEL_SAVE_PATH.write_bytes(b'fixture')
    CNN.CLASS_SAVE_PATH.write_text('["other", "lensing"]')
    monkeypatch.setattr('searchlight.evaluation.reporting.save_figures', lambda output: None)
    result = evaluate(manifest, demo_templates(), tmp_path / 'results', cnn=CNN(),
                      preprocessing={}, bootstrap_repeats=0, pipeline_order=order,
                      fixed_thresholds={'cnn': None, 'matched_filter': 0})
    rows = json.loads((tmp_path / 'results/scores.json').read_text())
    assert len({r['combined']['score'] for r in rows}) > 1
    assert not any(r['combined']['accepted'] for r in rows)
    assert result['combined']['test']['threshold'] is None
    assert result['combined']['test']['recall'] == 0
    assert len(result['combined']['test']['curve']['precision']) > 2
    stage = json.loads((tmp_path / 'results/attrition.json').read_text())['test']
    first = 'matched_filter' if order == 'matched_filter_then_cnn' else 'cnn'
    assert stage['order'][0] == first
    assert stage['after_first'] == sum(r[first]['accepted'] for r in rows if r['split'] == 'test')
    assert stage['after_second'] == 0
    assert all(result[key]['test']['n'] == stage['input_rows'] for key in ['matched_filter', 'cnn', 'combined'])


def test_each_component_has_its_own_amplitude_result(tmp_path):
    from searchlight.evaluation.simple import summarize_results
    details, scores = [], []
    for group in ['a', 'b']:
        for amplitude in [100, 1000]:
            key = f'{group}-{amplitude}'
            # Low amplitude: CNN accepts but MF misses. High: MF finds, CNN rejects.
            mf, cnn = amplitude == 1000, amplitude == 100
            details.append(dict(sample_id=key, split='test', observable=True,
                                amplitude=amplitude*1e-6, group_id=group,
                                recovered=mf, combined_recovered=mf and cnn))
            scores.append(dict(sample_id=key, cnn={'accepted': cnn}))
    (tmp_path / 'scores.json').write_text(json.dumps(scores))
    (tmp_path / 'event_details.json').write_text(json.dumps(details))
    summary = summarize_results(tmp_path, [100, 1000], repeats=10)
    components = summary['by_component']
    assert [r['recall'] for r in components['cnn']['rows']] == [1, 0]
    assert [r['recall'] for r in components['matched_filter']['rows']] == [0, 1]
    assert [r['recall'] for r in components['combined']['rows']] == [0, 0]
    assert components['matched_filter']['minimum_tested_amplitude_ppm'] == 1000
    assert components['cnn']['minimum_tested_amplitude_ppm'] is None
    import csv
    with (tmp_path / 'amplitude_recovery.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert {r['component'] for r in rows} == {'matched_filter', 'cnn', 'combined'}


def test_repeated_runs_reserve_fresh_directories_and_preserve_existing_outputs(tmp_path):
    from searchlight.evaluation.simple import create_run_directory
    base = tmp_path / 'runs' / 'evaluation'
    first = create_run_directory(base)
    sentinel = first / 'partial-injections.txt'
    sentinel.write_text('keep this partial run')
    second = create_run_directory(base)
    third = create_run_directory(base)
    assert first == base
    assert second == base.with_name('evaluation-001')
    assert third == base.with_name('evaluation-002')
    assert sentinel.read_text() == 'keep this partial run'
    assert all(p.is_dir() for p in (first, second, third))


def test_reuse_injections_skips_preparation_and_preserves_old_files(tmp_path, monkeypatch):
    from searchlight.evaluation import simple, experiment
    old=tmp_path/'old/injections'
    old.mkdir(parents=True)
    source=tmp_path/'curve.npz'
    manifest=EvaluationManifest((EvaluationSample('control',str(source),'real_labelled','test','tic:1',tic='1'),))
    manifest.write(old/'manifest.json')
    checkpoint=tmp_path/'model.pth';checkpoint.write_bytes(b'fixture')
    templates=tmp_path/'bank.npz';templates.write_bytes(b'fixture')
    monkeypatch.setattr(simple,'check_runtime',lambda:None)
    monkeypatch.setattr(simple,'make_injections',lambda *a,**k: pytest.fail('must reuse saved injections'))
    monkeypatch.setattr(experiment,'prepare',lambda *a,**k:pytest.fail('must not prepare backgrounds'))
    captured={}
    def score(config):
        captured.update(config)
        path=Path(config['output']);path.mkdir()
        (path/'metrics.json').write_text('{}')
    monkeypatch.setattr(experiment,'score',score)
    monkeypatch.setattr(simple,'summarize_results',lambda *a,**k:{'by_component':{}})
    monkeypatch.setattr(simple,'render',lambda *a:None)
    out=tmp_path/'new'
    simple.run(dict(output=str(out),checkpoint=str(checkpoint),templates=str(templates),
                    reuse_injections=str(old/'manifest.json'),workers=1,matched_filter_backend='cpu'))
    assert captured['manifest']==str(old/'manifest.json')
    assert (out/'complete.json').exists()
    assert (old/'manifest.json').exists()
    assert not (out/'injections').exists()


@pytest.mark.parametrize("backend", [None, "cpu", "cuda_fft"])
def test_resume_cli_uses_original_settings_and_gpu_override(tmp_path, monkeypatch, backend):
    import runpy,sys
    from searchlight.evaluation import simple
    previous=tmp_path/'prior';(previous/'injections').mkdir(parents=True)
    (previous/'injections/manifest.json').write_text('{}')
    (previous/'config.json').write_text(json.dumps(dict(output=str(previous),requested_output=str(tmp_path/'evaluation'),
        seed=42,templates='/fixed-bank.npz',workers=8)))
    current=tmp_path/'current.json';current.write_text(json.dumps(dict(output='unused',seed=999)))
    captured={}
    monkeypatch.setattr(simple,'run',lambda config:captured.update(config))
    argv=['evaluate_pipeline.py',str(current),'--resume-from',str(previous),'--workers','2']
    if backend is not None:
        argv.extend(['--backend',backend])
    monkeypatch.setattr(sys,'argv',argv)
    runpy.run_path(str(Path(__file__).parents[1]/'bin/evaluate_pipeline.py'),run_name='__main__')
    assert captured['seed']==42
    assert captured['workers']==2
    assert captured['matched_filter_backend']==(backend or 'cuda_fft')
    assert captured['reuse_injections']==str(previous/'injections/manifest.json')
    assert captured['templates']=='/fixed-bank.npz'
    assert captured['output']==str(tmp_path/'evaluation')
