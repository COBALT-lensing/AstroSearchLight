"""Full configuration wiring, broad eccentric selection and periodic event truth."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from searchlight.evaluation.datasets import EvaluationManifest, EvaluationSample
from searchlight.evaluation.preprocessing import LightCurve
from searchlight.evaluation.simple import make_injections, summarize_results


def test_full_configuration_resolves_assets_and_covers_manuscript_ranges():
    from searchlight.evaluation.experiment import read_config
    from searchlight.evaluation.proposals import RV_MANUSCRIPT_RANGES
    config = read_config(Path(__file__).parents[1] / 'examples/evaluation/full.json')
    assert {k: tuple(v) for k, v in config['injection']['ranges'].items()} == RV_MANUSCRIPT_RANGES
    assert config['injection']['coverage_policy'] == 'all_events'
    assert config['shapes_per_target'] > 1
    assert config['template_version'] == '1.0'
    assert config['templates'].endswith('templates/matched_filter/1.0/templates.npz')
    assert Path(config['backgrounds']['catalog']).is_absolute()
    assert Path(config['backgrounds']['exclude_prior_runs']).is_absolute()


def test_rv_sampler_uses_corrected_eccentric_geometry_and_duration():
    pytest.importorskip('selflensing')
    from astropy import units as u
    from selflensing.models import SelfLensingSystem
    from searchlight.evaluation.proposals import sample_rv_system
    rng = np.random.default_rng(824)
    for _ in range(8):
        p = sample_rv_system(rng, proposal='rv_manuscript')
        system = SelfLensingSystem(a=p['a']*u.au, i=p['i']*u.rad,
            Macc=p['Macc']*u.M_sun, Mcomp=p['Mcomp']*u.M_sun,
            Rcomp=p['Rcomp']*u.R_sun, Teff=p['Teff']*u.K,
            e=p['e']*u.dimensionless_unscaled, periapsis_phase=p['periapsis_phase']*u.rad, d=p['d']*u.kpc)
        assert bool(system.sl_possible) and not bool(system.rlof)
        assert p['duration_days'] == pytest.approx(float(system.tau_eff.to_value(u.day)))


@pytest.mark.parametrize('proposal', ['rv_code', 'rv_manuscript'])
def test_both_mass_proposals_are_log_uniform_before_physical_rejection(monkeypatch, proposal):
    pytest.importorskip('selflensing')
    from types import SimpleNamespace
    from astropy import units as u
    import selflensing.models
    from searchlight.evaluation.proposals import sample_rv_system, sampling_distributions
    # Disable selection to test the proposal rather than the lensing-conditioned
    # population. Physical acceptance is tested independently above.
    monkeypatch.setattr(selflensing.models, 'SelfLensingSystem', lambda **kwargs:
        SimpleNamespace(sl_possible=True, rlof=False, porb=10*u.day, tau_eff=.1*u.day))
    rng = np.random.default_rng(28731)
    rows = [sample_rv_system(rng, proposal=proposal, ranges={'Macc': (.5, 50.)}) for _ in range(1024)]
    for name in ['Mcomp', 'Macc']:
        values = np.array([row[name] for row in rows])
        assert np.all((values >= .5) & (values <= 50.))
        # Equal probability per decade; a uniform linear draw would put only
        # about 9% below 5 solar masses, and decisively fail this check.
        assert .45 < np.mean(values < 5.) < .55
        assert 4. < np.median(values) < 6.
        assert sampling_distributions(proposal)[name] == 'log_uniform'


def test_unrestricted_injections_keep_repeated_and_unsampled_events(tmp_path, monkeypatch):
    from searchlight.evaluation import injections
    calls = []
    def sample(rng, **options):
        calls.append(options)
        return dict(duration_days=.03, period_days=.5)
    def profile(params, center, **options):
        return lambda time: .01 * np.exp(-.5 * (((time-center+.25) % .5-.25)/.012)**2)
    monkeypatch.setattr(injections, 'sample_physical_system', sample)
    monkeypatch.setattr(injections, 'physical_profile', profile)
    time = np.arange(0, 2, 1/720)
    raw = LightCurve(time, np.ones(len(time)), np.full(len(time), .001),
        np.ones(len(time), dtype=int), np.full(len(time), 120.))
    # Isolated usable blocks; proposal reference epochs can fall between them.
    raw.quality[(time < .4) | (time > 1.6)] = 0
    source = tmp_path / 'background.npz'
    raw.write(source)
    manifest = EvaluationManifest((EvaluationSample('target', str(source), 'real_labelled', 'test', 'tic:1', tic='1'),))
    result = make_injections(manifest, tmp_path / 'injections', amplitudes_ppm=[1000], seed=3,
        injection={'proposal': 'rv_manuscript', 'ranges': {'a': [.1, 200]}, 'coverage_policy': 'all_events', 'tail_error_budget_ppm': 1e-5})
    assert calls == [{'proposal': 'rv_manuscript', 'ranges': {'a': [.1, 200]}}]
    injected = result.samples[-1]
    assert injected.parameters['tail_error_bound_ppm'] <= 1e-5 * (1 + 1e-12)
    assert len(injected.parameters['events']) >= 3
    periods = np.diff([e['time'] for e in injected.parameters['events']])
    np.testing.assert_allclose(periods, .5)
    assert result.metadata['injection']['coverage_policy'] == 'all_events'
    assert result.metadata['profiles'][0]['prior_coverage_rejections'] == 0
    # Unlike an observed-peak normalization, the reference-peak convention is
    # independent of flagged cadences, gaps and chance sample/peak alignment.
    assert 'independent of coverage' in result.metadata['amplitude_convention']


def test_cnn_amplitude_recall_counts_each_curve_once_despite_periodic_truth(tmp_path):
    events = [dict(sample_id='short', split='test', observable=True, amplitude=.001,
                   group_id='a', recovered=True, combined_recovered=True)] * 10
    events += [dict(sample_id='long', split='test', observable=True, amplitude=.001,
                    group_id='b', recovered=False, combined_recovered=False)]
    (tmp_path/'event_details.json').write_text(json.dumps(events))
    (tmp_path/'scores.json').write_text(json.dumps([
        {'sample_id':'short','cnn':{'accepted':True}}, {'sample_id':'long','cnn':{'accepted':False}}]))
    summary = summarize_results(tmp_path, [1000], repeats=10)
    assert summary['by_component']['cnn']['rows'][0]['recall'] == .5
    assert summary['by_component']['matched_filter']['rows'][0]['recall'] == pytest.approx(10/11)


def test_full_cohort_filters_strata_and_excludes_pilot_targets(tmp_path, monkeypatch):
    from searchlight.evaluation import cohorts
    rows = [dict(tic=str(100*s+i), sector=s, cadence_seconds=120., source=str(tmp_path/f'{s}-{i}.fits'))
            for s in [1, 2, 3] for i in range(5)]
    catalogue = tmp_path/'catalog.json'
    catalogue.write_text(json.dumps(rows))
    exclude = tmp_path/'pilot.json'
    EvaluationManifest((EvaluationSample('pilot', '/pilot.fits', 'real_labelled', 'test',
        'tic:100', tic='100'),)).write(exclude)
    def prepare(config, output):
        return EvaluationManifest(tuple(EvaluationSample(r['sample_id'], r['source'], 'real_labelled',
            'test', 'tic:'+r['tic'], tic=r['tic'], sector=r['sector']) for r in config['sources']))
    monkeypatch.setattr(cohorts, 'prepare_sources', prepare)
    monkeypatch.setattr(cohorts, 'read_lightcurve', lambda source: LightCurve(np.arange(10), np.ones(10),
        np.full(10,.001), np.zeros(10,dtype=int), np.full(10,120.)))
    result = cohorts.prepare_cohort(dict(output=str(tmp_path/'cohort'), catalog=str(catalogue),
        exclude_manifest=str(exclude), sectors=[1,2], cadences_seconds=[120], targets_per_stratum=4))
    assert len(result.samples) == 8
    assert {s.sector for s in result.samples} == {1,2}
    assert '100' not in {s.tic for s in result.samples}


def test_missing_catalogue_fetches_once_and_caches_for_reuse(tmp_path, monkeypatch):
    from searchlight.evaluation import cohorts
    records = [dict(tic='123', sector=1, cadence_seconds=120., source='mast:example_lc.fits')]
    calls = []
    def fetch(sectors, cadences):
        calls.append((sectors, cadences))
        return records
    monkeypatch.setattr(cohorts, 'mast_catalog', fetch)
    config = dict(catalog=str(tmp_path/'new-cache/catalog.json'), sectors=[1], cadences_seconds=[120.])
    assert cohorts.load_catalog(config) == (records, 'mast_query')
    assert cohorts.load_catalog(config) == (records, 'saved_catalogue')
    assert calls == [([1], [120.])]
    assert not list((tmp_path/'new-cache').glob('tmp*'))


def test_prior_run_exclusions_do_not_require_old_run_directories(tmp_path):
    from searchlight.evaluation.cohorts import excluded_targets
    config = dict(exclude_prior_runs=str(tmp_path/'runs'))
    assert excluded_targets(config) == set()
    prior = tmp_path/'runs/simple-evaluation/results/manifest.json'
    prior.parent.mkdir(parents=True)
    EvaluationManifest((EvaluationSample('control', '/control.npz', 'real_labelled',
        'test', 'tic:123', tic='123'),)).write(prior)
    assert excluded_targets(config) == {'123'}
