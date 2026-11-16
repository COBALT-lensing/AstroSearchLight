"""Injection sampling, tail truncation and cohort bookkeeping."""

import json
from pathlib import Path

import numpy as np
import pytest

from searchlight.evaluation.proposals import sample_rv_system, RV_MANUSCRIPT_RANGES
from searchlight.evaluation.cohorts import select_products
from searchlight.evaluation.assets import physical_identity
from searchlight.evaluation.datasets import EvaluationManifest, EvaluationSample


def test_tail_bound_broadcasts_radii_and_decreases_with_distance():
    from searchlight.evaluation.injections import magnification_tail_bound_ppm
    separation = np.array([.5, 1., 1.1, 2., 5.])
    bound = magnification_tail_bound_ppm(separation, .1)
    assert np.isinf(bound[:2]).all()
    assert (bound[2:] > 0).all()
    assert (np.diff(bound[2:]) < 0).all()
    larger = magnification_tail_bound_ppm(separation[2:], .2)
    assert (larger > bound[2:]).all()
    ratios = np.array([.1, .2, .3])
    np.testing.assert_allclose(magnification_tail_bound_ppm(separation[2:], ratios),
        [magnification_tail_bound_ppm(distance, q) for distance, q in zip(separation[2:], ratios)])
    with pytest.raises(ValueError):
        magnification_tail_bound_ppm(separation, 0.)


def test_tail_shortcut_skips_only_bounded_offdisc_evaluations(monkeypatch):
    pytest.importorskip('selflensing')
    import rustlens
    from searchlight.evaluation.injections import physical_profile
    actual = rustlens.integrated_witt_mao_magnification
    calls = []
    def record(offsets, re, radius):
        calls.extend(offsets)
        return actual(offsets, re, radius)
    monkeypatch.setattr(rustlens, 'integrated_witt_mao_magnification', record)
    p = dict(a=1.,i=np.pi/2,Macc=10.,Mcomp=1.,Rcomp=1.,Teff=5800.,e=0.,periapsis_phase=0.)
    result = physical_profile(p, 0., tail_cutoff_ppm=.01)(np.array([0., 5.]))
    assert len(calls) == 1
    assert result[0] > .01
    assert result[1] == 0.


def test_rv_manuscript_proposal_is_seeded_and_within_paper_ranges():
    pytest.importorskip('selflensing')
    first, second = np.random.default_rng(97), np.random.default_rng(97)
    for _ in range(10):
        p = sample_rv_system(first, proposal='rv_manuscript')
        assert p == sample_rv_system(second, proposal='rv_manuscript')
        for name, (low, high) in RV_MANUSCRIPT_RANGES.items():
            assert low <= p[name] <= high
        assert p['duration_days'] > 0
        assert p['Rcomp'] == pytest.approx(p['Mcomp']**.8)
    with pytest.raises(ValueError):
        sample_rv_system(first, ranges={'Macc': [-1, 2]}, proposal='rv_manuscript')


def test_catalogue_selection_is_order_independent_and_uses_target_strata():
    rows = [dict(tic=str(i), sector=s, cadence_seconds=120., source=f'mast:{s}:{i}_lc.fits')
            for s in [1, 13] for i in range(10)]
    a = select_products(rows, per_stratum=3, seed=2)
    b = select_products(list(reversed(rows)), per_stratum=3, seed=2)
    assert a == b
    assert len(a) == 6
    assert {r['sector'] for r in a} == {1, 13}
    assert len({(r['tic'],r['sector']) for r in a}) == len(a)


def test_physical_identity_ignores_observation_distance_and_epoch():
    p = dict(a=1.,i=1.57,Macc=10.,Mcomp=1.,Rcomp=1.,e=.2,periapsis_phase=.4,d=1.)
    assert physical_identity(p) == physical_identity({**p,'d':50.,'event_time':123.})
    assert physical_identity(p) != physical_identity({**p,'Macc':11.})
    assert physical_identity(None) is None


