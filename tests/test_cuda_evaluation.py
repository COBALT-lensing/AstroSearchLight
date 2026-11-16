"""FFT acceleration preserves full-template normalization and detection cuts."""
from dataclasses import asdict
import numpy as np
import pytest

from searchlight.matched_filtering import MatchedFilterTemplate, detect_matched_filter_peaks
from searchlight.evaluation.scoring import score_matched_filter
from searchlight.evaluation.cuda_filter import FFTBank


@pytest.mark.parametrize('n,m,shift', [(21,101,0), (121,31,0), (31,32,0), (30,102,13), (40,81,-20)])
def test_fft_scores_and_events_match_production(n,m,shift):
    rng = np.random.default_rng(43)
    dt = 1/720
    template_time = (np.arange(m)-m//2+shift)*dt
    templates = [MatchedFilterTemplate(1 + .05*np.exp(-.5*(template_time/(width*dt))**2), template_time,
                                      event_half_width=8*dt) for width in [2,4,7]]
    templates.append(MatchedFilterTemplate(1-.1*np.exp(-.5*(template_time/(3*dt))**2), template_time))
    time = np.arange(n)*dt
    flux = 1 + rng.normal(0,.001,n) + .03*np.exp(-.5*((np.arange(n)-n//2)/3)**2)
    bank = FFTBank(templates, device='cpu', batch_size=2)
    one, two = score_matched_filter('x',time,flux,templates), bank.score('x',time,flux)
    assert one.template_index == two.template_index
    assert one.peak_time == two.peak_time
    assert one.score == pytest.approx(two.score, abs=1e-12)
    for height in [0, .005, .1]:
        expected = detect_matched_filter_peaks(time,flux,templates,height=height,prominence=0,min_score_sigma=None,dedupe_overlaps=True,backend='cpu')
        actual = bank.detect(time,flux,height=height)
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):
            assert a.template_index==b.template_index
            assert a.peak_time==b.peak_time
            assert a.start_index==b.start_index and a.end_index==b.end_index
            assert a.score==pytest.approx(b.score,abs=1e-12)
            assert a.prominence==pytest.approx(b.prominence,abs=1e-12)


def test_fft_resampling_and_full_normalization_are_preserved():
    dt=1/720
    source_time=np.arange(-30,31)*5*dt
    templates=[MatchedFilterTemplate(1+.001*np.exp(-.5*(source_time/(width*dt))**2),source_time)
               for width in [3,10,25]]
    time=np.arange(43)*dt
    flux=1+.01*np.exp(-.5*((np.arange(43)-3)/2)**2)+np.random.default_rng(3).normal(0,.001,43)
    bank=FFTBank(templates,device='cpu',batch_size=2,cache_gib=.0001)
    first=score_matched_filter('x',time,flux,templates)
    second=bank.score('x',time,flux)
    assert first.score==pytest.approx(second.score,rel=1e-10,abs=1e-12)
    assert first.peak_time==second.peak_time
    expected=detect_matched_filter_peaks(time,flux,templates,height=0,prominence=0,min_score_sigma=None,backend='cpu')
    actual=bank.detect(time,flux,height=0)
    assert [r.peak_time for r in expected]==[r.peak_time for r in actual]
    np.testing.assert_allclose([r.score for r in expected],[r.score for r in actual],rtol=1e-10,atol=1e-12)
    bank.prepare(dt*1.01,42)
    assert bank.cache_bytes <= bank.cache_limit


def test_compact_kernel_cache_spills_to_bounded_host_cache_and_reuses_it(monkeypatch):
    dt=1/720
    t=np.arange(-40,41)*dt
    templates=[MatchedFilterTemplate(1+.01*np.exp(-.5*(t/(5*dt))**2),t)]
    bank=FFTBank(templates,device='cpu',cache_gib=0.000003,host_cache_gib=.001)
    a=bank.prepare(dt,30)
    bank.prepare(dt*1.001,30)
    assert bank.host_cache
    monkeypatch.setattr(bank,'_grid',lambda cadence: (_ for _ in ()).throw(AssertionError('must reuse cached preparation')))
    b=bank.prepare(dt,30)
    np.testing.assert_array_equal(a['kernels'].numpy(),b['kernels'].numpy())
    assert bank.cache_bytes<=bank.cache_limit
    assert bank.host_cache_bytes<=bank.host_cache_limit


def test_production_center_flux_cut_can_skip_expensive_bank_work(monkeypatch):
    dt=1/720
    t=np.arange(-40,41)*dt
    bank=FFTBank([MatchedFilterTemplate(1+.01*np.exp(-.5*(t/(5*dt))**2),t)],device='cpu')
    monkeypatch.setattr(bank,'prepare',lambda *a: (_ for _ in ()).throw(AssertionError('should reject before preparation')))
    assert bank.detect(np.arange(200)*dt,np.ones(200),height=0)==[]
