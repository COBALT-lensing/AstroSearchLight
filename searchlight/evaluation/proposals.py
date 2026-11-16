"""RV-range proposals with log-uniform masses; these are not population priors."""

import numpy as np
from astropy import constants as c

RV_CODE_RANGES = {
    'a': (.1, 200.), 'i': (0., np.pi/2), 'Macc': (.1, 50.),
    'Mcomp': (.5, 50.), 'e': (0., .999999), 'periapsis_phase': (0., 2*np.pi), 'd': (.1, 100.),
}
RV_MANUSCRIPT_RANGES = {**RV_CODE_RANGES, 'i': tuple(np.deg2rad([89.99, 90.])), 'Macc': (.5, 50.)}


def sampling_distributions(proposal):
    """Describe proposed draws, before lensing/Roche-overflow rejection."""
    common = dict(a='log_uniform', Macc='log_uniform', Mcomp='log_uniform',
                  periapsis_phase='uniform')
    if proposal in {'rv_code', 'rv_manuscript'}:
        return {**common, 'e': 'uniform_log1p', 'i': 'uniform_angle', 'd': 'uniform'}
    if proposal == 'evaluation':
        return {**common, 'e': 'uniform', 'cos_i': 'uniform'}
    raise ValueError('Unknown physical proposal: ' + str(proposal))


def sample_rv_system(rng, *, ranges=None, max_attempts=10000, proposal='rv_code'):
    """RV generator distributions conditional on lensing and no Roche overflow.

    a/Macc/Mcomp log-uniform, e log-uniform in 1+e, others uniform. Source mass
    intentionally differs from the RV generator's original uniform draw.
    Fractional light-curve injection does not turn distance into observed noise.
    """
    if proposal not in {'rv_code', 'rv_manuscript'}:
        raise ValueError('Unknown RV proposal')
    bounds = dict(RV_CODE_RANGES if proposal == 'rv_code' else RV_MANUSCRIPT_RANGES)
    if set(ranges or {}) - set(bounds):
        raise ValueError('Unknown RV parameter range')
    bounds.update(ranges or {})
    for name, pair in bounds.items():
        if len(pair) != 2 or not np.isfinite(pair).all() or pair[0] >= pair[1]:
            raise ValueError('Invalid RV parameter range: '+name)
    if (any(bounds[k][0] <= 0 for k in ['a','Macc','Mcomp','d'])
            or not 0 <= bounds['e'][0] < bounds['e'][1] < 1
            or not 0 <= bounds['i'][0] < bounds['i'][1] <= np.pi/2):
        raise ValueError('RV ranges outside physical bounds')
    for _ in range(max_attempts):
        p = {}
        for name, (low, high) in bounds.items():
            p[name] = float(np.exp(rng.uniform(np.log(low), np.log(high)))) if name in {'a','Macc','Mcomp'} else (
                float(np.expm1(rng.uniform(np.log1p(low), np.log1p(high)))) if name == 'e'
                else float(rng.uniform(low, high)))
        from astropy import units as u
        from selflensing.models import SelfLensingSystem
        radius = p['Mcomp']**.8
        temperature = float((p['Mcomp']**3.5*c.L_sun.value /
            (4*np.pi*c.sigma_sb.value*(radius*c.R_sun.value)**2))**.25)
        system = SelfLensingSystem(a=p['a']*u.au, i=p['i']*u.rad,
            Macc=p['Macc']*u.M_sun, Mcomp=p['Mcomp']*u.M_sun,
            Rcomp=radius*u.R_sun, Teff=temperature*u.K,
            e=p['e']*u.dimensionless_unscaled, periapsis_phase=p['periapsis_phase']*u.rad,
            d=p['d']*u.kpc)
        # Use the installed, pinned full observer-frame geometry. The old a*cos(i)
        # criterion and circular speed expression are invalid for this e range.
        if not bool(system.sl_possible) or bool(system.rlof):
            continue
        period = float(np.asarray(system.porb.to_value(u.day)))
        duration = float(np.asarray(system.tau_eff.to_value(u.day)))
        if not np.isfinite(duration) or duration <= 0:
            continue
        p.update(Rcomp=radius, Teff=temperature, period_days=period, duration_days=duration)
        return p
    raise RuntimeError('No viable RV system within max_attempts; revise proposal or increase budget')
