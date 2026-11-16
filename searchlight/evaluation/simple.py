"""Configurable real-background physical injection and amplitude evaluation."""
from dataclasses import replace
import csv
import json
from pathlib import Path

import numpy as np

from .datasets import EvaluationManifest
from .preprocessing import PreprocessingConfig, read_lightcurve, preprocess
from .runner import write_json


def amplitude_summary(details, amplitudes, *, repeats=500, seed=42, required_recall=.9, success_key="combined_recovered"):
    """Whole-target bootstrap at each amplitude; no interpolation or extrapolation."""
    if not 0 < required_recall <= 1 or repeats < 0:
        raise ValueError("required_recall must be in (0, 1] and repeats nonnegative")
    rng = np.random.default_rng(seed)
    rows = []
    for amplitude in amplitudes:
        selected = [r for r in details if r['split'] == 'test' and r['observable']
                    and np.isclose(r['amplitude'] * 1e6, amplitude)]
        groups = sorted({r['group_id'] for r in selected})
        values = {g: [float(r[success_key]) for r in selected if r['group_id'] == g] for g in groups}
        draws = []
        if len(groups) >= 2:
            for _ in range(repeats):
                sample = rng.choice(groups, len(groups), replace=True)
                draws.append(np.mean([v for g in sample for v in values[g]]))
        count = sum(r[success_key] for r in selected)
        interval = np.quantile(draws, [.025, .975]).tolist() if draws else [None, None]
        rows.append(dict(amplitude_ppm=float(amplitude), events=len(selected), targets=len(groups),
                         recovered=int(count), recall=count / len(selected) if selected else None,
                         lower=interval[0], upper=interval[1]))
    # Require every higher tested amplitude to pass too: empirical curves can be non-monotonic.
    qualifying = [r['amplitude_ppm'] for i, r in enumerate(rows)
                  if all(x['recall'] is not None and x['recall'] >= required_recall for x in rows[i:])]
    return {'required_recall': required_recall, 'minimum_tested_amplitude_ppm': min(qualifying) if qualifying else None,
            'definition': 'Lowest tested amplitude meeting the recall target at this and all higher tested amplitudes; point estimate, conditional on observable events',
            'rows': rows}


COMPONENT_LABELS = {'matched_filter': 'Matched filter', 'cnn': 'CNN', 'combined': 'Matched filter → CNN'}


def summarize_results(results, amplitudes, *, seed=42, required_recall=.9, repeats=500):
    """Export the same amplitude experiment for each detector and their cascade."""
    results = Path(results)
    scores = {r['sample_id']: r for r in json.loads((results / 'scores.json').read_text())}
    details = json.loads((results / 'event_details.json').read_text())
    for event in details:
        event['cnn_accepted'] = scores[event['sample_id']]['cnn']['accepted']
    summaries = {}
    for component, key in [('matched_filter', 'recovered'), ('cnn', 'cnn_accepted'), ('combined', 'combined_recovered')]:
        component_details = details
        if component == 'cnn':
            eligible = {r['sample_id']: r for r in details if r['observable']}
            component_details = list(eligible.values())
        summaries[component] = amplitude_summary(component_details, amplitudes, seed=seed,
            required_recall=required_recall, repeats=repeats, success_key=key)
        summaries[component]['measure'] = 'Injected-curve acceptance' if component == 'cnn' else 'Localized event recovery'
    # Preserve the combined top-level fields for existing result readers.
    summary = {**summaries['combined'], 'by_component': summaries}
    write_json(results / 'amplitude_recovery.json', summary)
    with (results / 'amplitude_recovery.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['component', 'measure', *summary['rows'][0]])
        writer.writeheader()
        for component, values in summaries.items():
            writer.writerows({'component': component, 'measure': values['measure'], **row} for row in values['rows'])
    return summary


def _inject_target(job):
    from .injections import sample_physical_system, physical_profile, integrate_profile, PHYSICAL_GEOMETRY
    import hashlib
    original, output, amplitudes_ppm, shapes, seed, settings, injection = job
    # Target identity, rather than scheduling or the preceding target's draws,
    # determines every random decision. Serial and parallel runs are identical.
    target_seed = int.from_bytes(hashlib.sha256(f'{seed}:{original.sample_id}'.encode()).digest()[:8], 'big')
    rng = np.random.default_rng(target_seed)
    rows, diagnostics = [], []
    raw = read_lightcurve(original.source, settings)
    clean, segments, _ = preprocess(raw, settings)
    control_path = output / f'{original.sample_id}-control.npz'
    raw.write(control_path)
    rows.append(replace(original, source=str(control_path.resolve()), label='non-lensing',
                        label_provenance='Uninjected benchmark control; astrophysical label unknown'))
    span = float(np.ptp(clean.time))
    rejected = 0
    coverage_policy = injection.get('coverage_policy', 'isolated')
    nodes = injection.get('nodes', 31)
    for shape in range(shapes):
        for attempt in range(injection.get('max_attempts', 1000)):
            options = {key: injection[key] for key in ('ranges', 'proposal') if key in injection}
            params = sample_physical_system(rng, **options)
            width = params['duration_days']
            eligible = [s for s in segments if np.ptp(s.time) > 8 * width]
            if coverage_policy == 'all_events' or (params['period_days'] > 2 * span and eligible
                    and width > 3 * np.median(clean.exposure_seconds) / 86400):
                break
            rejected += 1
        else:
            raise ValueError('No sufficiently sampled isolated physical event; use longer/higher-cadence backgrounds')
        if coverage_policy == 'isolated':
            segment = eligible[int(rng.integers(len(eligible)))]
            center = rng.uniform(segment.time[0] + 4 * width, segment.time[-1] - 4 * width)
        else:
            center = rng.uniform(clean.time[0], clean.time[-1])
        profile = physical_profile(params, center, tail_cutoff_ppm=.001)
        finite = np.isfinite(raw.time) & np.isfinite(raw.flux) & np.isfinite(raw.exposure_seconds) & (raw.exposure_seconds > 0)
        signal = np.zeros(len(raw.time))
        signal[finite] = integrate_profile(raw.time[finite], raw.exposure_seconds[finite], profile, nodes=nodes)
        # Locate the true profile peak and half-height width, rather than equating tau_eff with FWHM.
        cadence = float(np.median(clean.exposure_seconds)) / 86400
        half_window = min(max(3 * width, 2 * cadence), .45 * params['period_days'])
        grid = np.linspace(center - half_window, center + half_window, injection.get('profile_grid_points', 2049))
        smooth = integrate_profile(grid, np.full(len(grid), np.median(clean.exposure_seconds)), profile, nodes=nodes)
        peak = float(grid[np.argmax(smooth)])
        above = grid[smooth >= smooth.max() / 2]
        fwhm = float(above[-1] - above[0])
        baseline = float(np.median(raw.flux[finite]))
        if baseline <= 0:
            raise ValueError("Background median flux must be positive")
        if (not np.isfinite(signal).all() or smooth.max() <= 0 or fwhm <= 0
                or (coverage_policy == 'isolated' and signal.max() <= 0)):
            raise ValueError('Invalid physical injection profile')
        tail_cutoff = .001
        if coverage_policy == 'all_events':
            budget = injection.get('tail_error_budget_ppm', .01)
            tail_cutoff = min(tail_cutoff, budget * float(smooth.max()) * 1e6 / max(amplitudes_ppm))
            if tail_cutoff < .001:
                bounded_profile = physical_profile(params, center, tail_cutoff_ppm=tail_cutoff)
                signal[finite] = integrate_profile(raw.time[finite], raw.exposure_seconds[finite], bounded_profile, nodes=nodes)
        for index, amplitude in enumerate(amplitudes_ppm):
            # For unrestricted events, amplitude must not depend on whether a peak
            # happens to fall in a gap or beyond the observing boundary.
            reference_peak = smooth.max() if coverage_policy == 'all_events' else signal.max()
            factor = amplitude * 1e-6 / reference_peak
            key = f'{original.sample_id}-shape{shape}-amp{index}'
            path = output / f'{key}.npz'
            replace(raw, flux=raw.flux + baseline * signal * factor).write(path)
            if coverage_policy == 'all_events':
                period = params['period_days']
                first = int(np.ceil((clean.time[0] - fwhm - peak) / period))
                last = int(np.floor((clean.time[-1] + fwhm - peak) / period))
                events = [dict(time=peak + cycle * period, duration_days=fwhm,
                    tolerance_days=max(2 * cadence, fwhm / 2), amplitude=float(amplitude * 1e-6))
                    for cycle in range(first, last + 1)]
            else:
                events = [dict(time=peak, duration_days=fwhm, tolerance_days=max(2 * cadence, fwhm / 2),
                               amplitude=float(amplitude * 1e-6))]
            rows.append(replace(original, sample_id=key, source=str(path.resolve()),
                suite='spoc_injection', parent_id=original.sample_id, label='lensing',
                label_provenance='Exposure-integrated, amplitude-scaled physical injection',
                parameters={'events': events, 'amplitude': float(amplitude * 1e-6), 'physical_parameters': params,
                            'amplitude_scale': float(factor), 'geometry_model': PHYSICAL_GEOMETRY,
                            'integration_nodes': nodes, 'tail_error_bound_ppm': tail_cutoff * float(factor)}))
        diagnostics.append(dict(sample_id=original.sample_id, shape=shape, physical_parameters=params,
                                peak_time=peak, fwhm_days=fwhm, reference_epoch=center,
                                prior_coverage_rejections=rejected))
    return rows, diagnostics


def make_injections(manifest, output, *, amplitudes_ppm, shapes=1, seed=42, preprocessing=None, workers=1, injection=None):
    """Scale independent physical profiles to measured exposure-averaged amplitudes.

    This is a morphology/noise sensitivity test, not a binary-population simulation.
    One isolated event per example; peak time and FWHM come from the sampled
    exposure-averaged profile. The native physical source shape is saved.
    """
    from .provenance import dependency_provenance
    from .proposals import sampling_distributions
    injection = dict(injection or {})
    unknown = set(injection) - {'proposal', 'ranges', 'coverage_policy', 'nodes', 'max_attempts', 'profile_grid_points', 'tail_error_budget_ppm'}
    if unknown:
        raise ValueError(f'Unknown injection options: {sorted(unknown)}')
    if injection.get('coverage_policy', 'isolated') not in {'isolated', 'all_events'}:
        raise ValueError('coverage_policy must be isolated or all_events')
    if not np.isfinite(injection.get('tail_error_budget_ppm', .01)) or injection.get('tail_error_budget_ppm', .01) <= 0:
        raise ValueError('tail_error_budget_ppm must be finite and positive')
    if injection.get('nodes', 31) < 2 or injection.get('profile_grid_points', 2049) < 33:
        raise ValueError('At least two integration nodes and 33 profile grid points required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    settings = PreprocessingConfig(**(preprocessing or {}))
    rows = []
    diagnostics = []
    from .parallel import process_map
    jobs = [(original, output, amplitudes_ppm, shapes, seed, settings, injection) for original in manifest.samples]
    for target_rows, target_diagnostics in process_map(_inject_target, jobs, workers=workers,
            desc='Inject physical profiles', progress=True):
        rows.extend(target_rows)
        diagnostics.extend(target_diagnostics)
    result = EvaluationManifest(tuple(rows), metadata={
        'time_unit': 'day', 'seed': seed, 'seed_scheme': 'sha256-target-v1', 'amplitudes_ppm': amplitudes_ppm,
        'scope': 'Amplitude-scaled physical morphologies in delivered SPOC flux; conditional sensitivity, not population completeness',
        'injection': injection,
        'sampling_distributions': sampling_distributions(injection.get('proposal', 'evaluation')),
        'amplitude_convention': 'Reference exposure-averaged peak, independent of coverage' if injection.get('coverage_policy') == 'all_events' else 'Maximum sampled exposure-averaged signal',
        'controls': 'Uninjected controls are benchmark negatives, not independently confirmed non-lensing objects',
        'simulator': dependency_provenance('selflensing', 'rustlens', 'jaxoplanet'),
        'event_policy': 'Exposure-averaged peak; observability half-window = measured FWHM; matching tolerance = max(2 cadences, FWHM/2)',
        'profiles': diagnostics})
    result.write(output / 'manifest.json')
    return result


def check_runtime():
    """Validate numerical settings needed by the evaluation."""
    import jax
    if not jax.config.x64_enabled:
        raise RuntimeError('Physical evaluation requires JAX_ENABLE_X64=true')


def create_run_directory(requested):
    """Atomically reserve a fresh directory without overwriting previous runs."""
    requested = Path(requested).expanduser().resolve()
    candidate, index = requested, 0
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            index += 1
            candidate = requested.with_name(f'{requested.name}-{index:03d}')


def run(config):
    config = {"matched_filter_backend": "cuda_fft", **config}
    from .experiment import prepare, score
    from .simulations import create_physical_templates
    from .parallel import worker_count
    workers = worker_count(config.get('workers', 'auto'))
    cnn_workers = min(workers, worker_count(config.get('cnn_workers', min(workers, 8))))
    if not str(config.get('cnn', {}).get('device', 'cpu')).startswith('cpu'):
        cnn_workers = 1
    print(f'CPU processes: up to {workers}; CPU CNN processes: up to {cnn_workers}', flush=True)
    check_runtime()
    if config.get('matched_filter_backend') == 'cuda_fft':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('cuda_fft requires a CUDA GPU visible to PyTorch; no GPU is available in this process')
        print(f'GPU matched filter: {torch.cuda.get_device_name(0)} (float64 FFT)', flush=True)
    output = Path(config['output']).resolve()
    from .. import config as artifact_config
    config = dict(config)
    if "model_version" in config or not config.get("checkpoint"):
        config["model_version"] = artifact_config.resolve_version("simulens", config.get("model_version"))
        config["checkpoint"] = str(artifact_config.simulens_directory(config["model_version"]) / "model_weights.pth")
    if "template_version" in config:
        config["template_version"] = artifact_config.resolve_version("template", config["template_version"])
        config["templates"] = str(artifact_config.template_library_path(config["template_version"]))
    checkpoint = Path(config['checkpoint']).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'CNN checkpoint missing: {checkpoint}')
    if config.get('templates') and not Path(config['templates']).is_file():
        raise FileNotFoundError(f"Template bank missing: {config['templates']}")
    amplitudes = sorted(set(float(a) for a in config.get('amplitudes_ppm', [100, 300, 1000, 3000, 10000])))
    if not amplitudes or not np.isfinite(amplitudes).all() or min(amplitudes) <= 0:
        raise ValueError('amplitudes_ppm must contain positive finite values')
    if config.get('shapes_per_target', 1) < 1:
        raise ValueError('At least one shape per target is required')
    requested_output = output
    output = create_run_directory(requested_output)
    if output != requested_output:
        print(f'Output already exists: {requested_output}; starting a fresh run.', flush=True)
    print(f'Run directory: {output}', flush=True)
    config = {**config, 'requested_output': str(requested_output), 'output': str(output)}
    seed = config.get('seed', 42)
    settings = config.get('preprocessing', {})
    write_json(output / 'config.json', {**config, 'resolved_workers': workers, 'resolved_cnn_workers': cnn_workers})
    if config.get('reuse_injections'):
        injection_path = Path(config['reuse_injections'])
        injected = EvaluationManifest.read(injection_path)
        injection_output = injection_path.parent
        template_path = Path(config['templates'])
        print(f'[1–3/5] Reusing completed injections: {injection_path}', flush=True)
    else:
        print('[1/5] Prepare real SPOC backgrounds', flush=True)
        if config.get('backgrounds', {}).get('kind') == 'catalogue':
            from .cohorts import prepare_cohort
            manifest = prepare_cohort({**config['backgrounds'], 'output': str(output / 'backgrounds'),
                                       'preprocessing': settings})
        elif config.get('manifest'):
            manifest = EvaluationManifest.read(config['manifest'])
        else:
            manifest = prepare({**config.get('backgrounds', {}), 'output': str(output / 'prepared'), 'preprocessing': settings, 'progress': True})
        manifest.validate()
        for split in ('tuning', 'test'):
            if len({s.group_id for s in manifest.samples if s.split == split}) < 2:
                raise ValueError(f'Provide at least two independent {split} targets')
        if any(s.parent_id or s.label == 'lensing' or s.suite == 'simulated' for s in manifest.samples):
            raise ValueError('Input must contain original real backgrounds, not injected or confirmed-positive examples')
        if config.get('templates'):
            from ..matched_filtering import _load_template_library_file as load_template_library
            template_path = Path(config['templates'])
            print(f'[2/5] Load fixed template bank: {template_path}', flush=True)
            templates = load_template_library(template_path)
        else:
            print('[2/5] Generate independent corrected physical templates', flush=True)
            template_path = output / 'templates.npz'
            templates = create_physical_templates(template_path, seed=seed + 1,
                count=config.get('template_count', 8), cadence_seconds=120,
                max_half_window_days=1, tail_cutoff_ppm=.001, progress=True, workers=workers)
        print(f'Templates: {len(templates)}', flush=True)
        nodes = config.get('injection', {}).get('nodes', 31)
        print(f'[3/5] Inject amplitude grid ({nodes}-point exposure integration)', flush=True)
        injected = make_injections(manifest, output / 'injections', amplitudes_ppm=amplitudes,
            shapes=config.get('shapes_per_target', 1), seed=seed + 2, preprocessing=settings,
            workers=workers, injection=config.get('injection'))
        from .assets import physical_identity
        template_ids = {physical_identity(p) for p in (templates.metadata or {}).get('systems', [])}
        if any(physical_identity(s.parameters.get('physical_parameters', {})) in template_ids for s in injected.samples):
            raise ValueError('Template/injection physical-system overlap')
    print('[4/5] Evaluate matched filter, CNN, and sequential gates on held-out examples', flush=True)
    if not config.get('reuse_injections'):
        injection_output = output / 'injections'
    score({'manifest': str(injection_output / 'manifest.json'), 'templates': str(template_path),
           'output': str(output / 'results'), 'checkpoint': str(checkpoint), 'classes': config.get('classes'),
           'cnn': {**config.get('cnn', {'device': 'cpu', 'split_sizes': [None, 7, 2], 'max_segments_per_item': 24}),
                   'workers': cnn_workers},
           'preprocessing': settings, 'normalize_scores': True,
           'evaluation': {'seed': seed, 'max_fpr': config.get('max_fpr', .05), 'bootstrap_repeats': config.get('bootstrap_repeats', 500),
                          'pipeline_order': config.get('pipeline_order', 'matched_filter_then_cnn'),
                          'backend': config.get('matched_filter_backend', 'cuda_fft'),
                          'workers': workers, 'progress': True, 'min_event_samples': 3, 'min_event_coverage': .5}})
    print('[5/5] Summarize amplitude sensitivity and render figures', flush=True)
    summary = summarize_results(output / 'results', amplitudes, seed=seed,
                                required_recall=config.get('required_recall', .9), repeats=config.get('bootstrap_repeats', 500))
    render(output / 'results')
    metrics = json.loads((output / 'results/metrics.json').read_text())
    for component, values in summary['by_component'].items():
        operating = metrics[component]['test']
        label = COMPONENT_LABELS[component]
        if component == "combined" and config.get("pipeline_order") == "cnn_then_matched_filter":
            label = "CNN → matched filter"
        print(f"{label}: recall={operating['recall']}; "
              f"benchmark precision={operating['precision']}; "
              f"amplitude at requested recall={values['minimum_tested_amplitude_ppm']} ppm "
              f"({values['measure']}; None means not established)", flush=True)
    write_json(output / 'complete.json', {'status': 'complete'})
    print(f'Complete: {output}\nOpen notebooks/evaluation.ipynb and set RUN to this directory.', flush=True)


def render(results):
    """Compare all three methods using saved scores, with no repeated inference."""
    import matplotlib.pyplot as plt
    results = Path(results)
    all_metrics = json.loads((results / 'metrics.json').read_text())
    summary = json.loads((results / 'amplitude_recovery.json').read_text())
    if 'by_component' not in summary:
        run = json.loads((results / 'run.json').read_text())
        summary = summarize_results(results, [r['amplitude_ppm'] for r in summary['rows']],
                                    seed=run['seed'], required_recall=summary['required_recall'])
    run = json.loads((results / 'run.json').read_text())
    order = run.get('pipeline_order', 'matched_filter_then_cnn')
    labels = {**COMPONENT_LABELS, 'combined': 'Matched filter → CNN' if order == 'matched_filter_then_cnn' else 'CNN → matched filter'}
    fig, grid = plt.subplots(3, 3, figsize=(14, 11), layout='constrained')
    for axes, component in zip(grid, labels):
        metrics = all_metrics[component]['test']
        sensitivity = summary['by_component'][component]
        curve = metrics['curve']
        axes[0].step(curve['recall'], curve['precision'], where='post')
        if metrics['recall'] is not None and metrics['precision'] is not None:
            axes[0].scatter([metrics['recall']], [metrics['precision']], label='Operating point')
            axes[0].legend()
        axes[0].set(title=labels[component], xlabel='Recall (target-sector)', ylabel='Benchmark precision',
                    xlim=(0, 1), ylim=(0, 1.02))
        counts = np.asarray(metrics['confusion_matrix'])
        totals = counts.sum(axis=1, keepdims=True)
        matrix = np.divide(counts, totals, out=np.full(counts.shape, np.nan), where=totals>0)
        axes[1].imshow(matrix, cmap='Blues', vmin=0, vmax=1)
        for (i, j), value in np.ndenumerate(matrix):
            text = 'undefined' if not np.isfinite(value) else f'{value:.1%}'
            axes[1].text(j, i, f'{text}\n(n={counts[i, j]})', ha='center', va='center',
                         color='white' if value >= .5 else 'black')
        axes[1].set(title=labels[component]+'\nWithin each truth class', xticks=[0, 1], yticks=[0, 1],
                    xticklabels=['Rejected', 'Accepted'], yticklabels=['Control', 'Injected'],
                    xlabel='Predicted decision', ylabel='Benchmark truth')
        rows = sensitivity['rows']
        axes[2].plot([r['amplitude_ppm'] for r in rows], [r['recall'] for r in rows], 'o-')
        axes[2].fill_between([r['amplitude_ppm'] for r in rows],
                             [np.nan if r['lower'] is None else r['lower'] for r in rows],
                             [np.nan if r['upper'] is None else r['upper'] for r in rows], alpha=.2)
        axes[2].axhline(sensitivity['required_recall'], ls='--', color='grey')
        axes[2].set(title=labels[component], xscale='log', xlabel='Injected amplitude (ppm)',
                    ylabel=sensitivity['measure'], ylim=(0, 1.02))
        axes[2].minorticks_off()
        axes[2].set_xticks([r['amplitude_ppm'] for r in rows], [f"{r['amplitude_ppm']:g}" for r in rows])
    for suffix in ['png', 'pdf']:
        fig.savefig(results / f'simple_evaluation.{suffix}', dpi=160)
    plt.close(fig)
