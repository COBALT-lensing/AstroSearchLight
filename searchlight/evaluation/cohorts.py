"""Outcome-blind, sector/cadence-stratified selection of public SPOC products."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .ingestion import cache_product, prepare_sources
from .preprocessing import read_lightcurve


def mast_catalog(sectors, cadences):
    from astroquery.mast import Observations
    records = []
    for sector in sectors:
        for cadence in cadences:
            print(f'Querying SPOC sector {sector}, cadence {cadence:g} seconds...', flush=True)
            table = Observations.query_criteria(obs_collection='TESS', provenance_name='SPOC',
                sequence_number=sector, dataproduct_type='timeseries', t_exptime=[cadence-1, cadence+1])
            for row in table:
                tic, uri = str(row['target_name']).strip(), str(row['dataURL'])
                if tic.isdecimal() and uri.endswith('_lc.fits') and str(row['dataRights']).upper() == 'PUBLIC':
                    records.append({'tic': str(int(tic)), 'sector': int(sector),
                                    'cadence_seconds': float(cadence), 'source': uri})
    return records


def select_products(records, *, per_stratum, seed):
    if per_stratum < 1:
        raise ValueError('At least one target per sector/cadence stratum is required')
    # One exact product per TIC/sector/cadence, independent of table row order.
    unique = {}
    for row in records:
        key = (str(int(row['tic'])), int(row['sector']), float(row['cadence_seconds']))
        if key not in unique or row['source'] > unique[key]['source']:
            unique[key] = dict(row)
    groups = {}
    for row in unique.values():
        groups.setdefault((row['sector'], row['cadence_seconds']), []).append(row)
    selected = []
    for stratum, rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda row: hashlib.sha256(
            f"{seed}:{stratum}:{row['tic']}:{row['source']}".encode()).hexdigest())
        selected.extend(ordered[:per_stratum])
    return selected


def load_catalog(config):
    """Reuse a saved catalogue or fetch/cache one without historical run files."""
    path = Path(config['catalog']) if config.get('catalog') else None
    if path is not None and path.is_file():
        print(f'Loading SPOC catalogue: {path}', flush=True)
        return json.loads(path.read_text()), 'saved_catalogue'
    if not config.get('sectors'):
        raise ValueError('A missing catalogue requires sectors to query MAST')
    print('No cached SPOC catalogue; querying MAST for configured sectors/cadences...', flush=True)
    records = mast_catalog(config['sectors'], config.get('cadences_seconds', [120.]))
    if path is not None and records:
        import tempfile
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(records, stream)
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'Cached SPOC catalogue: {path}', flush=True)
    return records, 'mast_query'


def excluded_targets(config):
    """Explicit exclusions plus any locally available earlier evaluation targets."""
    from .datasets import EvaluationManifest
    paths = []
    if config.get('exclude_manifest'):
        paths.append(Path(config['exclude_manifest']))
    if config.get('exclude_prior_runs'):
        root = Path(config['exclude_prior_runs'])
        for pattern in ('*/results/manifest.json', '*/injections/manifest.json', '*/spoc-cohort/prepared/manifest.json'):
            paths.extend(root.glob(pattern))
    excluded = {str(int(str(tic).upper().removeprefix('TIC').strip())) for tic in config.get('exclude_tics', [])}
    for path in set(paths):
        for sample in EvaluationManifest.read(path).samples:
            if sample.tic is not None:
                excluded.add(str(int(str(sample.tic).upper().removeprefix('TIC').strip())))
    return excluded


def prepare_cohort(config):
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=False)
    records, catalogue_source = load_catalog(config)
    if not records:
        raise ValueError('No eligible public SPOC products')
    (output/'catalog.json').write_text(json.dumps(records, indent=2)+'\n')
    if config.get('sectors'):
        records = [r for r in records if int(r['sector']) in config['sectors']]
    if config.get('cadences_seconds'):
        records = [r for r in records if any(np.isclose(r['cadence_seconds'], c) for c in config['cadences_seconds'])]
    excluded = excluded_targets(config)
    print(f'Excluding {len(excluded)} target(s) from explicit or available prior-run records.', flush=True)
    records = [r for r in records if str(int(r['tic'])) not in excluded]
    selected = select_products(records, per_stratum=config.get('targets_per_stratum', 4), seed=config.get('seed', 934))
    from tqdm.auto import tqdm
    sources = []
    for row in tqdm(selected, desc='Prepare background cohort'):
        path = cache_product(row['source'], output/'products') if row['source'].startswith('mast:') else Path(row['source'])
        sources.append({**row, 'source': str(path.resolve()), 'label': None,
            'sample_id': f"tic{row['tic']}-sector{row['sector']}-cadence{row['cadence_seconds']:g}",
            'parameters': {'product_source': row['source'], 'selection': 'seeded sector/cadence-stratified sampling; no score/label selection'}})
    prepared = prepare_sources({'sources': sources, 'preprocessing': config.get('preprocessing', {}),
        'splits': {'seed': config.get('split_seed', 935), 'tuning_fraction': config.get('tuning_fraction', .3)}}, output/'prepared')
    rows = []
    for sample in prepared.samples:
        curve = read_lightcurve(sample.source)
        valid = np.isfinite(curve.flux) & np.isfinite(curve.flux_err) & (curve.quality == 0)
        noise = float(np.median(curve.flux_err[valid])/np.median(curve.flux[valid])*1e6) if valid.any() else None
        rows.append({'sample_id': sample.sample_id, 'tic': sample.tic, 'sector': sample.sector,
            'split': sample.split, 'cadence_seconds': sample.cadence_seconds,
            'tess_magnitude': curve.metadata.get('TESSMAG'), 'crowding': curve.metadata.get('CROWDSAP'),
            'noise_ppm': noise, 'retained_quality_fraction': float(valid.mean()), 'label': None})
    pd.DataFrame(rows).to_csv(output/'cohort_table.csv', index=False)
    summary = {'created_utc': datetime.now(timezone.utc).isoformat(), 'config': config,
        'catalogue_source': catalogue_source, 'excluded_target_count': len(excluded),
        'catalogue_products': len(records), 'selected_products': len(rows),
        'independent_targets': len({r['tic'] for r in rows}),
        'groups_by_split': {split: len({r['tic'] for r in rows if r['split'] == split}) for split in ['tuning','test']},
        'products_by_sector': dict(Counter(str(r['sector']) for r in rows)),
        'selection': 'uniform hash ranking within sector/cadence strata, followed by all-target grouping',
        'sampling_frame': 'public SPOC-targeted products in the configured sectors/cadences; not all TESS stars',
        'labels': 'all originals unknown; suitable for injection recovery/candidate rates, not real-data ROC negatives',
        'test_scores_accessed': False}
    (output/'cohort_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return prepared
