#!/usr/bin/env python3
"""Run the real-data injection evaluation from one JSON configuration."""
import argparse
from pathlib import Path
import sys
import os
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path, help='JSON configuration; paths relative to this file')
    parser.add_argument('--workers', type=lambda value: 'auto' if value == 'auto' else int(value),
                        help='CPU process count (default: configuration or auto, up to 32)')
    parser.add_argument('--backend', choices=['cpu', 'jax', 'cuda_fft'], help='Matched-filter backend (default: configuration or cuda_fft)' )
    parser.add_argument('--resume-from', type=Path, help='Reuse completed injections and original settings from a saved run; score into a fresh directory')
    parser.add_argument('--output', type=Path, help='Output directory (relative to the current directory); existing names get a numbered suffix')
    parser.add_argument('--cnn-workers', type=int, help='CPU CNN processes (default: up to 8)')
    args = parser.parse_args()
    from searchlight.evaluation.experiment import read_config
    from searchlight.evaluation.simple import run
    config = read_config(args.config)
    config.setdefault("matched_filter_backend", "cuda_fft")
    if args.resume_from is not None:
        import json
        previous = args.resume_from.expanduser().resolve()
        runtime_backend = config.get("matched_filter_backend")
        config = json.loads((previous / 'config.json').read_text())
        if runtime_backend is not None:
            config["matched_filter_backend"] = runtime_backend
        manifest = Path(config.get('reuse_injections') or previous / 'injections/manifest.json')
        if not manifest.is_file():
            parser.error(f'Completed injection manifest missing: {manifest}')
        config['output'] = config.get('requested_output', str(previous))
        config['reuse_injections'] = str(manifest)
        config['templates'] = config.get('templates') or str(previous / 'templates.npz')
    if args.backend is not None:
        config['matched_filter_backend'] = args.backend
    if args.output is not None:
        config["output"] = str(args.output.expanduser().resolve())
    if args.workers is not None:
        config["workers"] = args.workers
    if args.cnn_workers is not None:
        config["cnn_workers"] = args.cnn_workers
    # Nested local source paths follow the same rule as top-level inputs.
    if 'backgrounds' in config:
        for record in config['backgrounds'].get('sources', []):
            if not record['source'].startswith('mast:'):
                record['source'] = str((args.config.resolve().parent / Path(record['source']).expanduser()).resolve())
    run(config)


if __name__ == '__main__':
    main()
