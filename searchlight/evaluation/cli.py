"""Configure and run simulation, SPOC, injection and cascade evaluations."""

import argparse
from pathlib import Path
from .. import config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Generate/evaluate analytic fixtures")
    demo.add_argument("output", type=Path)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--tuning-groups", type=int, default=40)
    demo.add_argument("--test-groups", type=int, default=80)
    run = commands.add_parser("run", help="Evaluate an existing manifest and library")
    run.add_argument("manifest", type=Path)
    run.add_argument("--template-version", type=config.validate_version)
    run.add_argument("output", type=Path)
    run.add_argument("--positive-class", default="lensing")
    run.add_argument("--cnn", action="store_true")
    run.add_argument("--model-version", type=config.validate_version)
    run.add_argument("--backend", choices=["cpu", "jax", "auto"], default="cpu")
    run.add_argument("--normalize-scores", action="store_true")
    run.add_argument("--workers", type=lambda value: "auto" if value == "auto" else int(value), default="auto")
    run.add_argument("--bootstrap-repeats", type=int, default=500)
    for command in (demo, run):
        command.add_argument("--max-fpr", type=float, default=0.05)
    for name in ("prepare", "simulate", "inject", "score"):
        command = commands.add_parser(name, help=f"{name.capitalize()} using a JSON configuration")
        command.add_argument("config", type=Path)
    report = commands.add_parser("report", help="Regenerate figures from saved results")
    report.add_argument("directory", type=Path)
    templates = commands.add_parser("templates", help="Independently sampled physical template library")
    templates.add_argument("output", type=Path)
    templates.add_argument("--seed", type=int, default=123)
    templates.add_argument("--count", type=int, default=16)
    templates.add_argument("--cadence-seconds", type=float, default=120)
    templates.add_argument("--proposal", choices=["evaluation", "rv_code", "rv_manuscript"], default="evaluation")
    templates.add_argument("--max-half-window-days", type=float)
    templates.add_argument("--tail-cutoff-ppm", type=float)
    templates.add_argument("--workers", type=lambda value: "auto" if value == "auto" else int(value), default="auto")
    args = parser.parse_args(argv)
    if args.command in {"prepare", "simulate", "inject", "score"}:
        from . import experiment
        result = getattr(experiment, args.command)(experiment.read_config(args.config))
        print(f"{args.command} completed")
        if args.command != "score":
            print(f"Manifest: {len(result.samples)} rows")
        return
    if args.command == "report":
        from .reporting import save_figures
        save_figures(args.directory)
        print(f"Figures regenerated: {args.directory}")
        return
    if args.command == "templates":
        from .simulations import create_physical_templates
        if args.output.exists():
            parser.error("Template output exists")
        create_physical_templates(args.output, seed=args.seed, count=args.count, cadence_seconds=args.cadence_seconds,
                                  proposal=args.proposal, max_half_window_days=args.max_half_window_days,
                                  tail_cutoff_ppm=args.tail_cutoff_ppm, workers=args.workers, progress=True)
        return
    from .runner import evaluate
    if args.command == "demo":
        from .simulations import create_demo, demo_templates
        if args.output.exists():
            parser.error("Demo output already exists; choose a new run directory")
        manifest = create_demo(args.output / "data", seed=args.seed,
                               tuning_groups=args.tuning_groups, test_groups=args.test_groups)
        metrics = evaluate(manifest, demo_templates(), args.output / "results", max_fpr=args.max_fpr,
                           provenance={"kind": "analytic engineering demonstration"})
    else:
        from .datasets import EvaluationManifest
        from ..matched_filtering import load_template_library
        cnn = None
        if args.cnn or args.model_version:
            from .cnn import EvaluationCNN
            from .parallel import worker_count
            cnn = EvaluationCNN(version=args.model_version, workers=min(worker_count(args.workers), 8))
        metrics = evaluate(EvaluationManifest.read(args.manifest), load_template_library(args.template_version),
            args.output, positive_class=args.positive_class, max_fpr=args.max_fpr, cnn=cnn,
            preprocessing={}, normalize_scores=args.normalize_scores, backend=args.backend,
            bootstrap_repeats=args.bootstrap_repeats, workers=args.workers, progress=True)
    for component, splits in metrics.items():
        if "test" in splits:
            test = splits["test"]
            print(f'{component}: test n={test["n"]}, ROC AUC={test["roc_auc"]}, '
                  f'AP={test["average_precision"]}, confusion={test["confusion_matrix"]}')


if __name__ == "__main__":
    main()
