"""Train SimuLens into an explicitly selected run directory."""

import argparse
import json
from pathlib import Path

from searchlight.models.simulens import SimuLens
from searchlight import generation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Directory containing one FITS folder per class")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for this run's model and checkpoints")
    parser.add_argument("--resume", action="store_true", help="Continue the output directory's latest training checkpoint")
    parser.add_argument("--init-checkpoint", type=Path, help="Start a new run from existing inference weights and their model_classes.json")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing run instead of resuming it")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=100, help="Total epoch limit, including epochs already completed")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=SimuLens.LOADER_WORKERS)
    parser.add_argument("--device", help="Torch device, e.g. cpu or cuda:0; automatically selected by default")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--scheduler-step-size", type=int, default=5)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--split-sizes", type=float, nargs="+", default=[float("inf"), 7, 2], help="Segment lengths in days; use inf for the full curve")
    parser.add_argument("--max-segments", type=int, default=24)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true", help="Initialize randomly instead of using ImageNet weights")
    parser.add_argument("--dataset-provenance", type=Path, help="JSON describing dataset generation, ranges, seeds, and simulator revisions")
    parser.add_argument("--generate-data", action="store_true",
                        help="Create the dataset with the shared photometry generator before training")
    generation.add_generation_arguments(parser)
    args = parser.parse_args(argv)
    if args.generate_data and args.resume:
        parser.error("--generate-data cannot be combined with --resume")
    if args.resume and args.init_checkpoint:
        parser.error("--resume and --init-checkpoint cannot be combined")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite cannot be combined")
    if args.generate_data:
        generation.generate_from_args(args.dataset, args)
    provenance = json.loads(args.dataset_provenance.read_text()) if args.dataset_provenance else None
    trainer = SimuLens(output_dir=args.output_dir, device=args.device, lr=args.lr,
        scheduler_step_size=args.scheduler_step_size, scheduler_gamma=args.scheduler_gamma,
        split_sizes=args.split_sizes, max_segments_per_item=args.max_segments,
        loader_batch_size=args.batch_size, loader_workers=args.workers,
        use_amp=not args.no_amp, pretrained=not (args.no_pretrained or args.resume or args.init_checkpoint))
    run = trainer.train(args.dataset, seed=args.seed, max_epochs=args.max_epochs,
        patience=args.patience, min_delta=args.min_delta, overwrite=args.overwrite,
        resume=args.output_dir.expanduser() / "training_checkpoint.pth" if args.resume else None,
        provenance=provenance, init_checkpoint=args.init_checkpoint)
    print(f"Best epoch: {run['best_epoch']}; model: {trainer.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
