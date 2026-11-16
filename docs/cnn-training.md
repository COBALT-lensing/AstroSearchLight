# CNN training

Select a dataset with `--dataset-version`. Its class folders are loaded from
`~/.astrosearchlight/datasets/simulens/<version>/`, containing `.fits` or
`.fits.gz` light curves. Labels are the sorted
folder names. Every class must contain at least two light curves; empty class
folders are rejected. The trainer can generate a dataset first with `--generate-data`; otherwise it reads existing FITS files.

```sh
uv sync --locked
uv run python bin/train_cnn.py --dataset-version 2.0 \
  --version 2.0 --seed 42 --max-epochs 100
```

Omit `--dataset-version` to use `dataset_version` from the configuration file
or `DEFAULT_DATASET_VERSION` in `searchlight/config.py`. Dataset and model
versions are independent. `--generate-data` requires an explicit dataset version
and refuses to replace an existing dataset.

Choose a new model version. The CLI requires `--version` and writes to
`~/.astrosearchlight/models/simulens/<version>/`. Existing artifacts are protected
unless `--overwrite` is supplied. See [artifact versions](artifact-versions.md)
for default selection and preserving prior versions.
To fine-tune an existing SimuLens on an updated dataset, start a new run with
`--version 2.0 --init-version 1.0`. The source version's `model_classes.json`
must exactly match the new dataset's class mapping. This loads the model weights
with a fresh optimizer, split, and training history; the initial checkpoint's
SHA-256 is recorded. It does not require the old training dataset or optimizer.

ImageNet ResNet34 weights are downloaded on first initialization unless already
cached. `--no-pretrained` starts from random weights and needs no such download.

The model jointly classifies the full light curve and consecutive 7-day and
2-day segments, rendered as 224 × 224 scatter plots. At most 24 segments per
curve are used. Configure these with `--split-sizes inf 7 2` and
`--max-segments 24`. Other options include `--device`, `--batch-size`, `--workers`,
`--lr`, the learning-rate scheduler settings, and `--no-amp`; see `--help`.

## Validation and stopping

A seeded, stratified split holds out approximately 20% of each class, with at
least one curve from each class in each partition. All segments of a curve stay
in the same partition. Related simulations or curves from the same source are
not automatically grouped: curate those relationships before training to avoid
leakage. This holdout controls model selection; it is not an independent final
test set.

Validation disables dropout, BatchNorm updates, and gradients. Losses are means
weighted by the number of curves, including the last partial batch. The model
with the lowest validation loss is retained independently of the live weights.
Training stops after `--patience` epochs without an improvement exceeding
`--min-delta`, or at `--max-epochs` (default 100). Defaults are patience 5 and
minimum improvement 0.

## Artifacts and resume

Each run writes:

- `model_weights.pth`: the best validation model, compatible with existing inference.
- `model_classes.json`: the matching ordered class names.
- `inference.json`: segment lengths and segment cap, restored automatically for classification.
- `training_checkpoint.pth`: latest completed epoch, optimizer, scheduler, mixed
  precision scaler, best weights, history, RNG states, and loader generator state.
- `training_run.json`: settings, split indices, per-file SHA-256 hashes and relative
  paths, metrics, code revision/working-tree status and source hashes, dependency versions and source
  identities, and environment information. Split indices refer to the manifest.

```sh
uv run python bin/train_cnn.py --dataset-version 2.0 \
  --version 2.0 --seed 42 --max-epochs 150 --resume
```

Resume restores the latest completed epoch, not the best inference model.
`--max-epochs` is the total target, including previously completed epochs.
Repeat the original training options: changes to data, classes, or training
settings are rejected. The dataset version and file contents must remain identical. Resume does not download ImageNet weights.
The prior run metadata is retained in the resumed run. Legacy inference-only
checkpoints cannot resume training because they contain no optimizer or RNG state;
use `--init-version` to start a new fine-tuning run from those weights instead.

Seeds and RNG state make runs repeatable within the same software and hardware
environment; GPU backends and different environments can still differ numerically.
CPU regression tests compare resumed and uninterrupted training exactly.
An interruption during an epoch restarts that epoch from the preceding checkpoint.
Individual artifacts are replaced atomically; they are not a multi-file transaction.

## Dataset generation and provenance

New simulated datasets come from [the shared photometry generator](generation.md).
The separate `lc-simulator` project is no longer needed. Install the simulator
with `uv sync --locked --extra template-generation`, then generate separately
or pass `--generate-data` to the training CLI. Three classes, noiseless/TESSGP
options, physical ranges, and selection settings are documented in that guide.

The trainer automatically reads and verifies a generated dataset's
`generation.json`, including its completion status and per-file SHA-256 hashes.
Its full generation provenance is embedded in the training run. Modified or
incomplete generated datasets are rejected before training artifacts are written.

For historical or externally generated data without this manifest, the trainer
still hashes the actual input files. It cannot infer the simulator revision,
distributions, or generation seed from those files. Supply that information with
`--dataset-provenance FILE`. Installed simulator versions are also recorded but
do not establish the provenance of existing data.

Local FITS plot caches now include resolved file paths, sizes, nanosecond timestamps,
and inode identities, so replacing or editing a FITS file invalidates its plots.
Remote MAST/HTTP sources retain their existing caching behavior.
