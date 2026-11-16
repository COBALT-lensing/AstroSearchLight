# AstroSearchLight

Vision-based light curve classification.

## Injection–recovery evaluation

Run the small real-SPOC evaluation with the actual CNN and matched filter:

```sh
uv sync --locked --extra template-generation --extra evaluation
python bin/evaluate_pipeline.py examples/evaluation/simple.json
```

For the larger parameter-range experiment using the fixed production template
bank, run `.venv/bin/python bin/evaluate_pipeline.py examples/evaluation/full.json`.
See the evaluation guide for sample counts and scope.

Open [notebooks/evaluation.ipynb](notebooks/evaluation.ipynb) for precision–recall,
a confusion matrix, and sensitivity versus injected amplitude for the CNN,
matched filter, and their sequential combination. See the
[evaluation guide](searchlight/evaluation/README.md) for inputs, settings and
interpretation, and [the reduced plan](EVALUATION_PLAN.md) for scope.
The supplied configuration reuses cached raw backgrounds and generates fresh
injections/templates with the corrected physics pinned in `uv.lock`.

## Matched-filter searches

The `searchlight.matched_filtering` module searches light curves with a bank of
sampled templates, returning candidate windows with matched-filter scores and
peak prominences.

Generate the default self-lensing template library with:

```shell
uv run python bin/generate_matched_filter_template_library.py --version 2.0
```

The generator writes `~/.astrosearchlight/templates/matched_filter/<version>/templates.npz`.
CNN datasets, models, and matched-filter banks use numeric artifact versions; see
[artifact versions and defaults](docs/artifact-versions.md). Install the self-lensing dependency with:

```shell
uv sync --extra template-generation --all-groups
```

The `default` preset draws 50,000 viable systems with the shared photometry
parameter ranges and distributions, then generates noiseless templates at
10-minute cadence. Configure `--template-count`, `--template-seed`, or
`--ranges-json`. Generation uses the shared batched JAX simulator.
See [shared generation](docs/generation.md) for ranges, noise, and CNN datasets.

### Completeness calibration

For recall-first searches, the template generator can empirically choose
matched-filter thresholds against a held-out validation set:

```shell
uv run python bin/generate_matched_filter_template_library.py \
  --version 2.0 \
  --calibrate-completeness \
  --validation-size 1000 \
  --target-recall 0.9
```

Calibration works by drawing random self-lensing-capable systems from the
shared photometry proposal, generating validation light curves centered
on their lensing events, then trying increasingly permissive threshold pairs
against a single generated template bank until the requested recall is reached.
A validation curve counts as matched when one of the returned detections lands within
`--peak-tolerance-samples` of the injected event center. The selected settings
and all attempted evaluations are stored in the output library metadata under
`calibration`. While progress output is enabled, the validation progress bar
updates its recall percentage after each curve is evaluated. The script also
prints which threshold step it is running. Candidate systems are filtered for
self-lensing and Roche-lobe constraints before their light curves are generated.
Validation curves are cached by default in
`~/.astrosearchlight/cache/matched_filter_validation_cache` and reused for later
runs with the same validation-generation settings. The random seed is recorded
in the cached metadata and included in the cache key. Validation evaluation
can run through the original CPU detector or a batched JAX backend. The default
`auto` mode uses JAX only when JAX reports a GPU device; otherwise it stays on
the CPU path.

Useful calibration options:

- `--calibrate-completeness`: enable the empirical validation loop. Without this,
  the script writes templates from the default sampler.
- `--validation-size N`: number of held-out light curves to generate. The default
  is `1000`; smaller values are useful for smoke tests.
- `--validation-seed N`: random seed for the held-out systems. If omitted, the
  script generates a non-deterministic seed and records it in the output
  metadata. A changed seed creates an independent validation set and cache entry.
- `--no-validation-cache`: disable reading and writing cached validation light
  curves.
- `--validation-cache-dir PATH`: directory for cached validation light curves.
  The default is `~/.astrosearchlight/cache/matched_filter_validation_cache`.
- `--validation-backend auto|cpu|jax`: backend for validation evaluation.
  `auto` uses the JAX backend only when a GPU is visible to JAX. The JAX backend
  is optimized for calibration recall and uses a JAX-native local-peak test
  rather than returning full `MatchedFilterMatch` objects.
- `--validation-batch-size N`: number of validation curves per JAX batch. Larger
  batches can improve GPU utilization but require more device memory.
- `--validation-template-batch-size N`: number of templates per JAX batch. Lower
  this first if JAX reports GPU out-of-memory during validation evaluation.
- `--generation-batch-size N`: number of self-lensing systems per batched JAX
  light-curve generation call. Lower this if JAX reports GPU out-of-memory while
  generating the template library.
- `--target-recall X`: stop once validation recall reaches this value. The
  default is `0.9`.
- `--peak-tolerance-samples N`: maximum sample offset between a detected peak and
  the injected event center for the validation curve to count as recovered.
- `--max-templates N`: cap the number of generated templates. This is mainly for
  quick calibration smoke tests; `--template-count` sets the requested bank size.
- `--ranges-json FILE`: override physical parameter bounds for templates and validation.
- `--fixed-distance`: fix the default proposal at 1 kpc instead of sampling distance.
- `--cpu-workers N`: worker processes for CPU validation evaluation.
- `--workers N`: compatibility alias for `--cpu-workers` when `--cpu-workers` is
  not set.

For GPU-backed calibration, use the JAX validation backend with CPU workers for
CPU fallback evaluation:

```shell
uv run python bin/generate_matched_filter_template_library.py \
  --version 2.0 \
  --calibrate-completeness \
  --validation-backend jax \
  --cpu-workers 12
```

The matched-filter detector defaults are intentionally permissive for
completeness: `prominence_sigma=3.0` and `min_score_sigma=4.0`. When a loaded
template library contains calibrated thresholds, `detect_matched_filter_peaks`
uses those calibrated values unless the call explicitly overrides them.
Individual calls can still pass stricter thresholds, or set
`min_score_sigma=None` to disable the robust score threshold entirely.

For many light curves on a shared time grid, pass a two-dimensional flux array
with shape `(n_light_curves, n_times)` to `detect_matched_filter_peaks`, or call
`detect_matched_filter_peaks_batched` directly. The detector builds a
best-template envelope for each light curve, keeping the highest score and
winning template at each cadence, then runs peak finding once on that envelope.
Use `batch_size=N` to cap the number of templates in each envelope correlation
batch; the default is 512 templates per batch.
Both public detector entry points accept `backend="auto"|"cpu"|"jax"` for the
matched-filter correlations. The default `auto` uses JAX when JAX can see a GPU
and otherwise keeps the SciPy CPU path; pass `backend="jax"` to force JAX.
For template libraries generated by this project, detections use each template's
stored `tau_eff` and library `window_scale` to report the compact event region;
otherwise the reported match span covers the full matched template window.

The project depends on `jax[cuda12]` so a CUDA-capable JAX backend can be used
when the machine has compatible NVIDIA drivers and the GPU is visible to the
process. You can check what JAX sees with:

```shell
uv run python -c "import jax; print(jax.devices())"
```

The generator disables JAX's default GPU memory preallocation with
`XLA_PYTHON_CLIENT_PREALLOCATE=false` unless that environment variable is
already set.

## Models

### SimuLens

ResNet trained to jointly classify segmented light curves. Light curves are split into segments of 2 days, 7 days, and infinite length (i.e. full light curve). Segments are plotted as scatter plots and the CNN is trained to classify the plots.

Trained on simulated TESS data:

- (Self-)Lensing: 10,297
- Stellar flares: 10,043
- Not-lensing (including transits): 9,012

Generate new CNN data with the shared photometry simulator (TESSGP noise by default):

```sh
uv run python bin/generate_training_data.py --version 2.0 --count-per-class 10000
```

Use `--noiseless` for deterministic curves. The three existing classes are retained.
See [generation settings](docs/generation.md) for physical ranges and sampling.

Datasets are stored under `~/.astrosearchlight/datasets/simulens/<version>/`.
To train on a dataset version, select it and a new model version:

```sh
uv run python bin/train_cnn.py --dataset-version 2.0 --version 2.0
```

See [CNN training](docs/cnn-training.md) for validation, reproducibility,
checkpoints/resume, and dataset provenance requirements.

### HandLens (planned)

To be trained on manual (Zooniverse) labels of real light curves.

### FalseLens (planned)

To be trained on manual (Zooniverse) vetting/false positive removal with CCD/pixel data.

### Deprecated

- XGBoost
- Upsilon
