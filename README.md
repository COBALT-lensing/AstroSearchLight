# AstroSearchLight

Vision-based light curve classification.

## Matched-filter searches

The `searchlight.matched_filtering` module searches light curves with a bank of
sampled templates, returning candidate windows with matched-filter scores and
peak prominences.

Generate the default self-lensing template library with:

```shell
uv run python bin/generate_matched_filter_template_library.py
```

The generator writes `data/matched_filter_templates.npz` by default. Install its
private self-lensing dependency with:

```shell
uv sync --extra template-generation --all-groups
```

By default, the generator uses a broad, rounded `coarse` grid over semi-major
axis, component masses, eccentricity, and periapsis phase, then filters out
systems that are not self-lensing-capable or are in Roche-lobe overflow using a
batched JAX filter. Use `--grid-preset completeness` for a denser hand-curated
grid designed to cover a wider range of event morphologies, or `--grid-preset
limits` to sample the upstream `py-self-lensing` parameter limits instead.
Generated curves use `py-self-lensing`'s batched JAX light-curve interface.

### Completeness calibration

For recall-first searches, the template generator can empirically choose a
template grid and matched-filter thresholds against a held-out validation set:

```shell
uv run python bin/generate_matched_filter_template_library.py \
  --calibrate-completeness \
  --validation-size 1000 \
  --target-recall 0.9 \
  --calibration-grid-presets coarse,completeness
```

Calibration works by drawing random self-lensing-capable systems from the
`py-self-lensing` parameter limits, generating validation light curves centered
on their lensing events, then trying template-grid presets and increasingly
permissive threshold pairs until the requested recall is reached. A validation
curve counts as matched when one of the returned detections lands within
`--peak-tolerance-samples` of the injected event center. The selected settings
and all attempted evaluations are stored in the output library metadata under
`calibration`. While progress output is enabled, the validation progress bar
updates its recall percentage after each curve is evaluated. The script also
prints which calibration grid and threshold step it is running. Before template
generation starts for a grid, it filters candidate systems for self-lensing and
Roche-lobe constraints; for dense grids this can be a noticeable stage, so it
has its own progress bar. Validation curves are cached by default in
`.cache/matched_filter_validation_cache` and reused for later runs with the same
validation-generation settings. The random seed is recorded in the cached
metadata but is not part of the cache key. Validation evaluation can run through
the original CPU detector or a batched JAX backend. The default `auto` mode uses
JAX only when JAX reports a GPU device; otherwise it stays on the CPU path.

Useful calibration options:

- `--calibrate-completeness`: enable the empirical validation loop. Without this,
  the script just writes the requested grid preset.
- `--validation-size N`: number of held-out light curves to generate. The default
  is `1000`; smaller values are useful for smoke tests.
- `--validation-seed N`: random seed for the held-out systems. If omitted, the
  script generates a non-deterministic seed and records it in the output
  metadata. Validation caching ignores the seed when looking up an existing
  cache entry, so use `--no-validation-cache` when you need a fresh validation
  set for the same generation settings.
- `--no-validation-cache`: disable reading and writing cached validation light
  curves.
- `--validation-cache-dir PATH`: directory for cached validation light curves.
  The default is `.cache/matched_filter_validation_cache`.
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
- `--calibration-grid-presets A,B`: comma-separated grid presets to try in order.
  The default is `coarse,completeness`; use
  `--calibration-grid-presets completeness` to skip the coarse step.
- `--peak-tolerance-samples N`: maximum sample offset between a detected peak and
  the injected event center for the validation curve to count as recovered.
- `--max-templates N`: cap the number of generated templates. This is mainly for
  quick calibration smoke tests; leaving it unset gives the calibration loop the
  full grid.
- `--samples-per-parameter N`: granularity for `--grid-preset limits`; fixed
  presets such as `coarse` and `completeness` use their curated values unless
  explicit `--a-values`, `--e-values`, etc. are supplied.
- `--include-distance` and `--distance-samples N`: include distance as a grid
  dimension. Distance mostly changes amplitude, so it is fixed at `1 kpc` by
  default.
- `--cpu-workers N`: worker processes for CPU validation evaluation.
- `--workers N`: compatibility alias for `--cpu-workers` when `--cpu-workers` is
  not set.

For GPU-backed calibration, use the JAX validation backend with CPU workers for
CPU fallback evaluation:

```shell
uv run python bin/generate_matched_filter_template_library.py \
  --calibrate-completeness \
  --calibration-grid-presets completeness \
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
`detect_matched_filter_peaks_batched` directly. The batched path resamples and
normalizes templates once, then correlates each template against a batch of
light curves. Use `batch_size=N` to cap peak memory for very large searches.
For template libraries generated by this project, detections use each template's
stored `tau_eff` and library `window_scale` to search and report the compact
event region rather than the full shared template grid. The reported match span
is then tightened to the matched-filter peak width; adjust
`match_window_rel_height` if you want a narrower or wider highlighted peak.

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

### HandLens (planned)

To be trained on manual (Zooniverse) labels of real light curves.

### FalseLens (planned)

To be trained on manual (Zooniverse) vetting/false positive removal with CCD/pixel data.

### Deprecated

- XGBoost
- Upsilon
