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
systems that are not self-lensing-capable or are in Roche-lobe overflow. It uses
up to four worker processes by default; pass `--workers N` to tune this. Use
`--grid-preset completeness` for a denser hand-curated grid designed to cover a
wider range of event morphologies, or `--grid-preset limits` to sample the
upstream `py-self-lensing` parameter limits instead.

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
metadata but is not part of the cache key.

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
- `--workers N`: number of worker processes used while generating template
  curves.

The matched-filter detector defaults are intentionally permissive for
completeness: `prominence_sigma=3.0` and `min_score_sigma=4.0`. Individual calls
to `detect_matched_filter_peaks` can still pass stricter thresholds, or set
`min_score_sigma=None` to disable the robust score threshold entirely.

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
