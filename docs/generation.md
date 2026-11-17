# Shared physical light-curve generation

`searchlight.generation` replaces `lc-simulator` for new SearchLight datasets.
It uses the `selflensing.batch.batched_light_curve_jit` forward model
and `batched_gp_light_curve_jit` for optional TESSGP noise. The same physical
sampling and batch generation serve CNN datasets and matched-filter templates.

Install the simulator and its packaged TESS response/noise data:

```sh
uv sync --locked --extra template-generation
```

Generation uses the available JAX backend, with CPU fallback.
`JAX_PLATFORMS=cpu` explicitly selects CPU.
Module-level JIT functions and one cached GP-input factory are reused across
batches. The default generation batch size is 64. Reduce it if device memory
is limited.

## Default proposal

The default preset samples the following physical parameters. These bounds
are defined locally so upstream limit changes cannot silently alter generated
datasets.

| Parameter | Bounds | Sampling before rejection |
|---|---|---|
| Semi-major axis, AU | 0.1–200 | Log-uniform |
| Inclination, radians | 0–π/2 | Uniform angle |
| Lens mass, solar masses | 0.1–50 | Log-uniform |
| Companion mass, solar masses | 0.5–50 | Uniform |
| Eccentricity | 0–0.999999 | Uniform in log(1 + e) |
| Periapsis phase, radians | 0–2π | Uniform |
| Distance, kpc | 0.1–100 | Uniform |

These are simulation proposals, not a population prior. Systems are rejected
if they cannot self-lens, undergo Roche-lobe overflow, or have invalid derived
quantities. Geometry and crossing times use the installed simulator's eccentric
observer-frame expressions. Stellar radius and temperature come from the
simulator's physical relations.

`--ranges-json FILE` overrides any bounds. The JSON is a mapping such as
`{"a": [1, 100], "i": [1.5706, 1.5707963267948966]}`. Equal endpoints fix a
parameter. Units are those in the table. Default limits and sampling distributions
are available as `DEFAULT_RANGES` and `SAMPLING_DISTRIBUTIONS` in the module.

## CNN datasets

```sh
uv run python bin/generate_training_data.py --version 2.0 \
  --count-per-class 10000 --generation-seed 20260825
```

A new dataset version is written to
`~/.astrosearchlight/datasets/simulens/2.0/`. Its directory must not already exist;
existing versions cannot be overwritten. This writes the three existing classes:

- `lensing`: self-lensing + ellipsoidal variations + Doppler boosting, with TESSGP noise.
- `flare`: self-lensing disabled, with an exponential flare added to the physical
  mean before GP sampling. Flare amplitudes are uniform from 0 to 9 above baseline;
  decay times are log-uniform from 10² to 10⁶⋅⁵ seconds.
- `not-lensing`: self-lensing-disabled controls and, by default, 30% inverted
  physical means (`2 - mean_flux`) as transit/eclipsing-like confounders.
  `--inverted-fraction` changes this fraction.

Examples are independent draws; positive/control pairs are not shared across
classes. The default is 10,000 **retained** examples per class. The Python API
also accepts explicit `counts={"lensing": ..., "flare": ..., "not-lensing": ...}`.
Each class needs at least two examples for training's stratified holdout.

The default window is 27 days at exactly 600-second cadence
(3,888 samples), with event centres uniform across the sector. The positive
selection cut requires the sampled self-lensing amplitude to exceed five times
the expected fractional white-noise uncertainty. This cut does not include the
correlated GP variance. It changes the retained parameter distribution relative
to the raw proposal. `--detection-sigma 0` disables it.

The noisy path reproduces TESSGP's `GS` conventions: independent granulation
and seismic amplitudes and seismic quality factors from upstream prior helpers,
count-space mean and covariance scales, and instrument white noise. GP sampling
uses days converted to seconds by the batched API, then returns normalized flux.
A cached `FluxModel` avoids per-system response/noise file reloads.

Use `--noiseless` to save deterministic means instead. Their `flux_err` is zero
and the default detectability cut is disabled. A positive detectability threshold
is rejected in noiseless mode rather than silently using zero uncertainty.
`--cadence-seconds`, `--duration-days`, `--generation-batch-size`, and
`--max-proposal-draws` configure cadence, duration, memory use, and the total
rejection budget. Changes to cadence affect count baselines using the TESSGP
conventions; the upstream instrument noise-versus-magnitude table is retained.

Each FITS table has Astropy-native time, normalized flux, flux uncertainty,
physical truth, class/subtype, event time, and applicable GP/flare parameters.
`generation.json` records ranges, distributions, seed, batch size, selection,
class counts, generator source hash, simulator revisions, and per-file hashes.
An interrupted dataset remains marked incomplete; choose a new dataset version
to retry while retaining the partial output. Generation is not resumed in place.

The CNN trainer automatically verifies and records `generation.json`:

```sh
uv run python bin/train_cnn.py --dataset-version 2.0 --version 2.0
```

Or generate and train in one command using the same generation options:

```sh
uv run python bin/train_cnn.py --dataset-version 2.0 --version 2.0 \
  --generate-data --count-per-class 10000 --generation-seed 20260825
```

Omit `--dataset-version` when training to use the configured default dataset.
The Python API accepts versions as well: `generate_cnn_dataset("2.0")` and
`SimuLens(version="2.0").train(dataset_version="2.0")`.
See [CNN training](cnn-training.md) for checkpoint resume and fine-tuning options.
Seeds are repeatable for the same settings, batching, and software/hardware;
changing batch size changes the stream of parameter and GP draws.

## Matched-filter templates

```sh
uv run python bin/generate_matched_filter_template_library.py \
  --template-seed 20260826 --version 2.0
```

The bank is written to `~/.astrosearchlight/templates/matched_filter/2.0/templates.npz`.
Existing versions are protected unless `--overwrite` is supplied. See
[artifact versions](artifact-versions.md) for selection and default configuration.

The `default` preset draws 50,000 viable systems using the same shared proposal.
Change `--template-count` for an individual run; the shared default is
`DEFAULT_TEMPLATE_COUNT` in `searchlight/matched_filtering.py`.
Templates are **noiseless**, with all three physical components enabled, events
centred at zero, and 144-minute cadence. The cadence default is
`DEFAULT_TEMPLATE_CADENCE_MINUTES` in `searchlight/matched_filtering.py`.
It was calculated from library 2.0 as half the 10th-percentile effective crossing
time (about 144.113 minutes), rounded down to 144 minutes. At that cadence,
90.014% of that bank spans at least two cadence intervals per crossing; sampling
phase and pulse shape still affect the useful measurements. Override it with
`--cadence-minutes`. Template generation excludes systems whose `tau_eff` is
shorter than twice the selected cadence, drawing replacements to reach the
requested system count. At the default cadence the minimum is 288 minutes
(0.2 days). The cut is stored as `min_crossing_time_days` in the library metadata.
CNN photometry retains its 600-second default cadence and its existing selection.
The common time grid always includes zero;
its half-width defaults to 1.2 times the largest crossing time. The CNN's
noise/detectability selection is not applied to this shape bank. Use
`--fixed-distance` to fix 1 kpc; otherwise distance is sampled over the default range.
Distance affects noisy detectability, but not the normalized noiseless forward
model. The template seed differs from the CNN generation seed by default.

Use `--ranges-json` to override physical bounds. The `default` preset is the
only supported preset for template generation and calibration.

Completeness calibration evaluates threshold settings against a single generated
bank. Validation systems are independently sampled with the shared distributions
and passed through the same noiseless model, with their own seed. The template
crossing-time cut is not applied to validation systems, so calibration still
measures recovery of short events. Custom ranges propagate to both the
bank and validation set. Validation caches include the seed, effective proposal,
generator source hash, and simulator provenance, so old distributions or changed
seeds cannot silently reuse a prior validation set. This calibration measures
shape recovery, not survey completeness on noisy backgrounds.
