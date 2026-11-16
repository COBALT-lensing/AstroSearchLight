# Evaluation guide and remaining work

The evaluation measures how well the matched filter, CNN, and their combined
pipeline detect simulated self-lensing signals injected into real TESS SPOC
light curves. It is intended to support a methods/results section and to help
compare detector performance. The evaluation is implemented; the remaining work
is to run the desired experiment, inspect the results, and report them.

## What the evaluation does

Each original light curve is kept as an uninjected control. Independently drawn
self-lensing profiles are added at several amplitudes, so each background has
multiple injected counterparts. The physical model comes from the installed
`selflensing` dependency. Exposure averaging is included, and the original
observational uncertainties are retained.

Targets are divided into tuning and test sets. All observations and injections
from the same target remain together. Thresholds are chosen on tuning targets to
maximize recall subject to the configured control acceptance rate (5% by
default). Final performance is measured on test targets. Confidence intervals
resample whole targets to account for the dependence between repeated injections.

The same data are evaluated with the matched filter alone, the CNN alone, and
the combined pipeline. Combined acceptance requires both detectors to accept.
The component scores are reused for this comparison; the evaluation measures
detection performance rather than pipeline runtime. It does not train the CNN.

## Running an evaluation

From the repository root, install the locked environment:

```sh
uv sync --locked --extra evaluation --extra template-generation
```

The main configuration is [examples/evaluation/full.json](examples/evaluation/full.json).
It uses the trained CNN checkpoint and matched-filter template bank under
`~/.astrosearchlight/`. Check that these assets are available before starting.
The configuration selects public SPOC products from MAST; internet access is
needed for catalogue queries and light curves that are not already cached.

```sh
.venv/bin/python bin/evaluate_pipeline.py examples/evaluation/full.json
```

The full preset uses CUDA matched filtering and CPU CNN inference. For a machine
without CUDA, add `--backend cpu`; filtering with the full production template
bank can be very slow on CPU. `--workers` and `--cnn-workers` control CPU process
counts if memory or available processors require smaller limits.

The runner prints progress and the output directory. Existing runs are preserved:
if the requested name exists, a numbered directory is created. A top-level
`complete.json` marks a completed run.

### Choosing the experiment

Copy the configuration before making a different experiment. Paths inside it are
relative to the configuration file; `~` expands to your home directory. The most
useful settings are:

| Setting | Purpose |
|---|---|
| `output` | Base name of the saved run directory |
| `backgrounds` | Catalogue selection, sectors, cadence, target count, and tuning split |
| `manifest` | Alternative to `backgrounds`: reuse prepared observations and their splits |
| `checkpoint`, `templates` | CNN and matched-filter assets to evaluate |
| `shapes_per_target` | Number of independent physical profiles per background |
| `amplitudes_ppm` | Signal strengths to test |
| `injection.ranges` | Physical parameter ranges for the profile proposal |
| `max_fpr` | Maximum empirical control acceptance rate used in threshold tuning |
| `required_recall` | Recall required for the reported amplitude limit |
| `bootstrap_repeats` | Number of target-level uncertainty resamples |
| `seed` | Reproducible random draws for the experiment |

For a small development run, reduce `backgrounds.targets_per_stratum`,
`shapes_per_target`, and the amplitude list. Retain enough independent targets
in both splits to tune thresholds and estimate uncertainty. Small runs are useful
execution checks but provide limited statistical precision.

The alternative [simple.json](examples/evaluation/simple.json) is a pilot preset
using a locally prepared background manifest, a small generated template bank,
and isolated, well-covered injections. It requires that manifest to exist. It
is a different experimental setup from the full preset, which evaluates the
saved production template bank.

### Full preset: data and signals

The full preset requests 25 products in each of sectors 1, 13, 27, and 39 at
120-second cadence. It excludes targets found in available prior-run manifests.
The achieved cohort and tuning/test counts are saved; missing historical records
cannot establish exclusions from those datasets.

Eight shapes per background and eight amplitudes (30–100,000 ppm) give up to
6,400 injected curves and 100 controls. The physical proposal follows the RV
manuscript ranges, with both masses sampled logarithmically:

| Parameter | Range | Proposed distribution |
|---|---|---|
| Semi-major axis | 0.1–200 AU | Log-uniform |
| Lens and source masses | 0.5–50 solar masses each | Log-uniform |
| Inclination | 89.99–90 degrees | Uniform in angle |
| Eccentricity | 0–0.999999 | Uniform in log(1 + e) |
| Periapsis phase | 0–360 degrees | Uniform |
| Distance | 0.1–100 kpc | Uniform |

Configuration angles are in radians. Draws must satisfy the physical lensing
and Roche-lobe conditions, so accepted distributions can differ from the
proposal. Inspect the saved parameter plots rather than assuming uniform
coverage of the allowed ranges.

Each profile is rescaled to the requested exposure-averaged reference peak
amplitude. This holds shape fixed while testing signal strength. It is a
conditional morphology/noise experiment, not a simulated physical population.
Distance does not change the noise already present in a real background.

The full preset retains periodic, short, long, partial, and gap-affected events.
An event in a gap may have a much smaller sampled signal than its reference
amplitude, or no sampled signal. Coverage diagnostics distinguish these cases.
The pilot instead selects isolated, adequately sampled events and uses their
sampled peak to set amplitude; amplitude results from these policies should not
be compared as though the definitions were identical.

## Viewing and interpreting results

Open [notebooks/evaluation.ipynb](notebooks/evaluation.ipynb) and run its cells.
Set `RUN` to the desired output base name. The notebook selects the latest
completed matching run, including numbered variants, and displays saved results
without repeating inference. Check the displayed directory before interpreting
a comparison. [notebooks/injection_validation.ipynb](notebooks/injection_validation.ipynb)
shows representative saved controls, injected curves, and added signals.

Each detector has three main outputs:

- **Precision–recall curve:** classification of injected curves versus original
  controls as the score threshold changes. The combined curve varies a joint
  score; its selected operating point requires both tuned detector thresholds.
- **Confusion matrix:** accepted/rejected counts and within-class percentages
  at the tuning-selected thresholds. Read percentages alongside counts because
  injections substantially outnumber controls.
- **Amplitude sensitivity:** performance versus injected strength, with
  target-level uncertainty. CNN sensitivity is injected-curve acceptance.
  Matched-filter sensitivity requires a detected peak near a true event; the
  combined pipeline also requires CNN acceptance. Periodic curves can contribute
  multiple events to localized recovery but only one CNN classification.

The amplitude comparison uses adequately observed test injections. Observability
requires at least three exposures and 50% coverage within one event FWHM of the
peak. A detected peak matches within the larger of two cadences or half the event
FWHM. FWHM is measured from the exposure-averaged profile. Overall event results
also retain poorly covered events, and paired-control coincidences help identify
recoveries that may instead be background detections.

The reported amplitude limit is the lowest tested amplitude reaching the required
recall at that level and every higher tested level. A missing limit means the
amplitude grid did not establish it. Do not extrapolate a crossing beyond the
tested grid or interpret curve acceptance as proof of localized recovery.

Controls have unknown astrophysical labels. Their acceptance rate is a benchmark
background-candidate rate, rather than a confirmed astrophysical false-positive
rate. Precision depends strongly on the injected/control ratio: the full preset's
64:1 ratio gives an accept-all precision of about 98.46%. These results therefore
do not estimate blind-survey precision or population completeness. Injection into
SPOC light curves also does not measure signal loss earlier in the SPOC pipeline.

### Saved outputs

Within each completed run:

| Output | Use |
|---|---|
| `results/simple_evaluation.png` and `.pdf` | Three-method comparison figure |
| `results/metrics.json` | Classification metrics and component summaries |
| `results/amplitude_recovery.csv` | Amplitude sensitivity by component |
| `results/report.md` | Supporting diagnostic report |
| `results/event_summary.json` | Event recovery, including overall coverage results |
| `config.json`, manifests, and accompanying metadata | Settings, data identities, physical draws, model assets, and dependency information |

Scores and event details are retained so results can be inspected without another
inference run. Keep the run directory and any source directories referenced by
its manifests together.

## Reusing data or restarting scoring

To recompute scores and thresholds from completed injections:

```sh
.venv/bin/python bin/evaluate_pipeline.py examples/evaluation/full.json \
  --resume-from evaluation_runs/full-evaluation-001 --backend cuda_fft
```

This creates a fresh output directory, reuses the original scientific settings
and injections, and repeats scoring. It does not regenerate signals or resume
partway through an interrupted scoring stage. Keep the original directory because
the new run references its injection files.

To change amplitudes, physical distributions, or injection policy, start fresh
injection generation instead. Raw backgrounds can be reused through a prepared
`manifest`. In particular, rescoring old injections will not apply the new
log-uniform source-mass proposal.

## Current status and next experiment

The completed run `evaluation_runs/full-evaluation-002` rescored the injections
from `full-evaluation-001`. Its source masses were drawn using the earlier linear
proposal. A review found no all-positive classification bookkeeping error; weak
recall is a detector-performance result. Those completed data remain available.

The next steps are:

1. Generate and evaluate a fresh cohort with the current log-uniform source-mass
   proposal. Inspect the parameter distributions and representative injections.
2. Review all three methods, coverage counts, control candidates, and uncertainty.
   If the requested recall limit is absent, report that result or explicitly run
   a revised development experiment with a different amplitude grid.
3. Write the methods/results description with the background selection, proposal,
   class mixture, thresholds, recovery definitions, uncertainty, and SPOC scope.

Model improvements are separate from completing the evaluation. For additional
configuration examples and lower-level commands, see the
[evaluation module README](searchlight/evaluation/README.md).
