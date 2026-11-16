# Artifact versions and defaults

CNN datasets, models, and matched-filter banks are selected by numeric version strings,
such as `"1.0"` or `"2.1"`. The library resolves their locations:

```text
~/.astrosearchlight/
  config.json
  datasets/simulens/
    1.0/
      lensing/
      flare/
      not-lensing/
      generation.json
    1.0-validation/
      lensing/
      flare/
      not-lensing/
    2.0/
      lensing/
      flare/
      not-lensing/
      generation.json
  models/simulens/
    1.0/
      model_weights.pth
      model_classes.json
    2.0/
      model_weights.pth
      model_classes.json
      inference.json
      training_checkpoint.pth
      training_run.json
  templates/matched_filter/
    1.0/templates.npz
    2.0/templates.npz
```

The existing unversioned template bank has been moved into the `1.0` directory.
Historical CNN data from `simulations/1.0` and `simulations/1.0-validation`
now lives in `datasets/simulens/1.0` and `datasets/simulens/1.0-validation`.
The validation directory is a sibling of the training directory, keeping its
curves out of the training class discovery. The CNN's existing `1.0` model
directory is preserved. Generation and training do not
activate a new version automatically.

## Choosing defaults

Edit these constants in `searchlight/config.py`:

```python
DEFAULT_SIMULENS_VERSION = "1.0"
DEFAULT_TEMPLATE_VERSION = "1.0"
DEFAULT_DATASET_VERSION = "1.0"
```

`~/.astrosearchlight/config.json` is initialized to `{}`, which uses those
constants. Optional local selections can override them:

```json
{
  "simulens_version": "2.0",
  "template_version": "2.0",
  "dataset_version": "2.0"
}
```

Selection precedence is **explicit version → configuration file → constant**.
Delete a key from the configuration file to use that constant again. The file is
read when resolving a default; existing CNN instances retain their selected
version. Restart Python processes after changing the code constants.
Only numeric version strings are accepted; dataset/model/bank paths are not selection
arguments. No files are moved or replaced when switching defaults.

## Classifying with a version

```python
from searchlight.models.simulens import SimuLens
from searchlight.matched_filtering import detect_matched_filter_peaks, load_template_library

cnn = SimuLens(version="2.0", pretrained=False)
results = cnn.classify_paths(lightcurve_paths)

matches = detect_matched_filter_peaks(time, flux, template_version="2.0")
bank = load_template_library("2.0")
```

Omit the version to use the configured default. For the classification CLI:

```sh
uv run python bin/classify.py resnet /path/to/lightcurves --model-version 2.0
```

Classification restores segment lengths and the segment cap from `inference.json`.
For earlier trained runs it reads `training_run.json` instead. An older model
without either file uses the historical full-curve/7-day/2-day segments and cap
of 24. Explicit constructor segment settings override the saved settings.
The evaluation CNN uses the same restoration rules.

## Creating and preserving versions

Generate a dataset version and select it independently of the model version:

```sh
uv run python bin/generate_training_data.py --version 2.0
uv run python bin/train_cnn.py --dataset-version 2.0 --version 2.0
uv run python bin/generate_matched_filter_template_library.py --version 2.0
```

Dataset generation requires an explicit new version and writes to
`~/.astrosearchlight/datasets/simulens/<version>/`. An existing dataset version
cannot be overwritten, including by the training CLI's `--overwrite` option.
An interrupted generation also reserves its version; choose a new version to retry.
Training's `--dataset-version` is optional: omitting it selects `dataset_version`
from the configuration or `DEFAULT_DATASET_VERSION`. The model version remains
independent; several model versions can train on the same dataset version.
Training records the resolved dataset version, generation provenance, and file
hashes. Resume requires the same dataset version and contents.

Training and template generation require an explicit model/bank version. Existing
models and banks are protected unless `--overwrite` is supplied. Use a new version
to retain the previous data, model, and bank. Fine-tune with `--init-version 1.0`, or continue a
partially completed run with `--dataset-version 2.0 --version 2.0 --resume`. These do not change the
default version selections.

The Python equivalents are `generate_cnn_dataset("2.0")` and
`SimuLens(version="3.0").train(dataset_version="2.0")`. Existing external or
historical datasets can be placed under their own version directory with the
class-folder layout above. `config.validation_dataset_directory("1.0")` resolves
the separately preserved historical validation set. Compatibility aliases
`lightcurves.LC_DATA_PATH` and `LC_VALIDATION_PATH` use these configured paths;
restart notebooks after changing the selected version.

After evaluating the new artifacts, change the constants or configuration file
to select them. Roll back by selecting the previous versions.

Evaluation JSON configurations accept `model_version` and `template_version`.
Use a numeric string to pin an experiment or `null` to use the configured default.
Saved experiment snapshots retain resolved paths and hashes for reproducibility;
historical run configurations with those paths remain readable.

For another installation with an old unversioned bank or datasets under
`simulations/`, run
`uv run python -m searchlight.config` once. This creates the configuration file
if absent, moves the legacy bank to `1.0`, and migrates historical dataset
version directories into `datasets/simulens/`. Dataset migration preflights all
destinations and refuses to merge with existing versions. Repeating it is safe.
