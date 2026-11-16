import argparse
import csv
import sys
import warnings
from pathlib import Path

from searchlight.models.simulens import SimuLens
from searchlight.config import validate_version

# Suppress scipy.stats.shapiro UserWarning
warnings.filterwarnings("ignore", category=UserWarning, message=".*scipy.stats.shapiro")
# Suppress upsilont RuntimeWarnings
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*divide by zero.*log10"
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=".*invalid value encountered in scalar divide",
)


parser = argparse.ArgumentParser(description="Classify images using ResNet or XGBoost")
parser.add_argument(
    "classifier", type=str, help='Which classifier to use, "resnet" or "xgboost"'
)
parser.add_argument(
    "input", type=str, help="Input file or directory (use '-' for stdin)"
)
parser.add_argument("-o", "--output", type=Path, help="Output CSV file for results")

parser.add_argument("--model-version", type=validate_version, help="CNN version (defaults to the configured version)")

args = parser.parse_args()

classifier = args.classifier.lower()
input_path = Path(args.input) if args.input != "-" else sys.stdin
results_path = args.output

if __name__ == "__main__":
    if classifier == "resnet":
        classifier = SimuLens(version=args.model_version, pretrained=False, loader_workers=32)
    elif classifier == "xgboost":
        from searchlight.models.deprecated.tree import XGBFilter
        classifier = XGBFilter()
    elif classifier == "upsilon":
        from searchlight.models.deprecated.upsilon import UpsilonFilter
        classifier = UpsilonFilter()
    else:
        print("Unknown classifier specified", file=sys.stdout)
        sys.exit(3)

    if input_path == sys.stdin:
        paths = input_path.read().split()
        paths = [p.strip() for p in paths]
        classifications = classifier.classify_paths(paths)
    elif input_path.is_file():
        with open(input_path, "r") as input_file:
            paths = input_file.readlines()
            paths = [p.strip() for p in paths]
        classifications = classifier.classify_paths(paths)
    elif input_path.is_dir():
        classifications = classifier.classify_folder(input_path)
    elif not input_path.exists():
        print(f"{input_path}: Path does not exist", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"{input_path}: Is not a directory nor a file.", file=sys.stderr)
        sys.exit(2)

    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as results_f:
            results_w = csv.writer(results_f)
            results_w.writerow(["path", "label"] + classifier.classes)
            results_w.writerows(
                [
                    [path] + list(classification)
                    for path, classification in classifications.items()
                ]
            )
    else:
        for path, classification in classifications.items():
            print(*([path] + classification))
