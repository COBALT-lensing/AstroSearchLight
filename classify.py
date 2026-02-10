import argparse
import csv
import sys
from pathlib import Path

import logging

logging.basicConfig(level=logging.DEBUG)

from cnn import ResNetFilter
from tree import XGBFilter

parser = argparse.ArgumentParser(description="Classify images using ResNet or XGBoost")
parser.add_argument(
    "classifier", type=str, help='Which classifier to use, "resnet" or "xgboost"'
)
parser.add_argument(
    "input", type=str, help="Input file or directory (use '-' for stdin)"
)
parser.add_argument("-o", "--output", type=Path, help="Output CSV file for results")

args = parser.parse_args()

classifier = args.classifier.lower()
input_path = Path(args.input) if args.input != "-" else sys.stdin
results_path = args.output

if __name__ == "__main__":
    if classifier == "resnet":
        classifier = ResNetFilter()
    elif classifier == "xgboost":
        classifier = XGBFilter()
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
