import sys
import csv

from pathlib import Path

from resnet import ResNetFilter
import argparse

parser = argparse.ArgumentParser(description="Classify images using ResNet")
parser.add_argument(
    "input", type=str, help="Input file or directory (use '-' for stdin)"
)
parser.add_argument("-o", "--output", type=Path, help="Output CSV file for results")
# parser.add_argument(
#     "--skip-existing",
#     action="store_true",
#     help="Skip classification if output file exists",
# )
args = parser.parse_args()

input_path = Path(args.input) if args.input != "-" else sys.stdin
results_path = args.output
# skip_existing = args.skip_existing

if __name__ == "__main__":
    # if results_path is not None and skip_existing:
    #     with open(results_path, "r") as results_f:
    #         r = csv.reader(results_f)
    #         skip = {row[0] for row in r}
    # else:
    #     skip = set()

    rn = ResNetFilter()

    if input_path == sys.stdin:
        paths = input_path.read().split()
        paths = [p.strip() for p in paths]
        classifications = rn.classify_paths(paths)
    elif input_path.is_file():
        with open(input_path, "r") as input_file:
            paths = input_file.readlines()
            paths = [p.strip() for p in paths]
        classifications = rn.classify_paths(paths)
    elif input_path.is_dir():
        classifications = rn.classify_folder(input_path)
    elif not input_path.exists():
        print(f"{input_path}: Path does not exist", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"{input_path}: Is not a directory nor a file.", file=sys.stderr)
        sys.exit(2)

    if results_path is not None:
        with open(results_path, "w") as results_f:
            results_w = csv.writer(results_f)
            results_w.writerow(["path", "label"] + rn.classes)
            results_w.writerows(
                [
                    [path] + classification
                    for path, classification in classifications.items()
                ]
            )
    else:
        for path, classification in classifications.items():
            print(*([path] + classification))
