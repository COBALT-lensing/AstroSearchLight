import sys
import csv

from pathlib import Path

from resnet import ResNetFilter

results_path = sys.argv[2] if len(sys.argv) > 2 else None
input_path = Path(sys.argv[1])

if __name__ == "__main__":
    if not input_path.exists():
        print(f"{input_path}: Path does not exist")
        sys.exit(1)

    rn = ResNetFilter()

    if input_path.is_file():
        with open(input_path, "r") as input_file:
            paths = input_file.readlines()
            paths = [p.strip() for p in paths]
        labels = rn.classify_paths(paths)
    elif input_path.is_dir():
        labels = rn.classify_folder(input_path)
    else:
        print(f"{input_path}: Is not a directory nor a file.")
        sys.exit(2)

    if results_path is not None:
        with open(results_path, "w") as results_f:
            results_w = csv.writer(results_f)
            results_w.writerows([(path, label) for path, label in labels.items()])
    else:
        for path, label in labels.items():
            print(path, label)
