# %%
from collections import Counter

import numpy
import seaborn

from matplotlib import pyplot

from cnn import ResNetFilter
# from tree import XGBFilter
# from upsilon import UpsilonFilter
from lightcurves import LC_VALIDATION_PATH

# %%
LABELS = sorted(
    path.name for path in LC_VALIDATION_PATH.iterdir() if path.is_dir()
)


# %%
def classify_validation_folders(model):
    validation_results = {
        label: model.classify_folder(LC_VALIDATION_PATH / label, label=label)
        for label in LABELS
    }
    confusion_matrix = numpy.array([
        [
            Counter(v[0] for v in validation_results[label].values())[predicted_label]
            for predicted_label in LABELS
        ]
        for label in LABELS
    ])
    return confusion_matrix

def plot_confusion(confusion_matrix, title=None, out_path=None):
    pyplot.figure(figsize=(8, 6))
    seaborn.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=LABELS, yticklabels=LABELS)
    pyplot.ylabel('True Label')
    pyplot.xlabel('Predicted Label')
    if title is not None:
        pyplot.title(title)
    if out_path is None:
        pyplot.show()
    else:
        pyplot.savefig(out_path)

# %%
# t = XGBFilter()
# t_results, t_confusion = classify_validation_folders(t)
# plot_confusion(t_confusion, "XGBoost Confusion")

# %%
# u = UpsilonFilter()
# u_results, u_confusion = classify_validation_folders(u)
# plot_confusion(u_confusion, "UPSILON-T Confusion")

# %%
r = ResNetFilter(loader_workers=64)
r_confusion = classify_validation_folders(r)
plot_confusion(r_confusion, "ResNet Confusion", "data/confusion_resnet.png")


