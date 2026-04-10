from .searchlight.lightcurves import LC_DATA_PATH
from .searchlight.models.deprecated.tree import XGBFilter

if __name__ == "__main__":
    filter = XGBFilter()
    filter.train(LC_DATA_PATH)
