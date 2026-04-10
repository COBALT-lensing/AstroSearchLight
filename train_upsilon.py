from .searchlight.lightcurves import LC_DATA_PATH
from .searchlight.models.deprecated.upsilon import UpsilonFilter

if __name__ == "__main__":
    filter = UpsilonFilter()
    filter.train(LC_DATA_PATH)
