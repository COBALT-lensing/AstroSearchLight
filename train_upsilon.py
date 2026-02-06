from lightcurves import LC_DATA_PATH

from upsilon import UpsilonFilter

if __name__ == "__main__":
    filter = UpsilonFilter()
    filter.train(LC_DATA_PATH)
