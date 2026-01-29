from lightcurves import LC_DATA_PATH
from tree import XGBFilter

if __name__ == "__main__":
    filter = XGBFilter()
    filter.train(LC_DATA_PATH)
