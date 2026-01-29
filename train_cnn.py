from lightcurves import LC_DATA_PATH
from cnn import ResNetFilter

if __name__ == "__main__":
    filter = ResNetFilter()
    filter.train(LC_DATA_PATH)
