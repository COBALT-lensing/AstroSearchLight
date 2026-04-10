from .searchlight.lightcurves import LC_DATA_PATH
from .searchlight.models.simulens import SimuLens

if __name__ == "__main__":
    filter = SimuLens()
    filter.train(LC_DATA_PATH)
