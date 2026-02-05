from lightcurves import LC_DATA_PATH

from upsilon import UpsilonFilter

# from upsilont import test_predict

if __name__ == "__main__":
    filter = UpsilonFilter()
    filter.train(LC_DATA_PATH)
    # test_predict()
