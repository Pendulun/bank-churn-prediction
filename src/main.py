from src.data import download_data
from src.data_viz import plots
from src.preprocessing import preprocess
from src.spliting_data import split_data
from src.model import fit_models
from src.calibration import calibrate_models
from src.thresh_tunning import thresh_tune
from src.testing import test


def run():
    download_data.run()
    plots.run()
    preprocess.run()
    split_data.run()
    fit_models.run()
    calibrate_models.run()
    thresh_tune.run()
    test.run()


if __name__ == "__main__":
    run()
