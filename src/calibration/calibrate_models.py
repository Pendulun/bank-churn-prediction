import pandas as pd
import pathlib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import StratifiedKFold

from src import config, loaders, plotting, utils


def calibrate_models(models: dict,
                     cal_X: pd.DataFrame,
                     cal_Y: pd.Series,
                     splits: int = 5,
                     random_seed: int = 42) -> dict:
    calibrated_models = dict()
    cv = StratifiedKFold(
        n_splits=splits,
        shuffle=True,
        random_state=random_seed,
    )
    for model_name, model in models.items():
        calibrator = CalibratedClassifierCV(
            FrozenEstimator(model),
            method="isotonic",
            cv=cv,
            n_jobs=-1,
        )
        calibrator.fit(cal_X, cal_Y)
        calibrated_models[model_name] = calibrator

    return calibrated_models


def run():
    logger = utils.get_logger(__name__)
    logger.info("Loading base models...")
    base_models = loaders.load_models(config.BASE_MODELS_DIR)

    logger.info("Loading calibration data...")
    calib_X, calib_Y = loaders.load_splitted_data(config.CALIB_DATASET_PATH)

    logger.info("Calibrating models...")
    calib_models = calibrate_models(base_models,
                                    calib_X,
                                    calib_Y,
                                    splits=5,
                                    random_seed=config.RANDOM_STATE)

    logger.info("Loading training data...")
    train_X, train_Y = loaders.load_splitted_data(config.TRAIN_DATASET_PATH)

    logger.info("Getting OOF predictions...")
    base_oof_predictions = utils.get_oof_preds(base_models, train_X, train_Y)
    calibrated_oof_predictions = utils.get_oof_preds(calib_models, train_X,
                                                     train_Y)

    all_results = dict()
    for model_name, model in calibrated_oof_predictions.items():
        all_results[model_name + "_calib"] = model

    all_results.update(base_oof_predictions)

    logger.info("Ploting...")
    pathlib.Path(config.MODEL_CALIB_DIR_PATH).mkdir(parents=True,
                                                    exist_ok=True)
    target_plot = config.MODEL_CALIB_DIR_PATH + "calibration_display.png"
    plotting.plot_calibration_display(all_results, train_Y, target_plot)

    logger.info("Saving calibrated models...")
    loaders.save_models(config.CALIB_MODELS_DIR, calib_models)


if __name__ == "__main__":
    run()
