import logging
import pandas as pd
import pathlib
from sklearn.model_selection import train_test_split

from src import config, loaders


def get_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)
    return logger


def get_train_test_split(data: pd.DataFrame,
                         target_col: str,
                         test_size: float = 0.1,
                         random_state: int = 42) -> tuple[pd.DataFrame]:
    X = data.drop(columns=[target_col])
    y = data[target_col]
    train_X, test_X, train_Y, test_Y = train_test_split(
        X, y, random_state=random_state, stratify=y, test_size=test_size)
    return train_X, test_X, train_Y, test_Y


def run():
    logger = get_logger()
    logger.info("Loading preprocessed data...")
    try:
        data = pd.read_csv(config.PREPROCESSED_DATASET_PATH)
    except Exception as e:
        logger.error(
            f"Couldn't load the preprocessed data! Make sure it was created at {config.PREPROCESSED_DATASET_PATH} !"
        )
        return

    logger.info("Creating Train and Test sets from the processed data...")
    train_X, test_X, train_Y, test_Y = get_train_test_split(
        data,
        target_col=config.Y_COL,
        random_state=config.RANDOM_STATE,
        test_size=config.TEST_SIZE_P)

    logger.info("Creating Calibration set from the Training set...")
    train_X, cal_X, train_Y, cal_Y = train_test_split(
        train_X,
        train_Y,
        test_size=config.CALIB_SIZE_P,
        stratify=train_Y,
        random_state=config.RANDOM_STATE)

    # From the calibration set, we make the threshold tuning set
    logger.info("Creating Threshold tunning set from the Calibration set...")
    cal_X, thresh_X, cal_Y, thresh_Y = train_test_split(
        cal_X,
        cal_Y,
        test_size=config.THRESH_TUNNING_SIZE_P,
        stratify=cal_Y,
        random_state=config.RANDOM_STATE)

    pathlib.Path(config.SPLITTED_DATASET_DIR).mkdir(exist_ok=True,
                                                    parents=True)

    logger.info(
        f"Train size: {train_X.shape[0]} ({(train_X.shape[0]/data.shape[0])*100:.2f}%)."
    )
    logger.info(
        f"Test size: {test_X.shape[0]} ({(test_X.shape[0]/data.shape[0])*100:.2f}%)."
    )
    logger.info(
        f"Calibration size: {cal_X.shape[0]} ({(cal_X.shape[0]/data.shape[0])*100:.2f}%)."
    )
    logger.info(
        f"Threshold tunning size: {thresh_X.shape[0]} ({(thresh_X.shape[0]/data.shape[0])*100:.2f}%)."
    )

    logger.info(f"Saving train set to {config.TRAIN_DATASET_PATH}...")
    loaders.save_splited_data_as_one(train_X, train_Y,
                                     config.TRAIN_DATASET_PATH)

    logger.info(f"Saving test set to {config.TEST_DATASET_PATH}...")
    loaders.save_splited_data_as_one(test_X, test_Y, config.TEST_DATASET_PATH)

    logger.info(f"Saving calibration set to {config.CALIB_DATASET_PATH}...")
    loaders.save_splited_data_as_one(cal_X, cal_Y, config.CALIB_DATASET_PATH)

    logger.info(
        f"Saving threshold tunning set to {config.THRESH_TUNNING_DATASET_PATH}..."
    )
    loaders.save_splited_data_as_one(thresh_X, thresh_Y,
                                     config.THRESH_TUNNING_DATASET_PATH)


if __name__ == "__main__":
    run()
