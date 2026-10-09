import logging
import pandas as pd
import pathlib

from src import config


def get_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)
    return logger


def binarize_col_inplace(data: pd.DataFrame, col: str,
                         target_value: str) -> pd.DataFrame:
    """
    Binarize a column inplace checking against the target_value
    """
    data[col] = (data[col] == target_value).astype(int)
    return data


def main():
    logger = get_logger()
    logger.info(f"Reading raw dataset at {config.RAW_DATASET_PATH}")
    data = pd.read_csv(config.RAW_DATASET_PATH)
    logger.info(f"Preprocessing...")
    preprocessed = binarize_col_inplace(data, config.Y_COL,
                                        'Attrited Customer')
    logger.info(
        f"Saving preprocessing data to {config.PREPROCESSED_DATASET_PATH}")
    pathlib.Path(config.PREPROCESSED_DATASET_PATH).parent.mkdir(parents=True,
                                                                exist_ok=True)
    preprocessed.to_csv(config.PREPROCESSED_DATASET_PATH, index=None)


if __name__ == "__main__":
    main()
