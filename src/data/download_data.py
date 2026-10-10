import kagglehub

from src import config, utils


def run():
    # Download latest version
    logger = utils.get_logger(__name__)
    logger.info(
        f"Downloading {config.DATASET_NAME} to {config.RAW_DATASET_DIR}...")
    kagglehub.dataset_download(config.DATASET_NAME,
                               path=config.DATASET_FILE_NAME,
                               output_dir=config.RAW_DATASET_DIR)

    logger.info("Done!")


if __name__ == "__main__":
    run()
