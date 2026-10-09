import kagglehub
import logging

from src import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Download latest version
    logger.info(
        f"Downloading {config.DATASET_NAME} to {config.DATASET_DIR}...")
    kagglehub.dataset_download(config.DATASET_NAME,
                               path=config.DATASET_FILE_NAME,
                               output_dir=config.DATASET_DIR)

    logging.info("Done!")
