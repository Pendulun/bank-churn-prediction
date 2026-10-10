import pandas as pd
from sklearn.model_selection import TunedThresholdClassifierCV

from src import config, loaders, utils


def tune(models: dict,
         X: pd.DataFrame,
         Y: pd.Series,
         random_seed: int = 42) -> dict:
    tuned_models = {}

    for model_name, calibrated_model in models.items():

        tuned_model = TunedThresholdClassifierCV(estimator=calibrated_model,
                                                 scoring="f1",
                                                 cv="prefit",
                                                 refit=False,
                                                 thresholds=100,
                                                 random_state=random_seed,
                                                 n_jobs=-1)

        tuned_model.fit(X, Y)

        tuned_models[model_name] = tuned_model

        print(f"{model_name}: "
              f"best threshold = {tuned_model.best_threshold_:.4f}")

    return tuned_models


def run():
    logger = utils.get_logger(__name__)

    logger.info("Loading calibrated models...")
    calib_models = loaders.load_models(config.CALIB_MODELS_DIR)

    logger.info("Loading threshold tunning data...")
    thresh_X, thresh_Y = loaders.load_splitted_data(
        config.THRESH_TUNNING_DATASET_PATH)

    logger.info("Tunning classification threshold...")
    tunned_models = tune(calib_models,
                         thresh_X,
                         thresh_Y,
                         random_seed=config.RANDOM_STATE)

    logger.info("Saving threshold tunned models...")
    loaders.save_models(config.THRESH_TUNNED_MODELS_DIR, tunned_models)


if __name__ == "__main__":
    run()
