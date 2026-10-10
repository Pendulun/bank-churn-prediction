import pandas as pd
import pathlib
from sklearn.metrics import f1_score

from src import config, loaders, plotting, utils


def evaluate_models(calib_models: dict, thresh_tunned_models: dict,
                    X: pd.DataFrame, Y: pd.Series):
    results = []

    for model_name, calibrated_model in calib_models.items():

        y_proba = calibrated_model.predict_proba(X)[:, 1]

        y_pred_default = (y_proba >= 0.5).astype(int)

        f1_default = f1_score(Y, y_pred_default)

        best_threshold = thresh_tunned_models[model_name].best_threshold_

        y_pred_tuned = (y_proba >= best_threshold).astype(int)

        f1_tuned = f1_score(Y, y_pred_tuned)

        results.append({
            "model": model_name,
            "f1_default": f1_default,
            "f1_tuned": f1_tuned,
            "threshold": best_threshold
        })

    f1_comparison_df = pd.DataFrame(results)
    return f1_comparison_df


def get_probas(models: dict, X: pd.DataFrame) -> dict:
    tunned_models_test_preds = dict()
    for model_name, model in models.items():
        tunned_models_test_preds[model_name] = model.predict_proba(X)[:, 1]
    return tunned_models_test_preds


def run():
    logger = utils.get_logger(__name__)
    logger.info("Loading calib models...")
    calib_models = loaders.load_models(config.CALIB_MODELS_DIR)

    logger.info("Loading threshold models...")
    thresh_tunned_models = loaders.load_models(config.THRESH_TUNNED_MODELS_DIR)

    logger.info("Loading Test set...")
    test_X, test_Y = loaders.load_splitted_data(config.TEST_DATASET_PATH)

    logger.info("Evaluating models on test set...")
    eval_results = evaluate_models(calib_models, thresh_tunned_models, test_X,
                                   test_Y)

    test_probas = get_probas(thresh_tunned_models, test_X)
    test_gain_at_k = utils.get_gain_at_k(test_Y, test_probas)
    test_lift_at_k = utils.get_lift_at_k(test_Y, test_probas)
    test_lift_per_decile = utils.get_lift_per_decile(test_Y, test_probas)

    logger.info("Plotting results...")
    pathlib.Path(config.MODEL_TEST_DIR_PATH).mkdir(parents=True, exist_ok=True)
    plotting.plot_testing_results(
        eval_results, config.MODEL_TEST_DIR_PATH + "test_results.png")

    plotting.plot_gain_at_k(test_gain_at_k,
                            plot_path=config.MODEL_TEST_DIR_PATH +
                            "test_gain_at_k.png")
    plotting.plot_lift_at_k(test_lift_at_k,
                            plot_path=config.MODEL_TEST_DIR_PATH +
                            "test_lift_at_k.png")
    plotting.plot_lift_at_deciles(test_lift_per_decile,
                                  plot_path=config.MODEL_TEST_DIR_PATH +
                                  "test_lift_per_decile.png")


if __name__ == "__main__":
    run()
