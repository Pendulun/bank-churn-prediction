import joblib
import logging
import pandas as pd
import pathlib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from src import config, utils
from src.model import eval_model


def get_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)
    return logger


def grid_search_models(models: dict,
                       train_X: pd.DataFrame,
                       train_Y: pd.Series,
                       scoring: str = 'f1',
                       random_seed: int = 42) -> dict:
    best_models = dict()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_seed,
    )

    for model_name, model_config in models.items():
        print('Evaluating', model_name)
        model = model_config['base_model']
        param_grid = model_config['param_grid']

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        grid.fit(train_X, train_Y)

        best_models[model_name] = grid.best_estimator_

        print("Best params:", grid.best_params_)
        print(f"{scoring}:", grid.best_score_)
    return best_models


def get_models_grid() -> dict:
    models = {
        "logistic": {
            'base_model':
            LogisticRegression(solver='saga',
                               class_weight='balanced',
                               max_iter=1000),
            'param_grid': {
                'C': [0.05, 0.1, 0.25, 0.5, 0.75, 1, 2, 3],
                'l1_ratio': [0.0, 0.3, 0.5, 0.8, 1.0]
            }
        },
        'decision_tree': {
            'base_model':
            DecisionTreeClassifier(random_state=config.RANDOM_STATE,
                                   class_weight='balanced'),
            'param_grid': {
                'max_depth': [2, 5, 7, 10, 15],
                'min_samples_split': [2, 10, 30, 50],
            }
        }
    }
    return models


def load_train_data():
    data = pd.read_csv(config.TRAIN_DATASET_PATH)

    train_X = data[config.INPUT_COLS]
    train_Y = data[config.Y_COL]
    return train_X, train_Y


def plot_all(oof_predictions: dict, Y: pd.Series):
    eval_model.roc_curve(oof_predictions,
                         Y.values,
                         plot_path=config.MODEL_EVAL_DIR_PATH +
                         "base_roc_curve.png")
    eval_model.precision_recall_curve(oof_predictions,
                                      Y.values,
                                      plot_path=config.MODEL_EVAL_DIR_PATH +
                                      "precision_recall_curve.png")

    precisions_at_k = eval_model.get_precisions_at_k(Y, oof_predictions)
    eval_model.plot_precisions_at_k(precisions_at_k,
                                    plot_path=config.MODEL_EVAL_DIR_PATH +
                                    "precision_at_k_curve.png")

    gain_at_k = utils.get_gain_at_k(Y, oof_predictions)
    lift_at_k = utils.get_lift_at_k(Y, oof_predictions)
    lift_per_decile = utils.get_lift_per_decile(Y, oof_predictions)

    eval_model.plot_gain_at_k(gain_at_k,
                              plot_path=config.MODEL_EVAL_DIR_PATH +
                              "gain_at_k_curve.png")
    eval_model.plot_lift_at_k(lift_at_k,
                              plot_path=config.MODEL_EVAL_DIR_PATH +
                              "lift_at_k_curve.png")
    eval_model.plot_lift_at_deciles(lift_per_decile,
                                    plot_path=config.MODEL_EVAL_DIR_PATH +
                                    "deciles_lift_curve.png")


def run():
    logger = get_logger()

    logger.info("Loading training data...")
    X, Y = load_train_data()

    logger.info("Grid Searching...")
    models_grid = get_models_grid()
    best_models = grid_search_models(models_grid, X, Y)

    logger.info("Getting OOF predictions...")
    oof_predictions = eval_model.get_oof_preds(best_models, X, Y)

    logger.info("Plotting...")
    pathlib.Path(config.MODEL_EVAL_DIR_PATH).mkdir(exist_ok=True, parents=True)
    plot_all(oof_predictions, Y)

    logger.info("Saving models...")
    pathlib.Path(config.BASE_MODELS_DIR).mkdir(exist_ok=True, parents=True)
    for model_name, model in best_models.items():
        joblib.dump(model, config.BASE_MODELS_DIR + f"{model_name}.joblib")


if __name__ == "__main__":
    run()
