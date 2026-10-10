import pandas as pd
import pathlib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from src import config, loaders, plotting, utils


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


def plot_all(oof_predictions: dict, Y: pd.Series):
    plotting.roc_curve(oof_predictions,
                       Y.values,
                       plot_path=config.MODEL_EVAL_DIR_PATH +
                       "base_roc_curve.png")
    plotting.precision_recall_curve(oof_predictions,
                                    Y.values,
                                    plot_path=config.MODEL_EVAL_DIR_PATH +
                                    "precision_recall_curve.png")

    precisions_at_k = utils.get_precisions_at_k(Y, oof_predictions)
    plotting.plot_precisions_at_k(precisions_at_k,
                                  plot_path=config.MODEL_EVAL_DIR_PATH +
                                  "precision_at_k_curve.png")

    gain_at_k = utils.get_gain_at_k(Y, oof_predictions)
    lift_at_k = utils.get_lift_at_k(Y, oof_predictions)
    lift_per_decile = utils.get_lift_per_decile(Y, oof_predictions)

    plotting.plot_gain_at_k(gain_at_k,
                            plot_path=config.MODEL_EVAL_DIR_PATH +
                            "gain_at_k_curve.png")
    plotting.plot_lift_at_k(lift_at_k,
                            plot_path=config.MODEL_EVAL_DIR_PATH +
                            "lift_at_k_curve.png")
    plotting.plot_lift_at_deciles(lift_per_decile,
                                  plot_path=config.MODEL_EVAL_DIR_PATH +
                                  "deciles_lift_curve.png")


def run():
    logger = utils.get_logger(__name__)

    logger.info("Loading training data...")
    X, Y = loaders.load_splitted_data(config.TRAIN_DATASET_PATH)

    logger.info("Grid Searching...")
    models_grid = get_models_grid()
    best_models = grid_search_models(models_grid, X, Y)

    logger.info("Getting OOF predictions...")
    oof_predictions = utils.get_oof_preds(best_models, X, Y)

    logger.info("Plotting...")
    pathlib.Path(config.MODEL_EVAL_DIR_PATH).mkdir(exist_ok=True, parents=True)
    plot_all(oof_predictions, Y)

    logger.info("Saving models...")
    loaders.save_models(config.BASE_MODELS_DIR, best_models)


if __name__ == "__main__":
    run()
