import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src import config


def get_logger(logger_name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(logger_name)
    return logger


def get_oof_preds(models: dict, X, y) -> np.ndarray:
    oof_predictions = {}
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )
    for name, model in models.items():

        oof_proba = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]

        oof_predictions[name] = oof_proba
    return oof_predictions


def get_precisions_at_k(y_true: pd.Series, y_preds_per_model: dict) -> dict:
    precisions_at_k = dict()
    total_size = y_true.shape[0]
    ks = [int(total_size * pctg / 100) for pctg in range(1, 101)]
    for model_name, preds in y_preds_per_model.items():
        sorted_idxs = np.argsort(preds)[::-1]
        sorted_real = y_true.reset_index(drop=True).loc[sorted_idxs]

        for k in ks:
            precisions_at_k.setdefault(model_name, list()).append(
                sorted_real[:k].sum() / k)
    return precisions_at_k


def get_gain_at_k(y_true: pd.Series, y_preds_per_model: dict) -> dict:
    gain_at_k = dict()
    for model_name, preds in y_preds_per_model.items():
        sorted_idxs = np.argsort(preds)[::-1]
        sorted_real = y_true.reset_index(drop=True).loc[sorted_idxs]
        gain_at_k[model_name] = sorted_real.values.cumsum()
    return gain_at_k


def get_lift_at_k(y_true: pd.Series, y_preds_per_model: dict) -> dict:
    lift_at_k = dict()
    baseline = y_true.mean()
    total_size = y_true.shape[0]
    ks = [int(total_size * pctg / 100) for pctg in range(1, 101)]
    for model_name, preds in y_preds_per_model.items():
        sorted_idxs = np.argsort(preds)[::-1]
        sorted_real = y_true.reset_index(drop=True).loc[sorted_idxs]
        for k in ks:
            lift_at_k.setdefault(model_name, list()).append(
                (sorted_real[:k].sum() / k) / baseline)
    return lift_at_k


def get_lift_per_decile(y_true: pd.Series, y_preds_per_model: dict) -> dict:
    lifts_per_model = {}

    population_positive_rate = y_true.mean()
    n = len(y_true)

    for model_name, y_preds in y_preds_per_model.items():
        sorted_idxs = np.argsort(y_preds)[::-1]
        sorted_y_true = y_true.reset_index(drop=True).iloc[sorted_idxs]

        lifts = []

        for decile in range(10):
            start = int(decile * n / 10)
            end = int((decile + 1) * n / 10)

            decile_y_true = sorted_y_true.iloc[start:end]

            decile_positive_rate = decile_y_true.mean()
            lift = decile_positive_rate / population_positive_rate

            lifts.append(lift)

        lifts_per_model[model_name] = lifts

    return lifts_per_model
