import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src import config


def get_oof_preds(best_models: dict, train_X, train_Y) -> np.ndarray:
    oof_predictions = {}
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )
    for name, model in best_models.items():

        oof_proba = cross_val_predict(
            model,
            train_X,
            train_Y,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]

        oof_predictions[name] = oof_proba
    return oof_predictions


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


## Ploting funcs
def roc_curve(models_predictions: dict,
              true_Y: np.ndarray,
              plot_path: str = None):
    fig, ax = plt.subplots()
    for model_name, preds in models_predictions.items():
        RocCurveDisplay.from_predictions(true_Y, preds, name=model_name, ax=ax)
    if plot_path:
        plt.savefig(plot_path)
    plt.close()


def precision_recall_curve(models_predictions: dict,
                           true_Y: np.ndarray,
                           plot_path: str = None):
    fig, ax = plt.subplots()
    for model_name, preds in models_predictions.items():
        PrecisionRecallDisplay.from_predictions(true_Y,
                                                preds,
                                                name=model_name,
                                                ax=ax)
    if plot_path:
        plt.savefig(plot_path)
    plt.close()


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


def plot_precisions_at_k(precisions_at_k_per_model: dict,
                         plot_path: str = None):
    fig, ax = plt.subplots()
    ks = range(
        1,
        len(precisions_at_k_per_model[list(
            precisions_at_k_per_model.keys())[0]]) + 1)
    for model_name, precisions in precisions_at_k_per_model.items():
        ax.plot(ks, precisions, label=model_name)

    model_dots = dict()
    for pctg in range(10, 101, 10):
        k = ks[pctg - 1]
        ax.axvline(k, linestyle="--", color='grey')
        ax.text(k, 0.65, f"{pctg}%", color='grey')
        for model_name, precisions in precisions_at_k_per_model.items():
            precision = precisions[pctg - 1]
            ax.text(k, precision * 1.02, f"{precisions[pctg-1]:.2f}")
            model_dots.setdefault(model_name,
                                  dict()).setdefault('precisions',
                                                     list()).append(precision)
            model_dots.setdefault(model_name,
                                  dict()).setdefault('ks', list()).append(k)

    for model_name, model_results in model_dots.items():
        ax.scatter(model_results['ks'], model_results['precisions'])

    plt.xlim(0, ks[-1] * 1.1)

    plt.legend()
    plt.xlabel("%")
    plt.ylabel("Precision at k%")
    plt.title("Precision@k% curve")
    if plot_path:
        plt.savefig(plot_path)
    plt.close()


def plot_gain_at_k(gain_at_k_per_model: dict, plot_path: str = None):
    fig, ax = plt.subplots()
    ks_size = len(list(gain_at_k_per_model.values())[0])
    ks = range(1, ks_size + 1)
    for model_name, gains in gain_at_k_per_model.items():
        ax.plot(ks, gains, label=model_name)

    model_dots = dict()
    for pctg in range(10, 101, 10):
        idx = int((pctg / 100) * ks_size) - 1
        k = ks[idx]
        for model_name, gains in gain_at_k_per_model.items():
            gain = gains[idx]
            model_dots.setdefault(model_name,
                                  dict()).setdefault('gains',
                                                     list()).append(gain)
            model_dots.setdefault(model_name,
                                  dict()).setdefault('ks', list()).append(k)

    baseline = 0
    for model_name, model_results in model_dots.items():
        baseline = model_results['gains'][-1]
        ax.scatter(model_results['ks'], model_results['gains'])

    ax.plot([0, ks_size], [0, baseline],
            linestyle='--',
            color='grey',
            label='random')

    plt.xlim(0, ks[-1] * 1.1)
    plt.ylim(0)

    plt.legend()
    plt.xlabel("k")
    plt.ylabel("Gain at k")
    plt.title("gain@k curve")
    if plot_path:
        plt.savefig(plot_path)
    plt.close()


def plot_lift_at_k(lift_at_k_per_model: dict, plot_path: str = None):
    fig, ax = plt.subplots()
    ks_size = len(list(lift_at_k_per_model.values())[0])
    ks = range(1, ks_size + 1)
    for model_name, lifts in lift_at_k_per_model.items():
        ax.plot(ks, lifts, label=model_name)

    model_dots = dict()
    for pctg in range(10, 101, 10):
        idx = int((pctg / 100) * ks_size) - 1
        k = ks[idx]
        for model_name, lifts in lift_at_k_per_model.items():
            lift = lifts[idx]
            model_dots.setdefault(model_name,
                                  dict()).setdefault('lifts',
                                                     list()).append(lift)
            model_dots.setdefault(model_name,
                                  dict()).setdefault('ks', list()).append(k)

    baseline = 0
    for model_name, model_results in model_dots.items():
        baseline = model_results['lifts'][-1]
        ax.scatter(model_results['ks'], model_results['lifts'])

    plt.xlim(0, ks[-1] * 1.1)
    plt.ylim(0)

    plt.legend()
    plt.xlabel("%")
    plt.ylabel("Lift")
    plt.title("Cumulative Lift")
    if plot_path:
        plt.savefig(plot_path)
    plt.close()


def plot_lift_at_deciles(lifts_per_decile: dict, plot_path: str = None):
    lifts_dfs = list()
    for model_name, lifts in lifts_per_decile.items():
        model_lifts = pd.DataFrame()
        model_lifts['lift'] = lifts
        model_lifts['decile'] = range(1, len(lifts) + 1)
        model_lifts['model_name'] = model_name
        lifts_dfs.append(model_lifts)

    lifts_df = pd.concat(lifts_dfs)

    ax = sns.barplot(data=lifts_df, x='decile', y='lift', hue='model_name')
    ax.axhline(y=1, color='gray', linestyle='--', label="1.0X")
    plt.title("Deciles Lift")
    plt.legend()
    if plot_path:
        plt.savefig(plot_path)
    plt.close()
