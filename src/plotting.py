import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


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


def plot_calibration_display(models_predictions: dict,
                             true_Y: np.ndarray,
                             plot_path: str = None):
    fig, ax = plt.subplots()
    for model_name, preds in models_predictions.items():
        CalibrationDisplay.from_predictions(true_Y,
                                            preds,
                                            name=model_name,
                                            ax=ax,
                                            n_bins=10,
                                            strategy='quantile')
    if plot_path:
        plt.savefig(plot_path)
    plt.tight_layout()
    plt.close()


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


def plot_testing_results(results: pd.DataFrame, plot_path: str = None):
    plot_df = results.melt(id_vars=["model", "threshold"],
                           value_vars=["f1_default", "f1_tuned"],
                           var_name="threshold_type",
                           value_name="f1")

    sns.barplot(data=plot_df, x="model", y="f1", hue="threshold_type")

    plt.xlabel("Model")
    plt.ylabel("F1-score")
    plt.title("F1-score: default vs tuned threshold")
    plt.tight_layout()
    if plot_path:
        plt.savefig(plot_path)
    plt.close()
