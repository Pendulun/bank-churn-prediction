import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibrationDisplay


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
