import joblib
import pathlib


def load_models(models_dir: str) -> dict:
    """
    Load all joblib files in dir
    """
    target_dir = pathlib.Path(models_dir)
    target_models_files = target_dir.glob("*joblib")
    models = dict()
    for model_path in target_models_files:
        model_name = model_path.stem
        model = joblib.load(model_path)
        models[model_name] = model

    return models


def save_models(models_dir: str, models: dict):
    pathlib.Path(models_dir).mkdir(exist_ok=True, parents=True)
    for model_name, model in models.items():
        joblib.dump(model, models_dir + f"{model_name}.joblib")
