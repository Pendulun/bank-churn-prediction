import joblib
import pandas as pd
import pathlib

from src import config


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
    """
    Save models as joblib files to dir
    """
    pathlib.Path(models_dir).mkdir(exist_ok=True, parents=True)
    for model_name, model in models.items():
        joblib.dump(model, models_dir + f"{model_name}.joblib")


def load_raw_data() -> pd.DataFrame:
    """
    Load the raw dataset
    """
    data = pd.read_csv(config.RAW_DATASET_PATH)
    unwanted_cols = [
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"
    ]
    data.drop(columns=unwanted_cols, inplace=True)
    return data


def load_splitted_data(
        file_path: str,
        all_x_cols: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load X and Y data from csv file
    """
    data = pd.read_csv(file_path)
    if all_x_cols:
        X = data.drop(columns=[config.Y_COL])
    else:
        X = data[config.INPUT_COLS]
    y = data[config.Y_COL]
    return X, y


def save_data(data: pd.DataFrame, path: str):
    """
    Save data to path as csv
    """
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=None)


def save_splited_data_as_one(X: pd.DataFrame, Y: pd.Series, path: str):
    X[config.Y_COL] = Y
    X.to_csv(path, index=None)
