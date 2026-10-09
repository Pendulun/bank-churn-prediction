# bank-churn-prediction
Churn prediction with a [Kaggle dataset](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) using credit card data. Focused more on the process rather than on the results. It features the following steps: Data viz, Hiperparameter tunning, Model calibration and Threshold tunning.

The objective is to have a model that estimates churn probabilities by customer.

# Requirements

This project requires Python >=3.13. It is recomended to use `uv` to install all dependencies.

# Steps

## Install dependencies
At the projects root folder and with `uv` installed, run: `uv sync`. This will create a virtual environment inside the project with all the dependencies.

## Downloading data

To download the dataset, run: `uv run python src/data/download_data.py`. It will download the data at `./data/BankChurners.csv`
