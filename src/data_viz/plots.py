import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import seaborn as sns

from src import config, loaders, utils


def plot_feats_corr_with_target(X: pd.DataFrame,
                                Y: pd.Series,
                                dir_path: str = None):
    corrs: pd.Series = X.select_dtypes("number").corrwith(Y).sort_values(
        ascending=True)

    corrs.plot(kind='barh')
    plt.title("Feats correlation with churn")
    plt.tight_layout()
    if dir_path:
        plt.savefig(dir_path + "corrs_with_target.png")
    plt.close()


def plot_feat_kde_by_target_y(X: pd.DataFrame,
                              Y: pd.Series,
                              x_col: str,
                              title: str,
                              dir_path: str = None):
    sns.kdeplot(data=X, x=x_col, hue=Y, common_norm=False)
    plt.title(title)
    plt.tight_layout()
    if dir_path:
        plt.savefig(dir_path + f"{x_col}_vs_y.png")
    plt.close()


def plot_churn_rate_by_transaction_count(X: pd.DataFrame,
                                         Y: pd.Series,
                                         dir_path: str = None):
    transactions = sorted(X['Total_Trans_Ct'].unique().tolist())
    churn_rate = list()
    for t in transactions:
        target_index = X[X['Total_Trans_Ct'] == t].index
        churn_rate.append(Y[target_index].mean())
    plt.plot(transactions, churn_rate)
    plt.axhline(0.5, color='gray', linestyle='--', label='50% churn rate')
    plt.legend()
    plt.title("Churn rate by transaction count")
    plt.ylabel("Churn rate")
    plt.xlabel("Transaction count")
    if dir_path:
        plt.savefig(dir_path + "churn_rate_by_trans_count.png")
    plt.close()


def relplot_between_feats_divided_by_target(X: pd.DataFrame,
                                            Y: pd.Series,
                                            x_col: str,
                                            y_col: str,
                                            dir_path: str = None):
    sns.relplot(data=X, x=x_col, y=y_col, col=Y, alpha=0.5)
    if dir_path:
        plt.savefig(dir_path + f"{x_col}_vs_{y_col}.png")
    plt.close()


def plot_numerical(X: pd.DataFrame, Y: pd.Series):
    plot_feats_corr_with_target(X, Y, dir_path=config.DATA_VIZ_DIR_PATH)

    feats_and_titles = [
        ('Contacts_Count_12_mon', "Contacts count distribuition by client"),
        ('Months_Inactive_12_mon', "Months inactive distribuition by client"),
        ('Total_Trans_Ct', "Transaction count distribuition by client"),
        ('Total_Ct_Chng_Q4_Q1',
         "Transaction count change Q4-Q1 distribuition by client"),
        ('Months_on_book', "Months on book distribuition by client"),
    ]

    for feat, title in feats_and_titles:
        plot_feat_kde_by_target_y(X,
                                  Y,
                                  feat,
                                  title=title,
                                  dir_path=config.DATA_VIZ_DIR_PATH)

    plot_churn_rate_by_transaction_count(X,
                                         Y,
                                         dir_path=config.DATA_VIZ_DIR_PATH)

    relplot_between_feats_divided_by_target(X,
                                            Y,
                                            'Total_Trans_Ct',
                                            'Total_Ct_Chng_Q4_Q1',
                                            dir_path=config.DATA_VIZ_DIR_PATH)

    relplot_between_feats_divided_by_target(X,
                                            Y,
                                            'Months_on_book',
                                            'Total_Ct_Chng_Q4_Q1',
                                            dir_path=config.DATA_VIZ_DIR_PATH)


def run():
    logger = utils.get_logger(__name__)
    logger.info("Loading train set...")

    train_X, train_Y = loaders.load_splitted_data(config.TRAIN_DATASET_PATH,
                                                  all_x_cols=True)

    pathlib.Path(config.DATA_VIZ_DIR_PATH).mkdir(exist_ok=True, parents=True)

    logger.info("Plotting...")
    plot_numerical(train_X, train_Y)


if __name__ == "__main__":
    run()
