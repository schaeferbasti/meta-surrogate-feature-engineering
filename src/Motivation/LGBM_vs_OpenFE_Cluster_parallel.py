import argparse
import logging
import numpy as np
import os

import psutil
import ray

from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from autogluon.tabular import TabularPredictor
from autogluon.tabular.models import LGBModel

# from src.datasets.Splits import _save_stratified_splits
from tabrepo_2024_custom import zeroshot2024
import openml
from OpenFE.openfe_parallel import OpenFE
import lightgbm as lgb


def run_lgbm(X_train, y_train, X_test, y_test):
    lgb_train = lgb.Dataset(X_train, y_train, params={'verbose': -1})
    lgb_eval = lgb.Dataset(X_test, y_test, params={'verbose': -1}, reference=lgb_train)

    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": {"l2", "l1"},
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }
    gbm = lgb.train(
        params, lgb_train, num_boost_round=20, valid_sets=[lgb_eval], callbacks=[lgb.early_stopping(stopping_rounds=5)]
    )
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration, verbose_eval=False)
    rmse_test = mean_squared_error(y_test, y_pred) ** 0.5
    return rmse_test


def run_autogluon_lgbm(X_train, y_train, X_test, y_test, zeroshot=False):
    log = logging.getLogger(__name__)
    ray_mem_in_gb = 48
    log.info(f"Running on SLURM, initializing Ray with unique temp dir with {ray_mem_in_gb}GB.")
    ray_mem_in_b = int(ray_mem_in_gb * (1024.0 ** 3))
    tmp_dir_base_path = "/tmp"
    ray_dir = f"{tmp_dir_base_path}"
    print(f"Start local ray instances. Using {os.environ.get('RAY_MEM_IN_GB')} GB for Ray.")
    ray.shutdown()
    ray.init(
        address="local",
        _memory=ray_mem_in_b,
        object_store_memory=int(ray_mem_in_b * 0.3),
        _temp_dir=ray_dir,
        include_dashboard=False,
        logging_level=logging.INFO,
        log_to_driver=True,
        num_gpus=0,
        num_cpus=8,
        ignore_reinit_error=True
    )

    label = "target"
    X_train["target"] = y_train
    X_test["target"] = y_test

    allowed_models = [
        "GBM",
    ]

    for k in list(zeroshot2024.keys()):
        if k not in allowed_models:
            del zeroshot2024[k]
        else:
            if not zeroshot:
                zeroshot2024[k] = zeroshot2024[k][:1]

    # -- Run AutoGluon
    predictor = TabularPredictor(
        label=label,
        eval_metric="log_loss",  # roc_auc (binary), log_loss (multiclass)
        problem_type="multiclass",  # binary, multiclass
        verbosity=4,
    )

    predictor.fit(
        time_limit=int(60 * 60 * 4),
        memory_limit=48,
        num_cpus=8,
        num_gpus=0,
        train_data=X_train,
        presets="best_quality",
        dynamic_stacking=False,
        hyperparameters=zeroshot2024,
        # Validation Protocol
        num_bag_folds=8,
        num_bag_sets=1,
        num_stack_levels=0,
    )
    predictor.fit_summary(verbosity=-1)
    lb = predictor.leaderboard(X_test)

    ray.shutdown()

    return lb


def get_openfe_data(X_train, y_train, X_test, y_test, name):
    openFE = OpenFE()
    features = openFE.fit(data=X_train, label=y_train, n_jobs=1, name=name)  # generate new features
    X_train_openfe, X_test_openfe = openFE.transform(X_train, X_test, features, n_jobs=1)
    return X_train_openfe, y_train, X_test_openfe, y_test


def get_openml_dataset(openml_task_id: int) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    task = openml.tasks.get_task(
        openml_task_id,
        download_splits=True,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    train_idx, test_idx = task.get_train_test_split_indices()
    X, y = task.get_X_and_y(dataset_format="dataframe")  # type: ignore
    train_x, train_y = X.iloc[train_idx], y.iloc[train_idx]
    test_x, test_y = X.iloc[test_idx], y.iloc[test_idx]
    return train_x, train_y, test_x, test_y


"""
def factorize_data_old(X_train, y_train, X_test, y_test):
    lbl = preprocessing.LabelEncoder()
    for column in X_train.columns: #select_dtypes(include=['object', 'category'])
        # X_train[column], _ = pd.factorize(X_train[column])
        X_train[column] = lbl.fit_transform(X_train[column].astype(int))
    for column in X_test.columns:
        X_test[column] = lbl.fit_transform(X_test[column].astype(int))
    y_train_array, _ = pd.Series.factorize(y_train, use_na_sentinel=False)
    y_train = y_train.replace(y_train_array)
    y_test_array, _ = pd.Series.factorize(y_test, use_na_sentinel=False)
    y_test = y_test.replace(y_test_array)
    return X_train, y_train, X_test, y_test
"""


def factorize_data(X_train, y_train, X_test, y_test):
    # Identify categorical columns
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns

    # Apply LabelEncoder only to categorical columns
    for column in categorical_columns:
        lbl = preprocessing.LabelEncoder()
        X_train[column] = lbl.fit_transform(X_train[column].astype(str))  # Convert to string before encoding
        X_test[column] = lbl.transform(X_test[column].astype(str))  # Apply the same mapping to test data

    # Factorize target labels for consistency
    y_train, label_mapping = pd.factorize(y_train, use_na_sentinel=False)
    y_test = pd.Series(y_test).map(dict(enumerate(label_mapping))).fillna(0).astype(int)  # .interpolate(method="pad").astype(int)  # Ensure mapping consistency

    return X_train, y_train, X_test, y_test


"""
def fix_split_by_dropping_classes(
        x: np.ndarray,
        y: np.ndarray,
        n_splits: int,
        spliter_kwargs: dict,
) -> list[list[list[int], list[int]]]:
    # Fixes stratifed splits for edge case.
    # For each class that has fewer instances than number of splits, we oversample before split to n_splits and then remove all oversamples and
    # original samples from the splits; effectively removing the class from the data without touching the indices.
    val, counts = np.unique(y, return_counts=True)
    too_low = val[counts < n_splits]
    too_low_counts = counts[counts < n_splits]

    y_dummy = pd.Series(y.copy())
    X_dummy = pd.DataFrame(x.copy())
    org_index_max = len(X_dummy)
    invalid_index = []

    for c_val, c_count in zip(too_low, too_low_counts, strict=True):
        fill_missing = n_splits - c_count
        invalid_index.extend(np.where(y == c_val)[0])
        y_dummy = pd.concat(
            [y_dummy, pd.Series([c_val] * fill_missing)],
            ignore_index=True,
        )
        X_dummy = pd.concat(
            [X_dummy, pd.DataFrame(x).head(fill_missing)],
            ignore_index=True,
        )

    invalid_index.extend(list(range(org_index_max, len(y_dummy))))
    splits = _save_stratified_splits(
        _splitter=StratifiedKFold(**spliter_kwargs),
        x=X_dummy,
        y=y_dummy,
        n_splits=n_splits,
    )
    len_out = len(splits)
    for i in range(len_out):
        train_index, test_index = splits[i]
        splits[i][0] = [index for index in train_index if index not in invalid_index]
        splits[i][1] = [index for index in test_index if index not in invalid_index]

    return splits
"""


def log_memory_usage():
    """Logs memory usage of the current process."""
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 * 1024)  # Memory usage in MB
    print(f"Memory usage: {memory_usage:.2f} MB")


def main(args):
    log_memory_usage()
    dataset_id = args.dataset
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        f.write("Dataset: " + str(dataset_id) + "\n")
    log_memory_usage()
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        f.write("Get OpenML Data\n")
    X_train, y_train, X_test, y_test = get_openml_dataset(dataset_id)
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        pd.set_option('display.max_columns', None)
        f.write(str(X_train))
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        f.write("Factorize Data\n")
    X_train, y_train, X_test, y_test = factorize_data(X_train, y_train, X_test, y_test)
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        pd.set_option('display.max_columns', None)
        f.write(str(X_train))
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        f.write("Use OpenFE\n")
    X_train_openfe, y_train_openfe, X_test_openfe, y_test_openfe = get_openfe_data(X_train, y_train, X_test, y_test,
                                                                                   str(dataset_id))
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        pd.set_option('display.max_columns', None)
        f.write(str(X_train))
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        f.write("Start Experiments\n")
    log_memory_usage()
    lgbm_results = run_lgbm(X_train, y_train, X_test, y_test)
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        f.write("LGBM Results: " + str(lgbm_results) + "\n")
    log_memory_usage()
    lgbm_openfe_results = run_lgbm(X_train_openfe, y_train_openfe, X_test_openfe, y_test_openfe)
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        f.write("LGBM OpenFE Results: " + str(lgbm_openfe_results) + "\n")
    log_memory_usage()
    autogluon_lgbm_results = run_autogluon_lgbm(X_train, y_train, X_test, y_test, zeroshot=False)
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        f.write("Autogluon LGBM Results: " + str(autogluon_lgbm_results) + "\n")
    log_memory_usage()
    autogluon_lgbm_openfe_results = run_autogluon_lgbm(X_train_openfe, y_train_openfe, X_test_openfe, y_test_openfe,
                                                       zeroshot=False)
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        f.write("Autogluon LGBM OpenFE Results: " + str(autogluon_lgbm_openfe_results) + "\n")
    log_memory_usage()
    tuned_autogluon_lgbm_results = run_autogluon_lgbm(X_train, y_train, X_test, y_test, zeroshot=True)
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        f.write("Tuned Autogluon LGBM Results: " + str(tuned_autogluon_lgbm_results) + "\n")
    log_memory_usage()
    tuned_autogluon_lgbm_openfe_results = run_autogluon_lgbm(X_train_openfe, y_train_openfe, X_test_openfe,
                                                             y_test_openfe, zeroshot=True)
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        f.write("Tuned Autogluon LGBM OpenFE Results: " + str(tuned_autogluon_lgbm_openfe_results) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Motivation')
    parser.add_argument('--dataset', type=int, required=True, help='Motivation dataset')
    args = parser.parse_args()
    main(args)
