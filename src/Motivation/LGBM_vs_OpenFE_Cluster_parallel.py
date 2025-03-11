import argparse
import logging
import os

import psutil
import ray

from sklearn.metrics import log_loss
from sklearn import preprocessing
import pandas as pd

from autogluon.tabular import TabularPredictor

from tabrepo_2024_custom import zeroshot2024
import openml
from OpenFE.openfe_parallel import OpenFE
import lightgbm as lgb


def run_lgbm(X_train, y_train, X_test, y_test):
    """
    # lgb_train = lgb.Dataset(X_train, y_train, params={'verbose': -1})
    # lgb_eval = lgb.Dataset(X_test, y_test, params={'verbose': -1}, reference=lgb_train)
    
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
        "seed": 42,
    }

    # gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=[lgb_eval], callbacks=[lgb.early_stopping(stopping_rounds=5))
    # y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration, verbose_eval=False)
    # log_loss_test = log_loss(y_test, y_pred) ** 0.5
    """

    clf = lgb.LGBMClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(y_test)
    print(y_pred)
    df_train = pd.concat([X_train, pd.Series(y_train)], axis=1)
    df_test = pd.concat([X_test, pd.Series(y_test)], axis=1)
    df_original = pd.concat([df_train, df_test], axis=0)
    labels = df_original.iloc[:,-1].unique()
    log_loss_test = log_loss(y_test, y_pred, labels=labels) ** 0.5
    print(log_loss_test)
    return log_loss_test


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
        verbosity=0,
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
        #seed=42
    )
    predictor.fit_summary(verbosity=-1)
    lb = predictor.leaderboard(X_test)
    log_loss_test = lb.score_test[0]
    ray.shutdown()

    return log_loss_test


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

def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 * 1024)  # Memory usage in MB
    print(f"Memory usage: {memory_usage:.2f} MB")


def main(args):
    log_memory_usage()
    dataset_id = args.dataset
    df = pd.DataFrame(columns=['Dataset', 'LGBM', 'OpenFE + LGBM', 'Autogluon LGBM', 'OpenFE + Autogluon LGBM', 'Tuned Autogluon LGBM', 'OpenFE + Tuned Autogluon LGBM'])
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        f.write("Dataset: " + str(dataset_id) + "\n")
    log_memory_usage()
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        f.write("Get OpenML Data\n")
    X_train, y_train, X_test, y_test = get_openml_dataset(dataset_id)
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        #pd.set_option('display.max_columns', None)
        f.write(str(X_train))
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        f.write("Factorize Data\n")
    X_train, y_train, X_test, y_test = factorize_data_old(X_train, y_train, X_test, y_test)
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        #pd.set_option('display.max_columns', None)
        f.write(str(X_train))
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        f.write("Use OpenFE\n")
    X_train_openfe, y_train_openfe, X_test_openfe, y_test_openfe = get_openfe_data(X_train, y_train, X_test, y_test,                                                                               str(dataset_id))
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        #pd.set_option('display.max_columns', None)
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
    results_of_dataset = pd.Series({'Dataset': dataset_id, 'LGBM': lgbm_results, 'OpenFE + LGBM': lgbm_openfe_results, 'Autogluon LGBM': autogluon_lgbm_results, 'OpenFE + Autogluon LGBM': autogluon_lgbm_openfe_results, 'Tuned Autogluon LGBM': tuned_autogluon_lgbm_results, 'OpenFE + Tuned Autogluon LGBM': tuned_autogluon_lgbm_openfe_results})
    df.loc[len(df)] = results_of_dataset
    parquet_path = "results_" + str(dataset_id) + ".parquet"
    df.to_parquet(path=parquet_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Motivation')
    parser.add_argument('--dataset', type=int, required=True, help='Motivation dataset')
    args = parser.parse_args()
    main(args)
