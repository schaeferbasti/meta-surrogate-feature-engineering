import datetime
import os
import psutil

import pandas as pd

from sklearn.metrics import mean_squared_error
import openml

from openfe import OpenFE, transform

import lightgbm as lgb
from autogluon.tabular import TabularPredictor
from autogluon.tabular.models import LGBModel
from tabrepo_2024_custom import zeroshot2024


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
    import os
    import logging
    import ray
    log = logging.getLogger(__name__)
    ray_mem_in_gb = 48
    log.info(f"Running on SLURM, initializing Ray with unique temp dir with {ray_mem_in_gb}GB.")
    ray_mem_in_b = int(ray_mem_in_gb * (1024.0 ** 3))
    tmp_dir_base_path = "/tmp"
    ray_dir = f"{tmp_dir_base_path}"
    print(f"Start local ray instances. Using {os.environ.get('RAY_MEM_IN_GB')} GB for Ray.")
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
        memory_limit=48 * 1024 * 1024,
        num_cpu=8,
        num_gpu=0,
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


def get_openfe_data(X_train, y_train, X_test, y_test):
    openFE = OpenFE()
    features = openFE.fit(data=X_train, label=y_train, n_jobs=1)  # generate new features
    X_train_openfe, X_test_openfe = transform(X_train, X_test, features, n_jobs=1)
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


def factorize_data(X_train, y_train, X_test, y_test):
    for column in X_train.select_dtypes(include=['object', 'category']).columns:
        X_train[column], _ = pd.factorize(X_train[column])
    for column in X_test.select_dtypes(include=['object', 'category']).columns:
        X_test[column], _ = pd.factorize(X_test[column])
    y_train_array, _ = pd.Series.factorize(y_train, use_na_sentinel=False)
    y_train = y_train.replace(y_train_array)
    y_test_array, _ = pd.Series.factorize(y_test, use_na_sentinel=False)
    y_test = y_test.replace(y_test_array)
    return X_train, y_train, X_test, y_test

def log_memory_usage():
    """Logs memory usage of the current process."""
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 * 1024)  # Memory usage in MB
    print(f"Memory usage: {memory_usage:.2f} MB")


def main():
    with open("results.txt", "w") as f:
        f.write("Test different versions of LGBM with OpenFE \n" + str(datetime.datetime.now()) + "\n\n")
    log_memory_usage()
    dataset_ids = [190411, 189354, 189356, 359979, 146818, 359955, 359960, 359968, 359959, 168757,
                   359954, 359969, 359970, 359984, 168911, 359981, 359962, 359965, 190392, 190137, 359958, 168350,
                   359956, 359975, 359963, 168784, 190146, 146820, 359974, 2073, 359944, 359950, 359942, 359951, 360945,
                   167210, 359930, 359948, 359931, 359932, 359933, 359934, 359939, 359945, 359935, 359940]
    # 359983, 10090
    for dataset_id in dataset_ids:
        with open("results.txt", "a") as f:
            f.write("Dataset: " + str(dataset_id) + "\n")
        log_memory_usage()
        try:
            with open("results.txt", "a") as f:
                f.write("Get OpenML Data\n")
            X_train, y_train, X_test, y_test = get_openml_dataset(dataset_id)
            with open("results.txt", "a") as f:
                f.write("Factorize Data\n")
            X_train, y_train, X_test, y_test = factorize_data(X_train, y_train, X_test, y_test)
            with open("results.txt", "a") as f:
                f.write("Use OpenFE\n")
            X_train_openfe, y_train_openfe, X_test_openfe, y_test_openfe = get_openfe_data(X_train, y_train, X_test, y_test)
        except Exception as e:
            print(e)
            continue
        with open("results.txt", "a") as f:
            f.write("Start Experiments\n")
        try:
            log_memory_usage()
            lgbm_results = run_lgbm(X_train, y_train, X_test, y_test)
            with open("results.txt", "a") as f:
                f.write("LGBM Results " + str(lgbm_results) + "\n")
        except Exception as e:
            with open("results.txt", "a") as f:
                f.write("LGBM Results " + str(e) + "\n")
            continue
        try:
            log_memory_usage()
            lgbm_openfe_results = run_lgbm(X_train_openfe, y_train_openfe, X_test_openfe, y_test_openfe)
            with open("results.txt", "a") as f:
                f.write("LGBM OpenFE Results " + str(lgbm_openfe_results) + "\n")
        except Exception as e:
            with open("results.txt", "a") as f:
                f.write("LGBM OpenFE Results " + str(e) + "\n")
            continue
        try:
            log_memory_usage()
            autogluon_lgbm_results = run_autogluon_lgbm(X_train, y_train, X_test, y_test, zeroshot=False)
            with open("results.txt", "a") as f:
                f.write("Autogluon LGBM Results " + str(autogluon_lgbm_results) + "\n")
        except Exception as e:
            with open("results.txt", "a") as f:
                f.write("Autogluon LGBM Results " + str(e) + "\n")
            continue
        try:
            log_memory_usage()
            autogluon_lgbm_openfe_results = run_autogluon_lgbm(X_train_openfe, y_train_openfe, X_test_openfe, y_test_openfe, zeroshot=False)
            with open("results.txt", "a") as f:
                f.write("Autogluon LGBM OpenFE Results " + str(autogluon_lgbm_openfe_results) + "\n")
        except Exception as e:
            with open("results.txt", "a") as f:
                f.write("Autogluon LGBM OpenFE Results " + str(e) + "\n")
            continue
        try:
            log_memory_usage()
            tuned_autogluon_lgbm_results = run_autogluon_lgbm(X_train, y_train, X_test, y_test, zeroshot=True)
            with open("results.txt", "a") as f:
                f.write("Tuned Autogluon LGBM Results " + str(tuned_autogluon_lgbm_results) + "\n")
        except Exception as e:
            with open("results.txt", "a") as f:
                f.write("Tuned Autogluon LGBM Results " + str(e) + "\n")
            continue
        try:
            log_memory_usage()
            tuned_autogluon_lgbm_openfe_results = run_autogluon_lgbm(X_train_openfe, y_train_openfe, X_test_openfe, y_test_openfe, zeroshot=True)
            with open("results.txt", "a") as f:
                f.write("Tuned Autogluon LGBM OpenFE Results " + str(tuned_autogluon_lgbm_openfe_results) + "\n")
        except Exception as e:
            with open("results.txt", "a") as f:
                f.write("Tuned Autogluon LGBM OpenFE Results " + str(e) + "\n")
            continue


if __name__ == '__main__':
    main()
