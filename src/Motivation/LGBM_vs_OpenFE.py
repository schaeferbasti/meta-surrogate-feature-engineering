import datetime

import pandas as pd

from sklearn.metrics import mean_squared_error
import openml

from openfe import OpenFE, transform

import lightgbm as lgb
from autogluon.tabular.src.autogluon.tabular import TabularPredictor
from autogluon.tabular.src.autogluon.tabular.models import LGBModel
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


def run_autogluon_lgbm(X_train, y_train, X_test, y_test):
    autogluon = LGBModel()
    autogluon.fit(X=X_train, y=y_train)
    y_pred = autogluon.predict(X=X_test)
    rmse_test = mean_squared_error(y_test, y_pred) ** 0.5
    return rmse_test

def run_tuned_autogluon_lgbm(X_train, y_train, X_test, y_test):
    label = "target"
    X_train["target"] = y_train

    allowed_models = [
        "GBM",
    ]

    for k in list(zeroshot2024.keys()):
        if k not in allowed_models:
            del zeroshot2024[k]

    # -- Run AutoGluon
    predictor = TabularPredictor(
        label=label,
        eval_metric="mcc",
        problem_type="multiclass",
        verbosity=-1,
    )

    predictor.fit(
        time_limit=int(60 * 60 * 4),
        train_data=X_train,
        presets="best_quality",
        dynamic_stacking=False,
        hyperparameters=zeroshot2024,
        # Early Stopping
        ag_args_fit={
            "stopping_metric": "log_loss",
        },
        # Validation Protocol
        num_bag_folds=16,
        num_bag_sets=1,
        num_stack_levels=1,
    )
    predictor.fit_summary(verbosity=-1)
    y_pred = predictor.predict(X_test)

    rmse_test = mean_squared_error(y_test, y_pred) ** 0.5
    return rmse_test


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


def main():
    f = open("results.txt", "w")
    f.write("Test different versions of LGBM with OpenFE \n" + str(datetime.datetime.now()) +"\n\n")
    f.close()

    """
            If running on a SLURM cluster, we need to initialize Ray with extra options and a unique tempr dir.
            Otherwise, given the shared filesystem, Ray will try to use the same temp dir for all workers and crash (semi-randomly).
            """
    import os
    import logging
    import ray
    import uuid
    import base64
    import time
    log = logging.getLogger(__name__)
    ray_mem_in_gb = 32
    log.info(f"Running on SLURM, initializing Ray with unique temp dir with {ray_mem_in_gb}GB.")
    ray_mem_in_b = int(ray_mem_in_gb * (1024.0 ** 3))
    tmp_dir_base_path = "tmp_dir_base_path"
    uuid_short = base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b'=').decode('ascii')
    ray_dir = f"{tmp_dir_base_path}/{uuid_short}/ray"
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
    )

    dataset_ids = [190411, 359983, 189354, 189356, 10090, 359979, 146818, 359955, 359960, 359968, 359959, 168757, 359954, 359969, 359970, 359984, 168911, 359981, 359962, 359965, 190392, 190137, 359958, 168350, 359956, 359975, 359963, 168784, 190146, 146820, 359974, 2073, 359944, 359950, 359942, 359951, 360945, 167210, 359930, 359948, 359931, 359932, 359933, 359934, 359939, 359945, 359935, 359940]
    for dataset_id in dataset_ids:
        f = open("results.txt", "a")
        f.write("Dataset: " + str(dataset_id) + "\n")
        f.close()

        X_train, y_train, X_test, y_test = get_openml_dataset(dataset_id)
        X_train, y_train, X_test, y_test = factorize_data(X_train, y_train, X_test, y_test)
        X_train_openfe, y_train_openfe, X_test_openfe, y_test_openfe = get_openfe_data(X_train, y_train, X_test, y_test)
        try:
            lgbm_results = run_lgbm(X_train, y_train, X_test, y_test)
            f = open("results.txt", "a")
            f.write("LGBM Results " + str(lgbm_results) + "\n")
            f.close()
        except Exception as e:
            f = open("results.txt", "a")
            f.write("LGBM Results " + str(e) + "\n")
            f.close()
        try:
            lgbm_openfe_results = run_lgbm(X_train_openfe, y_train_openfe, X_test_openfe, y_test_openfe)
            f = open("results.txt", "a")
            f.write("LGBM OpenFE Results " + str(lgbm_openfe_results) + "\n")
            f.close()
        except Exception as e:
            f = open("results.txt", "a")
            f.write("LGBM OpenFE Results " + str(e) + "\n")
            f.close()
        try:
            autogluon_lgbm_results = run_autogluon_lgbm(X_train, y_train, X_test, y_test)
            f = open("results.txt", "a")
            f.write("Autogluon LGBM Results " + str(autogluon_lgbm_results) + "\n")
            f.close()
        except Exception as e:
            f = open("results.txt", "a")
            f.write("Autogluon LGBM Results " + str(e) + "\n")
            f.close()
        try:
            autogluon_lgbm_openfe_results = run_autogluon_lgbm(X_train_openfe, y_train_openfe, X_test_openfe, y_test_openfe)
            f = open("results.txt", "a")
            f.write("Autogluon LGBM OpenFE Results " + str(autogluon_lgbm_openfe_results) + "\n")
            f.close()
        except Exception as e:
            f = open("results.txt", "a")
            f.write("Autogluon LGBM OpenFE Results " + str(e) + "\n")
            f.close()
        try:
            tuned_autogluon_lgbm_results = run_tuned_autogluon_lgbm(X_train, y_train, X_test, y_test)
            f = open("results.txt", "a")
            f.write("Tuned Autogluon LGBM Results " + str(tuned_autogluon_lgbm_results) + "\n")
            f.close()
        except Exception as e:
            f = open("results.txt", "a")
            f.write("Tuned Autogluon LGBM Results " + str(e) + "\n")
            f.close()
        try:
            tuned_autogluon_lgbm_openfe_results = run_tuned_autogluon_lgbm(X_train_openfe, y_train_openfe, X_test_openfe, y_test_openfe)
            f = open("results.txt", "a")
            f.write("Tuned Autogluon LGBM OpenFE Results " + str(tuned_autogluon_lgbm_openfe_results) + "\n")
            f.close()
        except Exception as e:
            f = open("results.txt", "a")
            f.write("Tuned Autogluon LGBM OpenFE Results " + str(e) + "\n")
            f.close()


if __name__ == '__main__':
    main()
    