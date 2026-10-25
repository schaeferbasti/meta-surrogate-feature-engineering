import contextlib
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def get_all_amlb_dataset_ids():
    # Code from https://github.com/openml/automlbenchmark/blob/2c2a93dc3fc65fc3d6a77fe97ec9df1108551075/scripts/find_matching_datasets.py
    # small_config_url = "https://raw.githubusercontent.com/openml/automlbenchmark/2c2a93dc3fc65fc3d6a77fe97ec9df1108551075/resources/benchmarks/small.yaml"
    #medium_config_url = "https://raw.githubusercontent.com/openml/automlbenchmark/2c2a93dc3fc65fc3d6a77fe97ec9df1108551075/resources/benchmarks/medium.yaml"
    #large_config_url = "https://raw.githubusercontent.com/openml/automlbenchmark/2c2a93dc3fc65fc3d6a77fe97ec9df1108551075/resources/benchmarks/large.yaml"
    # small_configuration = yaml.load(requests.get(small_config_url).text, Loader=yaml.Loader)
    #medium_configuration = yaml.load(requests.get(medium_config_url).text, Loader=yaml.Loader)
    #large_configuration = yaml.load(requests.get(large_config_url).text, Loader=yaml.Loader)
    # benchmark_tids = set(
        # [problem.get("openml_task_id") for problem in small_configuration]
        # + [problem.get("openml_task_id") for problem in medium_configuration]
        # + [problem.get("openml_task_id") for problem in large_configuration]
    # )
    benchmark_tids = [146818, 146820, 168350, 168911, 190137, 190411, 359955, 359956, 359979]
    return benchmark_tids


def get_openml_dataset_split_and_metadata(openml_task_id: int) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict
]:
    task = openml.tasks.get_task(
        openml_task_id,
        download_splits=True,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    dataset_metadata = {"task_id": task.task_id, "task_type": task.task_type, "number_of_classes": 'N/A'}
    train_idx, test_idx = task.get_train_test_split_indices()
    X, y = task.get_X_and_y(dataset_format="dataframe")  # type: ignore
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    return X_train, y_train, X_test, y_test, dataset_metadata


def get_openml_dataset_split(openml_task_id: int) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame
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
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    return X_train, y_train, X_test, y_test


def get_openml_dataset_and_metadata(openml_task_id: int) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict
]:
    task = openml.tasks.get_task(
        openml_task_id,
        download_splits=True,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    dataset_metadata = {"task_id": task.task_id, "task_type": task.task_type, "number_of_classes": len(task.class_labels) if task.class_labels else 'N/A'}
    X, y = task.get_X_and_y(dataset_format="dataframe")  # type: ignore
    return X, y, dataset_metadata


def get_openml_dataset(openml_task_id: int) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    task = openml.tasks.get_task(
        openml_task_id,
        download_splits=True,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    X, y = task.get_X_and_y(dataset_format="dataframe")  # type: ignore
    return X, y


def split_data(data, target_label) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame
]:
    y = data[target_label]
    X = data.drop(target_label, axis=1)
    train_idx, test_idx, y_train, y_test = train_test_split(X.index, y, test_size=0.2)
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    return X_train, y_train, X_test, y_test


def split_data_original(data, target_label, X_train, X_test) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame
]:
    # Split feature-engineered data (written with concat_data: train rows first, then test rows) back into the
    # original OpenML train/test split, so that all methods are evaluated on the same test rows as "Original"
    y = data[target_label]
    X = data.drop(target_label, axis=1)
    n_train, n_test = len(X_train), len(X_test)
    if len(X) != n_train + n_test:
        raise ValueError(f"FE data has {len(X)} rows, but the original split has {n_train} train + {n_test} test rows")
    if X.index.is_unique and X.index.isin(X_train.index.append(X_test.index)).all():
        # Original row labels are kept (in any row order, e.g. Random_Model.py re-shuffles the rows)
        test_mask = X.index.isin(X_test.index)
    elif not X.index.is_unique:
        # Index of train and test part was reset by the FE method; rely on the row order of concat_data
        test_mask = np.arange(len(X)) >= n_train
    else:
        raise ValueError("Cannot map FE data rows to the original train/test split")
    return X[~test_mask], y[~test_mask], X[test_mask], y[test_mask]


def concat_data(X_train, y_train, X_test, y_test, target_label):
    y_train = y_train.to_frame(target_label)
    train_data = pd.concat([X_train, y_train], axis=1)
    y_test = y_test.to_frame(target_label)
    test_data = pd.concat([X_test, y_test], axis=1)
    data = pd.concat([train_data, test_data], axis=0)
    return data


def get_openfe_data(X_train, y_train, X_test, y_test, n_jobs=1):
    # Imported here so that environments without OpenFE (e.g. the TabPrep env) can still use this module
    from openfe import OpenFE, transform
    openFE = OpenFE()
    # OpenFE writes ./openfe_tmp_data.feather (transform() ignores tmp_save_path), which parallel jobs in the same
    # directory overwrite, so run it inside a private temporary working directory
    tmp_dir = tempfile.mkdtemp()
    try:
        with contextlib.chdir(tmp_dir):
            features = openFE.fit(data=X_train, label=y_train, n_jobs=n_jobs,
                                  tmp_save_path=os.path.join(tmp_dir, "openfe_tmp_data.feather"))  # generate new features
            X_train_openfe, X_test_openfe = transform(X_train, X_test, features, n_jobs=n_jobs)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return X_train_openfe, y_train, X_test_openfe, y_test

def get_tabprep_data(X_train, y_train, X_test, y_test, task_type, random_state=0):
    # Imported here: TabPrep needs AutoGluon from source (master), which differs from the pinned evaluation env
    from tabprep import TabPrepFeatureGenerator
    # TabPrep needs the problem type: "binary", "multiclass" or "regression"
    if task_type == "Supervised Classification":
        target_type = "binary" if y_train.nunique() == 2 else "multiclass"
        # Target encoding needs numeric labels; only used for fitting TabPrep,
        # the original labels are returned unchanged for your downstream models
        y_fit = pd.Series(LabelEncoder().fit_transform(y_train), index=y_train.index)
    else:
        target_type = "regression"
        y_fit = y_train.astype(float)

    tabprep = TabPrepFeatureGenerator(target_type=target_type, random_state=random_state)
    X_train_tabprep = tabprep.fit_transform(X_train, y_fit)  # fit on train only
    X_test_tabprep = tabprep.transform(X_test)               # apply to test, no refit

    return X_train_tabprep, y_train, X_test_tabprep, y_test

def get_name_and_split_and_save_dataset(openml_task_id):
    name = str(openml_task_id)
    task = openml.tasks.get_task(
        openml_task_id,
        download_splits=True,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    train_idx, test_idx = task.get_train_test_split_indices()
    X, y = task.get_X_and_y(dataset_format="dataframe")
    root_dir = "src/Metadata/d2v/dataset2vec/datasets/" + name + "/"
    try:
        os.makedirs(root_dir)
    except FileExistsError:
        pass
    len = X.shape[0]
    folds, validation_folds = get_folds_and_validation_folds(len)
    X.to_csv(root_dir + name + '_py.dat', header=False, index=False)
    folds.to_csv(root_dir + "/" + 'folds_py.dat',  header=False, index=False)
    y.to_csv(root_dir + "/" + 'labels_py.dat', header=False, index=False)
    validation_folds.to_csv(root_dir + "/" + 'validation_folds_py.dat',  header=False, index=False)
    return name, 0


def get_folds_and_validation_folds(len):
    folds = pd.DataFrame()
    for i in range(len):
        row = pd.DataFrame([np.random.choice([1, 0], size=4)])
        folds = pd.concat([folds, row], ignore_index=True)
    validation_folds = pd.DataFrame()
    for i in range(len):
        row = pd.DataFrame([np.random.choice([1, 0], size=4)])
        validation_folds = pd.concat([validation_folds, row], ignore_index=True)
    return folds, validation_folds
