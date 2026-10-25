"""OpenML data loading and reading/writing of feature-engineered (FE) data."""
import os

import numpy as np
import openml
import pandas as pd

from metafe.config import TARGET


def load_task(task_id):
    """Original OpenML train/test split of a task: X_train, y_train, X_test, y_test, metadata.

    The row index (OpenML row number) is kept. Note: OpenML returns the training rows grouped by class, so nothing
    may depend on the row order (see metafe.operators).
    """
    task = openml.tasks.get_task(task_id, download_splits=True, download_data=True, download_qualities=True,
                                 download_features_meta_data=True)
    train_idx, test_idx = task.get_train_test_split_indices()
    X, y = task.get_X_and_y(dataset_format="dataframe")
    metadata = {"task_id": task.task_id, "task_type": task.task_type}
    return X.iloc[train_idx], y.iloc[train_idx], X.iloc[test_idx], y.iloc[test_idx], metadata


def label_data(dataset_id, experiment):
    """Training rows used for the labels and metafeatures: X_train, y_train, metadata (at most max_label_rows rows,
    stratified by class; same rows in every call)."""
    X_train, y_train, _, _, metadata = load_task(dataset_id)
    max_rows = experiment.max_label_rows
    if max_rows is not None and len(X_train) > max_rows:
        from sklearn.model_selection import train_test_split

        positions = np.arange(len(X_train))
        try:
            stratify = y_train.to_numpy() if is_classification(metadata) else None
            keep, _ = train_test_split(positions, train_size=max_rows, random_state=0, stratify=stratify)
        except ValueError:  # a class with a single row
            keep, _ = train_test_split(positions, train_size=max_rows, random_state=0)
        keep = np.sort(keep)
        X_train, y_train = X_train.iloc[keep], y_train.iloc[keep]
    return X_train, y_train, metadata


def is_classification(metadata):
    return metadata["task_type"] == "Supervised Classification"


def concat_data(X_train, y_train, X_test, y_test):
    """One frame with the training rows first, then the test rows (format of all FE files)."""
    train = pd.concat([X_train, y_train.rename(TARGET)], axis=1)
    test = pd.concat([X_test, y_test.rename(TARGET)], axis=1)
    return pd.concat([train, test], axis=0)


def split_like_original(data, X_train, X_test):
    """Split FE data back into the original OpenML train/test rows: X_train, y_train, X_test, y_test."""
    y = data[TARGET]
    X = data.drop(columns=TARGET)
    if len(X) != len(X_train) + len(X_test):
        raise ValueError(f"FE data has {len(X)} rows, but the split has {len(X_train)} train + {len(X_test)} test rows")
    if X.index.is_unique and X.index.isin(X_train.index.append(X_test.index)).all():
        test_mask = X.index.isin(X_test.index)  # original row labels kept (in any order)
    elif not X.index.is_unique:
        test_mask = np.arange(len(X)) >= len(X_train)  # index was reset by the FE method: rely on concat_data's order
    else:
        raise ValueError("Cannot map the FE data rows to the original train/test split")
    return X[~test_mask], y[~test_mask], X[test_mask], y[test_mask]


def write_parquet(data, path):
    """Write atomically (temporary file + rename), so that a killed job never leaves a truncated file behind."""
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data.to_parquet(path + ".tmp")
    os.replace(path + ".tmp", path)


def read_parquet_or_none(path):
    try:
        return pd.read_parquet(path)
    except (FileNotFoundError, OSError, ValueError):  # missing, or a Git LFS pointer that was not pulled
        return None
