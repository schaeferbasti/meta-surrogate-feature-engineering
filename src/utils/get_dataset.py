import pandas as pd
import openml


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
    dataset_metadata = {"task_id": task.task_id, "task_type": task.task_type,
                        "number_of_classes": len(task.class_labels) if task.class_labels else 'N/A'}
    train_idx, test_idx = task.get_train_test_split_indices()
    X, y = task.get_X_and_y(dataset_format="dataframe")  # type: ignore
    train_x, train_y = X.iloc[train_idx], y.iloc[train_idx]
    test_x, test_y = X.iloc[test_idx], y.iloc[test_idx]
    return train_x, train_y, test_x, test_y, dataset_metadata


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
