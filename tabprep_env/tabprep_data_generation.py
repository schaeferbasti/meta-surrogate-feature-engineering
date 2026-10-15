import glob
import os
import re

import pandas as pd

from src.utils.get_data import split_data, get_tabprep_data, concat_data


def main():
    for raw_path in glob.glob("test_data/RAW_*.parquet"):
        dataset_id = re.match(r".*RAW_(\d+)\.parquet", raw_path).group(1)
        out_path = f"test_data/FE_{dataset_id}_TabPrep.parquet"
        if os.path.exists(out_path):
            continue
        X_train, y_train, X_test, y_test = split_data(pd.read_parquet(raw_path), "target")
        X_tr, y_tr, X_te, y_te = get_tabprep_data(X_train, y_train, X_test, y_test, task_type)
        X_tr, X_te = clean_columns(X_tr, X_te)
        concat_data(X_tr, y_tr, X_te, y_te, "target").to_parquet(out_path)

def clean_columns(X_train, X_test):
    mapping, seen = {}, {}
    for col in X_train.columns:
        new = re.sub('[^A-Za-z0-9_]+', '_', str(col))
        if new in seen:
            seen[new] += 1
            new = f"{new}_{seen[new]}"
        else:
            seen[new] = 0
        mapping[col] = new
    return X_train.rename(columns=mapping), X_test.rename(columns=mapping)


if __name__ == "__main__":
    TASK_TYPES = {
        1: "classification",
        2: "regression",
        3: "multilabel_classification",
        4: "learning_to_rank",
        5: "clustering",
        6: "survival_analysis",
    }
    main()