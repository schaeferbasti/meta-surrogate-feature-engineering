"""Sanity checks of feature-engineered (FE) data before the evaluation.

For every FE file: the rows map onto the original OpenML train/test split, and for every generated column
  - FAIL if it has values on the training rows but none on the test rows (feature not applied to the test rows),
  - WARN if its missing-value rate differs by more than 20 percentage points between train and test rows,
  - WARN if it is strongly rank-correlated (|Spearman| > 0.5) with the training row position within the rows of
    the same target value (row-ID-like feature; OpenML training rows are grouped by class, so such a feature can
    leak the target; within a class, a predictive feature has no reason to follow the row order).

    python -m metafe.evaluation.check_fe --experiment lightgbm [--dataset 146818 ...] [--methods MetaFE_1000 ...]
"""
import argparse
import sys

import numpy as np
import pandas as pd

from metafe import config
from metafe.data import load_task, split_like_original
from metafe.evaluation.evaluate import fe_files


def row_position_correlation(values, y):
    # Mean Spearman correlation with the row position within groups of equal target value (or all rows, regression)
    values = pd.to_numeric(values, errors="coerce").reset_index(drop=True)
    y = y.reset_index(drop=True)
    groups = [np.arange(len(y))] if pd.api.types.is_float_dtype(y) else list(y.groupby(y, observed=True).indices.values())
    correlations, weights = [], []
    for idx in groups:
        group = values.iloc[idx]
        if group.notna().sum() > 10 and group.nunique() > 1:
            correlations.append(group.corr(pd.Series(np.arange(len(idx)), index=group.index), method="spearman"))
            weights.append(len(idx))
    return np.average(correlations, weights=weights) if correlations else 0.0


def check_file(path, X_train, X_test):
    problems = []
    try:
        X_tr, y_tr, X_te, _ = split_like_original(pd.read_parquet(path), X_train, X_test)
    except ValueError as e:
        return [("FAIL", f"rows do not match the split: {e}")]
    for column in [c for c in X_tr.columns if c not in X_train.columns]:
        train_nan, test_nan = X_tr[column].isna().mean(), X_te[column].isna().mean()
        if train_nan < 1 and test_nan == 1:
            problems.append(("FAIL", f"{column}: no values on the test rows"))
        elif abs(train_nan - test_nan) > 0.2:
            problems.append(("WARN", f"{column}: missing values train {train_nan:.0%}, test {test_nan:.0%}"))
        correlation = row_position_correlation(X_tr[column], y_tr)
        if abs(correlation) > 0.5:
            problems.append(("WARN", f"{column}: Spearman correlation {correlation:.2f} with the row position"))
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_experiment_argument(parser)
    parser.add_argument("--dataset", type=int, nargs="+", default=sorted(config.DATASETS))
    parser.add_argument("--methods", nargs="+", default=None, help="Default: all FE data")
    args = parser.parse_args()
    experiment = config.EXPERIMENTS[args.experiment]
    n_files = n_fail = 0
    for dataset_id in args.dataset:
        files = {}  # (method, model) -> path; the baselines once
        for model_key in experiment.models:
            for method, path in fe_files(dataset_id, experiment, model_key).items():
                if args.methods is None or method in args.methods:
                    files[(method, None if method in config.BASELINES else model_key)] = path
        if not files:
            continue
        X_train, _, X_test, _, _ = load_task(dataset_id)
        for (method, model_key), path in files.items():
            n_files += 1
            problems = check_file(path, X_train, X_test)
            n_fail += any(level == "FAIL" for level, _ in problems)
            for level, message in problems:
                print(f"{level} {dataset_id} {method}{f' ({model_key})' if model_key else ''}: {message}")
    print(f"{n_files} files checked, {n_fail} with failures")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
