
"""Step 2: metafeature matrix = core matrix + pandas metafeatures of every candidate feature (surrogate training data).

The metafeatures of a candidate are pandas' describe() of the candidate feature on the training rows (of the removed
column for "without - <col>"), computed with the same operators as in MetaFE (metafe.operators) and on the same
training rows as the labels (metafe.data.label_data). The same function describes the candidates of a new
dataset in MetaFE, so training and prediction rows always match.

    python -m metafe.metafeatures build --experiment lightgbm --dataset 146818
    python -m metafe.metafeatures merge --experiment lightgbm
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd

from metafe import config
from metafe.data import label_data, write_parquet
from metafe.operators import parse_featurename, compute_feature, DELETE

METAFEATURE_COLUMNS = ["feature - count", "feature - unique", "feature - top", "feature - freq", "feature - mean",
                       "feature - std", "feature - min", "feature - 25%", "feature - 50%", "feature - 75%",
                       "feature - max"]
_DESCRIBE_ROWS = ["count", "unique", "top", "freq", "mean", "std", "min", "25%", "50%", "75%", "max"]


def describe_feature(feature):
    """The 11 pandas metafeatures of one feature (NaN where describe() has no such value, e.g. mean of strings)."""
    feature = pd.Series(feature).reset_index(drop=True).infer_objects()
    description = feature.to_frame().describe(include="all").iloc[:, 0]
    values = [description.get(row, np.nan) for row in _DESCRIBE_ROWS]
    values[2] = None if pd.isna(values[2]) else str(values[2])  # "top" as string, so that parquet has one type
    return values


def add_metafeatures(matrix, X_train, task_type):
    """Insert task_type and the metafeatures of each row's feature after the "operator" column.

    matrix needs the columns "feature - name" and "operator"; the operands of every feature must be columns of X_train.
    Each feature is computed once, also if it appears in several rows (one row per model).
    """
    described = {}
    for name in pd.unique(matrix["feature - name"]):
        operator, operands = parse_featurename(name)
        try:
            if operator == DELETE:
                feature = X_train[operands[0]]
            else:
                feature, _ = compute_feature(name, X_train)
            described[name] = describe_feature(feature)
        except Exception as e:
            print(f"[Warning] No metafeatures for {name}: {e}")
            described[name] = [np.nan] * len(METAFEATURE_COLUMNS)
    metafeatures = pd.DataFrame([described[n] for n in matrix["feature - name"]], index=matrix.index,
                                columns=METAFEATURE_COLUMNS)
    position = list(matrix.columns).index("operator") + 1
    return pd.concat([matrix.iloc[:, :position], pd.Series(task_type, index=matrix.index, name="task_type"),
                      metafeatures, matrix.iloc[:, position:]], axis=1)


def build(dataset_id, experiment):
    files = sorted(glob.glob(os.path.join(experiment.core_matrix_dir, "*", f"{dataset_id}.parquet")))
    if not files:
        raise FileNotFoundError(f"No labels of dataset {dataset_id} in {experiment.core_matrix_dir}")
    core = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).dropna(subset=["improvement"])
    X_train, _, metadata = label_data(dataset_id, experiment)  # the rows the labels were computed on
    start = time.time()
    matrix = add_metafeatures(core.reset_index(drop=True), X_train, metadata["task_type"])
    out_path = os.path.join(experiment.metafeature_matrix_dir, f"{dataset_id}.parquet")
    write_parquet(matrix, out_path)
    print(f"Dataset {dataset_id}: {len(matrix)} rows ({len(files)} models) in {time.time() - start:.0f} s -> {out_path}")


def merge(experiment):
    in_dir = experiment.metafeature_matrix_dir
    files = sorted(f for f in os.listdir(in_dir) if f.endswith(".parquet"))
    matrix = pd.concat([pd.read_parquet(os.path.join(in_dir, f)) for f in files], ignore_index=True)
    write_parquet(matrix, experiment.metafeature_matrix)
    print(f"{len(files)} datasets, {len(matrix)} rows -> {experiment.metafeature_matrix}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build", help="Metafeatures of the core matrix rows of one dataset (all models)")
    config.add_experiment_argument(p)
    p.add_argument("--dataset", type=int, required=True)
    p = sub.add_parser("merge", help="Merge the metafeature matrices of all datasets")
    config.add_experiment_argument(p)
    args = parser.parse_args()
    experiment = config.EXPERIMENTS[args.experiment]
    if args.command == "build":
        build(args.dataset, experiment)
    else:
        merge(experiment)


if __name__ == "__main__":
    main()
