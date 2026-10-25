"""Evaluate the original data and every feature-engineered (FE) dataset with the models of an experiment.

All methods use the original OpenML train/test split. A "fold" is a repeat on that split with AutoGluon seed = fold
(bagging fold split and model seed), so the folds show the variance of the model, not of the split.

Methods: "Original" (the OpenML data), the baselines data/fe/<method>/FE_<id>_<method>.parquet (OpenFE, TabPrep) and
MetaFE, data/<experiment>/fe/MetaFE_<model>/FE_<id>_MetaFE_<t>.parquet. Original and the baselines are evaluated
with every model; the MetaFE data for a model only with that model (MetaFE adapts the features to it).

Each result is cached as results/<experiment>/evaluation/<model>/<method>_Result_<id>_<fold>.parquet (delete it to
recompute). MetaFE files that are identical for several time limits are fitted once. An experiment with
baseline_results_from copies the cached Original/OpenFE/TabPrep results of that experiment (same models and setup).

    python -m metafe.evaluation.evaluate --experiment lightgbm --dataset 146818 [--fold 0] [--model GBM]
"""
import argparse
import hashlib
import os
import re

import pandas as pd

from metafe import config
from metafe.data import load_task, is_classification, split_like_original, write_parquet, read_parquet_or_none
from metafe.models import model_name, check_setup, fit_leaderboard

RESULT_COLUMNS = ["origin", "task_type", "dataset", "model", "score_val", "score_test"]


def fe_files(dataset_id, experiment, model_key):
    """{method: path} of the FE data of a dataset that is evaluated with the model: baselines and MetaFE_<model>."""
    files = {}
    for method in config.BASELINES:
        path = os.path.join(config.BASELINE_FE_DIR, method, f"FE_{dataset_id}_{method}.parquet")
        if os.path.exists(path):
            files[method] = path
    metafe_dir = experiment.metafe_dir(model_key)
    if os.path.isdir(metafe_dir):
        pattern = re.compile(rf"FE_{dataset_id}_(MetaFE_\d+)\.parquet")
        for name in sorted(os.listdir(metafe_dir)):
            match = pattern.fullmatch(name)
            if match:
                files[match.group(1)] = os.path.join(metafe_dir, name)
    return files


def result_path(experiment, model_key, method, dataset_id, fold):
    return os.path.join(experiment.evaluation_dir, model_key, f"{method}_Result_{dataset_id}_{fold}.parquet")


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def clean_column_names(X):
    # LightGBM does not accept special characters in column names; duplicates get a suffix
    names, seen = [], {}
    for column in X.columns:
        name = re.sub("[^A-Za-z0-9_]+", "", str(column))
        seen[name] = seen.get(name, -1) + 1
        names.append(name if seen[name] == 0 else f"{name}_{seen[name]}")
    return X.set_axis(names, axis=1)


def score(X_train, y_train, X_test, y_test, metadata, dataset_id, method, fold, model_key):
    """Result rows (one, or none if the model fails) of one model on one dataset."""
    classification = is_classification(metadata)
    try:
        lb = fit_leaderboard(X_train, y_train, X_test, y_test, classification, [model_key], seed=fold)
    except Exception as e:
        print(f"[Warning] AutoGluon failed: {e}")
        lb = pd.DataFrame(columns=["model", "score_val", "score_test"])
    lb = lb[lb["model"] == model_name(model_key)]
    task_type = "Classification" if classification else "Regression"
    return pd.DataFrame([[method, task_type, dataset_id, m, v, t] for m, v, t in
                         zip(lb["model"], lb["score_val"], lb["score_test"])], columns=RESULT_COLUMNS)


def evaluate(dataset_id, fold, model_key, experiment, methods=None):
    X_train, y_train, X_test, y_test, metadata = load_task(dataset_id)
    files = fe_files(dataset_id, experiment, model_key)
    all_methods = ["Original"] + list(files)
    by_hash = {}  # file hash -> result of an identical file evaluated in this run
    for method in methods or all_methods:
        if method not in all_methods:
            print(f"[Warning] No FE data for {method} on dataset {dataset_id}")
            continue
        path = result_path(experiment, model_key, method, dataset_id, fold)
        result = read_parquet_or_none(path)
        if result is None and experiment.baseline_results_from and not method.startswith("MetaFE"):
            other = config.EXPERIMENTS[experiment.baseline_results_from]
            result = read_parquet_or_none(result_path(other, model_key, method, dataset_id, fold))
            if result is not None:
                write_parquet(result, path)  # same setup: copy the cached result of the other experiment
        if result is None:
            digest = file_hash(files[method]) if method != "Original" else None
            if digest in by_hash:
                result = by_hash[digest].assign(origin=method)
                print(f"{method}: same data as {by_hash[digest]['origin'].iloc[0]}")
            elif method == "Original":
                result = score(X_train, y_train, X_test, y_test, metadata, dataset_id, method, fold, model_key)
            else:
                try:
                    X_tr, y_tr, X_te, y_te = split_like_original(pd.read_parquet(files[method]), X_train, X_test)
                except ValueError as e:
                    print(f"[Warning] {method} skipped: {e}")
                    continue
                result = score(clean_column_names(X_tr), y_tr, clean_column_names(X_te), y_te, metadata, dataset_id,
                               method, fold, model_key)
            if result.empty:
                print(f"{method}: {model_name(model_key)} failed, nothing written (retried in the next run)")
                continue
            write_parquet(result, path)
            if digest is not None:
                by_hash.setdefault(digest, result)
        elif method != "Original":
            by_hash.setdefault(file_hash(files[method]), result)
        print(f"{method}: val {result['score_val'].iloc[0]:.4f}, test {result['score_test'].iloc[0]:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_experiment_argument(parser)
    parser.add_argument("--dataset", type=int, nargs="+", default=sorted(config.DATASETS))
    parser.add_argument("--fold", type=int, nargs="+", default=config.FOLDS)
    parser.add_argument("--model", nargs="+", default=None, help="Model keys (default: the experiment's)")
    parser.add_argument("--methods", nargs="+", default=None, help="Default: Original and all FE data found")
    args = parser.parse_args()
    experiment = config.EXPERIMENTS[args.experiment]
    model_keys = args.model or experiment.models
    check_setup(experiment, model_keys)
    for model_key in model_keys:
        for dataset_id in args.dataset:
            for fold in args.fold:
                print(f"=== {model_name(model_key)}, dataset {dataset_id}, fold {fold}")
                evaluate(dataset_id, fold, model_key, experiment, args.methods)


if __name__ == "__main__":
    main()
