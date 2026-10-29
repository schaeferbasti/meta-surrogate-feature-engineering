"""Step 1: core matrix = the label of every candidate feature on every benchmark dataset, per model.

Label of a candidate feature (and model) = relative improvement of the AutoGluon validation score (out-of-fold, on the
training rows) when the feature is added to (or, for "without - <col>", removed from) the original data.
With label="cv" (experiment lightgbm_cv) it is instead the mean over CV_REPEATS repeats of the mean over CV_FOLDS folds
of the paired relative loss decrease of plain LightGBM (same folds and seed for the original data and the candidate,
metafe.models.cv_losses); the candidates run in parallel (--workers, default: the job's CPUs) and the per-repeat labels
are kept in <label_repeats_dir>/<model>/<id>.parquet.

The experiment (metafe/config.py) sets which labels are computed:
  - max_candidates: only a sample of the candidates of a dataset is labelled, balanced over the operators and the same
    for every model (so all models label the same candidates);
  - max_label_rows: larger datasets are labelled on a stratified sample of their training rows (the same rows for
    every candidate, model and for the metafeatures, see label_data);
  - reuse_legacy_labels: an old LightGBM label (data/legacy/core_matrix, computed with the leaky first operators) is
    reused when the fixed operators give the same training feature, because a label only depends on the training
    rows. Delete labels are always reused;
  - compute_labels=False: no AutoGluon fits at all, candidates without a reusable label are left out.

Every model has its own file <core_matrix_dir>/<model key>/<id>.parquet, so the models of a dataset can run as
separate (CPU and GPU) jobs. Labels are written after every candidate and resumed per feature, so a killed job
continues where it stopped. A candidate on which the model fails gets a NaN label (not retried, dropped by `merge`);
a model that fails on the original data is skipped for the dataset (nothing written, retried in the next run).

    python -m metafe.core_matrix build --experiment lightgbm --dataset 146818 [--models GBM] [--dry_run]
    python -m metafe.core_matrix merge --experiment lightgbm
"""
import argparse
import glob
import multiprocessing
import os
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from metafe import config
from metafe.data import label_data, is_classification, encode_target, write_parquet, read_parquet_or_none
from metafe.legacy_operators import legacy_train_feature
from metafe.models import model_name, check_setup, fit_leaderboard, validation_scores, relative_improvement, \
    cv_folds, cv_losses
from metafe.operators import candidate_featurenames, parse_featurename, compute_feature, same_feature, DELETE

COLUMNS = ["dataset - id", "feature - name", "operator", "model", "improvement"]
LEGACY_MODEL = "LightGBM_BAG_L1"  # only model of the old core matrix


def sample_candidates(candidates, max_candidates, seed=0):
    """At most max_candidates of the candidates (in their original order), balanced over the operators: round robin
    over the operators, a random candidate of each operator per round."""
    if max_candidates is None or len(candidates) <= max_candidates:
        return list(candidates)
    rng = np.random.default_rng(seed)
    by_operator = defaultdict(list)
    for name in candidates:
        by_operator[parse_featurename(name)[0]].append(name)
    queues = [[str(n) for n in rng.permutation(names)] for _, names in sorted(by_operator.items())]
    sample = set()
    while len(sample) < max_candidates:
        for queue in queues:
            if queue and len(sample) < max_candidates:
                sample.add(queue.pop())
    return [name for name in candidates if name in sample]


def with_candidate(name, X_train, f_train=None):
    """Training data with the candidate feature added (or, for a delete candidate, the column removed)."""
    operator, operands = parse_featurename(name)
    if operator == DELETE:
        return X_train.drop(columns=operands[0])
    if f_train is None:
        f_train, _ = compute_feature(name, X_train)
    X = X_train.copy()
    X[name] = f_train.to_numpy()
    return X


def validation_score(X_train, y_train, classification, model_key):
    """|score_val| of one model, or None if it fails."""
    try:
        scores = validation_scores(fit_leaderboard(X_train, y_train, None, None, classification, [model_key]))
        return scores.get(model_name(model_key))
    except Exception as e:
        print(f"[Warning] AutoGluon failed: {e}")
        return None


def legacy_labels(dataset_id, legacy_dir=config.LEGACY_CORE_MATRIX_DIR):
    old = read_parquet_or_none(os.path.join(legacy_dir, f"Operator_Model_Feature_Matrix_Core{dataset_id}.parquet"))
    if old is None:
        return {}
    old = old[old["model"] == LEGACY_MODEL].dropna(subset=["improvement"])
    return old.groupby("feature - name")["improvement"].first().to_dict()


def reusable_legacy_label(name, X_train, f_train, old_labels):
    """The old label of the candidate if the fixed operators give the same training feature, else None."""
    if name not in old_labels:
        return None
    operator, operands = parse_featurename(name)
    try:
        if operator == DELETE or same_feature(legacy_train_feature(operator, *[X_train[c] for c in operands]), f_train):
            return old_labels[name]
    except Exception:
        pass
    return None


def build(dataset_id, model_key, experiment, dry_run=False):
    out_path = os.path.join(experiment.core_matrix_dir, model_key, f"{dataset_id}.parquet")
    model = model_name(model_key)
    X_train, y_train, metadata = label_data(dataset_id, experiment)
    classification = is_classification(metadata)
    reuse = experiment.reuse_legacy_labels and model == LEGACY_MODEL
    old_labels = legacy_labels(dataset_id) if reuse else {}
    result = read_parquet_or_none(out_path)
    if result is None:
        result = pd.DataFrame(columns=COLUMNS)
    done = set(result["feature - name"])
    all_candidates = candidate_featurenames(X_train)
    candidates = sample_candidates(all_candidates, experiment.max_candidates)
    print(f"Dataset {dataset_id} ({metadata['task_type']}, {len(X_train)} label rows), model {model}: "
          f"{len(candidates)} of {len(all_candidates)} candidates, {len(old_labels)} old labels, "
          f"{len(done)} labels already in {out_path}")

    original_score = None
    stats = {"reused": 0, "computed": 0, "failed": 0}
    not_reusable = Counter()  # per operator, compute_labels=False
    start = time.time()
    for name in candidates:
        if name in done:
            continue
        operator, _ = parse_featurename(name)
        f_train = None
        if operator != DELETE:
            try:
                f_train, _ = compute_feature(name, X_train)
            except Exception as e:
                print(f"[Warning] {name} cannot be computed ({e}), skipped")
                continue

        label = reusable_legacy_label(name, X_train, f_train, old_labels) if reuse else None
        if label is not None:
            stats["reused"] += 1
        elif not experiment.compute_labels:
            not_reusable[operator] += 1
            continue
        elif dry_run:
            stats["computed"] += 1
            continue
        else:
            if original_score is None:
                original_score = validation_score(X_train, y_train, classification, model_key)
                print(f"Original validation score: {original_score}")
                if original_score is None:
                    print(f"{model} fails on the original data of dataset {dataset_id}, skipped")
                    break
            fit_start = time.time()
            score = validation_score(with_candidate(name, X_train, f_train), y_train, classification, model_key)
            label = np.nan if score is None else relative_improvement(original_score, score)
            stats["failed" if score is None else "computed"] += 1
            print(f"{name}: {label} ({time.time() - fit_start:.0f} s)")

        row = pd.DataFrame([[metadata["task_id"], name, operator, model, label]], columns=COLUMNS)
        result = pd.concat([result, row], ignore_index=True) if len(result) else row
        if not dry_run:
            write_parquet(result, out_path)
    what = "dry run, 'computed' = labels to compute" if dry_run else "done"
    print(f"Dataset {dataset_id}, {model} {what} in {time.time() - start:.0f} s: {stats}")
    if not_reusable:
        print(f"Left out (no reusable label): {sum(not_reusable.values())}, per operator {dict(not_reusable)}")


_CV = {}  # data of the CV labels, shared with the worker processes (fork)


def _cv_label(name):
    """(name, per-repeat labels or None, error message) of one candidate."""
    try:
        X = with_candidate(name, _CV["X"])
        if X.shape[1] == 0:
            return name, None, "no column left"
        gain = (_CV["base"] - cv_losses(X, _CV["y"], _CV["classification"], _CV["n_classes"], _CV["folds"])) \
            / _CV["base"]  # paired relative loss decrease per repeat and fold
        return name, gain.mean(1), None
    except Exception as e:
        return name, None, str(e)[:200]


def build_cv(dataset_id, model_key, experiment, workers):
    """Labels of all candidates with the repeated LightGBM CV (label="cv"), in parallel, resumable."""
    if model_key != "GBM":
        raise ValueError(f'label="cv" is a LightGBM label, not one of {model_key}')
    out_path = os.path.join(experiment.core_matrix_dir, model_key, f"{dataset_id}.parquet")
    repeats_path = os.path.join(experiment.label_repeats_dir, model_key, f"{dataset_id}.parquet")
    model = model_name(model_key)
    X_train, y_train, metadata = label_data(dataset_id, experiment)
    classification = is_classification(metadata)
    y, n_classes = encode_target(y_train) if classification else (pd.to_numeric(y_train).to_numpy(dtype="float64"), 0)
    result = read_parquet_or_none(out_path)
    repeats = read_parquet_or_none(repeats_path)
    if result is None or repeats is None:
        result, repeats = pd.DataFrame(columns=COLUMNS), pd.DataFrame()
    candidates = sample_candidates(candidate_featurenames(X_train), experiment.max_candidates)
    if X_train.shape[1] <= 1:
        candidates = [n for n in candidates if parse_featurename(n)[0] != DELETE]
    todo = [n for n in candidates if n not in set(result["feature - name"])]
    start = time.time()
    folds = cv_folds(y, classification, config.CV_REPEATS, config.CV_FOLDS)
    base = cv_losses(X_train, y, classification, n_classes, folds)
    _CV.update(X=X_train, y=y, classification=classification, n_classes=n_classes, folds=folds, base=base)
    print(f"Dataset {dataset_id} ({metadata['task_type']}, {len(X_train)} label rows), {model}, CV label: "
          f"{len(todo)} of {len(candidates)} candidates to do, {workers} workers; original loss {base.mean():.5f} "
          f"({time.time() - start:.0f} s)", flush=True)
    rows, repeat_rows, failed = [], [], 0

    def save():
        nonlocal result, repeats
        if rows:
            result = pd.concat([result, pd.DataFrame(rows, columns=COLUMNS)], ignore_index=True) if len(result) \
                else pd.DataFrame(rows, columns=COLUMNS)
            repeats = pd.concat([repeats, pd.DataFrame(repeat_rows)], ignore_index=True).drop_duplicates(
                "feature - name", keep="last")  # a job killed between the two writes leaves a repeat row behind
            write_parquet(repeats, repeats_path)
            write_parquet(result, out_path)
            rows.clear()
            repeat_rows.clear()

    with multiprocessing.get_context("fork").Pool(workers) as pool:
        for i, (name, per_repeat, error) in enumerate(pool.imap_unordered(_cv_label, todo), 1):
            if error is not None:
                failed += 1
                print(f"[Warning] {name}: {error}", flush=True)
            label = np.nan if per_repeat is None else float(per_repeat.mean())
            rows.append([metadata["task_id"], name, parse_featurename(name)[0], model, label])
            repeat_rows.append({"feature - name": name, "original_loss": float(base.mean()),
                                **{f"repeat_{r}": (np.nan if per_repeat is None else float(v))
                                   for r, v in enumerate(per_repeat if per_repeat is not None
                                                         else [np.nan] * config.CV_REPEATS)}})
            if i % 100 == 0 or i == len(todo):
                save()
                print(f"{i}/{len(todo)} labels ({time.time() - start:.0f} s)", flush=True)
    print(f"Dataset {dataset_id}, {model} done in {time.time() - start:.0f} s: {len(todo) - failed} computed, "
          f"{failed} failed")


def merge(experiment):
    files = sorted(glob.glob(os.path.join(experiment.core_matrix_dir, "*", "*.parquet")))
    matrix = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    n_rows = len(matrix)
    matrix = matrix.dropna(subset=["improvement"]).reset_index(drop=True)
    matrix["improvement"] = matrix["improvement"].astype("float64")
    write_parquet(matrix, experiment.core_matrix)
    print(f"{len(files)} files, {len(matrix)} labels ({n_rows - len(matrix)} without label dropped) "
          f"-> {experiment.core_matrix}")
    print(matrix.groupby("model")["dataset - id"].agg(labels="size", datasets="nunique").to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build", help="Compute the labels of one dataset")
    config.add_experiment_argument(p)
    p.add_argument("--dataset", type=int, required=True)
    p.add_argument("--models", nargs="+", default=None, help="AutoGluon model keys (default: the experiment's)")
    p.add_argument("--dry_run", action="store_true", help="No AutoGluon fits, only count labels to reuse / compute")
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)),
                   help='Parallel candidates with label="cv" (default: the job\'s CPUs, else 8)')
    p = sub.add_parser("merge", help="Merge the labels of all datasets and models")
    config.add_experiment_argument(p)
    args = parser.parse_args()
    experiment = config.EXPERIMENTS[args.experiment]
    if args.command == "build":
        model_keys = args.models or experiment.models
        if experiment.label == "cv":
            for model_key in model_keys:
                build_cv(args.dataset, model_key, experiment, args.workers)
            return
        if not args.dry_run:
            check_setup(experiment, model_keys)
        for model_key in model_keys:
            build(args.dataset, model_key, experiment, args.dry_run)
    else:
        merge(experiment)


if __name__ == "__main__":
    main()
