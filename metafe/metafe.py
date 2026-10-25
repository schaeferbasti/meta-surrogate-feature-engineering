"""MetaFE: recursive feature engineering with the surrogate model.

Repeat until the time limit or until no candidate is predicted to improve the model by at least --min_improvement:
  1. describe every candidate feature of the current data (all operators on all current columns, and deleting each
     column) with its metafeatures,
  2. predict each candidate's improvement with the surrogate (trained on all other benchmark datasets),
  3. add (or delete) the best candidate in the training AND test rows, fitted on the training rows only.

A feature that was added or deleted once is never a candidate again, only original columns can be deleted, and a
candidate whose training values are a copy of an existing column is skipped
(deleting a generated feature only undoes a step; without these rules MetaFE can add, delete and re-add the same
feature until the time is up).

One run serves all time limits: after every step, the current data is written for every time limit that has not
passed yet, so data/<experiment>/fe/MetaFE_<model>/FE_<id>_MetaFE_<t>.parquet is the data MetaFE (targeting the
model) would return with a time limit of t seconds.
Time is counted from the end of the surrogate training (in practice the surrogate is trained once, beforehand; its
training time is logged as step "surrogate"); with --count_surrogate_training it is counted from the start of the run.
The run is killed at the largest time limit (plus the surrogate training time) or when it exceeds --memory_limit_mb.
Steps are logged in FE_<id>_MetaFE_log.csv.

The surrogate is trained on the labels of all models of the experiment and predicts the improvement for --model.

    python -m metafe.metafe --experiment lightgbm --dataset 146818 [--time_limits 100 250 ... 7200]
    python -m metafe.metafe --experiment multimodel --dataset 146818 --model TABICL
"""
import argparse
import csv
import multiprocessing
import os
import time

import pandas as pd
import psutil

from metafe import config
from metafe.data import load_task, concat_data, write_parquet
from metafe.models import model_name, check_setup
from metafe.operators import apply_features, compute_feature, same_feature
from metafe.surrogate import Surrogate


def sanitize_column_name(name):
    # Feature names are parsed by their parentheses and commas, so original columns must not contain them
    return str(name).strip().replace(",", "_").replace("(", "_").replace(")", "_")


def output_path(out_dir, dataset_id, time_limit):
    return os.path.join(out_dir, f"FE_{dataset_id}_MetaFE_{time_limit}.parquet")


def next_best_candidate(candidates, X_train, min_improvement, excluded):
    """(name, predicted improvement) of the best candidate with a prediction >= min_improvement whose training values
    are not a copy of an existing column (copies have the parent's metafeatures, so the surrogate likes them as much);
    copies found on the way are added to excluded. None if there is no such candidate."""
    for idx in candidates["predicted"].sort_values(ascending=False).index:
        name, predicted = candidates.at[idx, "feature - name"], candidates.at[idx, "predicted"]
        if predicted < min_improvement:
            return None
        if name.startswith("without - "):
            return name, predicted
        f_train, _ = compute_feature(name, X_train)
        if any(same_feature(f_train, X_train[c]) for c in X_train.columns):
            excluded.append(name)
            continue
        return name, predicted
    return None


def run(dataset_id, model_key, experiment, time_limits, min_improvement=config.MIN_IMPROVEMENT,
        count_surrogate_training=False, clock_start=None):
    start = time.time()
    elapsed = lambda: time.time() - start
    X_train, y_train, X_test, y_test, metadata = load_task(dataset_id)
    X_train = X_train.rename(columns=sanitize_column_name)
    X_test = X_test.rename(columns=sanitize_column_name)
    out_dir = experiment.metafe_dir(model_key)
    model = model_name(model_key)
    log_path = os.path.join(out_dir, f"FE_{dataset_id}_MetaFE_log.csv")
    os.makedirs(out_dir, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "feature", "predicted_improvement", "elapsed_s"])

    def save(step, feature=None, predicted=None):
        t = elapsed()
        data = concat_data(X_train, y_train, X_test, y_test)
        for time_limit in time_limits:
            if time_limit >= t:  # this time limit has not passed yet, so its result is the current data
                write_parquet(data, output_path(out_dir, dataset_id, time_limit))
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([step, feature, predicted, round(t, 2)])
        print(f"[{t:.0f} s] step {step}: {feature or 'original data'} (predicted improvement {predicted})")

    surrogate = Surrogate(pd.read_parquet(experiment.metafeature_matrix), exclude_dataset=dataset_id)
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow(["surrogate", None, None, round(elapsed(), 2)])
    if not count_surrogate_training:
        start = time.time()
        if clock_start is not None:
            clock_start.value = start  # the time limit of run_with_limits starts now as well
    save(0)
    step, excluded = 0, []
    while True:
        candidates = surrogate.candidates(X_train, dataset_id, metadata["task_type"], model, exclude=excluded)
        if candidates.empty:
            break
        candidates["predicted"] = surrogate.predict(candidates)
        best = next_best_candidate(candidates, X_train, min_improvement, excluded)
        if best is None:
            print(f"No (new) candidate with predicted improvement >= {min_improvement}, stop")
            break
        name, predicted = best
        X_train, X_test = apply_features(X_train, X_test, [name])
        excluded += [name] if name.startswith("without - ") else [name, f"without - {name}"]
        step += 1
        save(step, name, round(float(predicted), 6))
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow(["end", None, None, round(elapsed(), 2)])
    print(f"MetaFE finished after {elapsed():.0f} s and {step} steps")


def run_with_limits(target, time_limit_s, memory_limit_mb, clock_start, check_interval_s=5):
    """Run target() in a child process; kill it time_limit_s after clock_start.value (a shared value that the child
    sets when its clock starts; inf = not started yet) or above memory_limit_mb (RSS). Returns the exit code."""
    process = multiprocessing.get_context("fork").Process(target=target)  # fork: target may be a lambda
    process.start()
    while process.is_alive():
        try:
            memory_mb = psutil.Process(process.pid).memory_info().rss / 2 ** 20
        except psutil.NoSuchProcess:
            break
        if time.time() - clock_start.value > time_limit_s:
            print(f"[Monitor] Time limit of {time_limit_s} s reached, stop")
            process.terminate()
        elif memory_mb > memory_limit_mb:
            print(f"[Monitor] Memory {memory_mb:.0f} MB > {memory_limit_mb} MB, stop")
            process.terminate()
        time.sleep(check_interval_s)
    process.join()
    return process.exitcode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_experiment_argument(parser)
    parser.add_argument("--dataset", type=int, required=True)
    parser.add_argument("--model", default=None,
                        help="Model key whose improvement is predicted (needed if the experiment has several models)")
    parser.add_argument("--time_limits", type=int, nargs="+", default=None, help="Default: the experiment's")
    parser.add_argument("--min_improvement", type=float, default=None, help="Default: the experiment's")
    parser.add_argument("--memory_limit_mb", type=int, default=64000)
    parser.add_argument("--count_surrogate_training", action="store_true",
                        help="Count the surrogate training time in the time limits")
    args = parser.parse_args()
    experiment = config.EXPERIMENTS[args.experiment]
    if args.model is None and len(experiment.models) > 1:
        parser.error(f"--model is required for experiment {experiment.name} (one of {', '.join(experiment.models)})")
    model_key = args.model or experiment.models[0]
    if model_key not in experiment.models:
        parser.error(f"Model {model_key} is not in experiment {experiment.name} ({', '.join(experiment.models)})")
    check_setup(experiment, [model_key])  # the surrogate is an AutoGluon CatBoost model
    time_limits = sorted(args.time_limits or experiment.time_limits)
    # The clock starts now, or (default) when the child has trained the surrogate
    clock_start = multiprocessing.get_context("fork").Value("d", time.time() if args.count_surrogate_training else float("inf"))
    exit_code = run_with_limits(
        lambda: run(args.dataset, model_key, experiment, time_limits,
                    experiment.min_improvement if args.min_improvement is None else args.min_improvement,
                    args.count_surrogate_training, clock_start),
        time_limit_s=max(time_limits), memory_limit_mb=args.memory_limit_mb, clock_start=clock_start)
    # Killed at the time limit is a normal end (all files are written); any other failure is an error
    out_dir = experiment.metafe_dir(model_key)
    missing = [t for t in time_limits if not os.path.exists(output_path(out_dir, args.dataset, t))]
    if missing:
        raise SystemExit(f"MetaFE failed (exit code {exit_code}), no data for time limits {missing}")


if __name__ == "__main__":
    main()
