"""Timing pilot (before the multimodel core matrix): how long does one label take per model and dataset size?

`run` fits the original data and --n_candidates candidates (balanced over the operators) of one dataset with one
model, exactly as the core matrix does (same label rows, same AutoGluon setup), and writes the time of every fit to
results/<experiment>/pilot/<model>_<id>.csv. `summary` fits log(time per fit) ~ log(label rows x columns) per model
and extrapolates the core matrix (max_candidates labels per dataset) and the evaluation to all datasets.

    python -m metafe.pilot run --experiment multimodel --dataset 168757 --model TABICL [--n_candidates 10]
    python -m metafe.pilot summary --experiment multimodel
"""
import argparse
import glob
import os
import platform
import time

import numpy as np
import pandas as pd

from metafe import config
from metafe.core_matrix import sample_candidates, with_candidate
from metafe.data import label_data, is_classification
from metafe.models import model_name, check_setup, fit_leaderboard, validation_scores, relative_improvement, num_gpus
from metafe.operators import candidate_featurenames

EVALUATION_FITS_PER_FOLD = 6  # Original, OpenFE, TabPrep and ~3 distinct MetaFE files (time limits) per dataset


def device(model_key):
    if num_gpus([model_key]):
        import torch
        return torch.cuda.get_device_name(0)
    return f"{config.NUM_CPUS} CPUs ({platform.processor() or platform.machine()})"


def run(dataset_id, model_key, experiment, n_candidates):
    X_train, y_train, metadata = label_data(dataset_id, experiment)
    classification = is_classification(metadata)
    candidates = sample_candidates(candidate_featurenames(X_train), n_candidates, seed=1)
    out_path = os.path.join(experiment.pilot_dir, f"{model_key}_{dataset_id}.csv")
    os.makedirs(experiment.pilot_dir, exist_ok=True)
    rows, original = [], None
    for name in [None] + candidates:
        X = X_train if name is None else with_candidate(name, X_train)
        start = time.time()
        try:
            score = validation_scores(fit_leaderboard(X, y_train, None, None, classification, [model_key])).get(
                model_name(model_key))
            error = "" if score is not None else "model missing from the leaderboard"
        except Exception as e:
            score, error = None, str(e)[:200]
        seconds = time.time() - start
        if name is None:
            original = score
        label = None if score is None or original is None or name is None else relative_improvement(original, score)
        rows.append({"dataset": dataset_id, "model": model_key, "device": device(model_key),
                     "task_type": metadata["task_type"], "rows": len(X), "columns": X.shape[1],
                     "feature": name or "original", "seconds": round(seconds, 2), "score_val": score,
                     "improvement": label, "error": error})
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"{name or 'original'}: {seconds:.1f} s, score {score}, improvement {label} {error}")
        if name is None and score is None:
            print(f"{model_name(model_key)} fails on dataset {dataset_id}, stop")
            break
    print(f"-> {out_path}")


def dataset_sizes():
    """Training rows, columns and (approximate) number of candidates of every benchmark dataset."""
    import openml

    candidates = pd.read_parquet(config.LEGACY_CORE_MATRIX_DIR.parent / "core_matrix.parquet",
                                 columns=["dataset - id"]).groupby("dataset - id").size()
    sizes = []
    for task_id in config.DATASETS:
        task = openml.tasks.get_task(task_id, download_data=False, download_qualities=False, download_splits=True,
                                     download_features_meta_data=False)
        train_idx, _ = task.get_train_test_split_indices()
        q = openml.datasets.get_dataset(task.dataset_id, download_data=False, download_qualities=True).qualities
        sizes.append({"dataset": task_id, "train_rows": len(train_idx), "columns": int(q["NumberOfFeatures"]) - 1,
                      "candidates": int(candidates.get(task_id, 0))})
    return pd.DataFrame(sizes)


def summary(experiment):
    files = glob.glob(os.path.join(experiment.pilot_dir, "*_*.csv"))
    if not files:
        raise SystemExit(f"No pilot results in {experiment.pilot_dir}")
    pilot = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    pilot["error"] = pilot["error"].fillna("")
    # Timing from the candidate fits (the first fit of a job also loads libraries / checkpoints)
    ok = pilot[(pilot["error"] == "") & (pilot["feature"] != "original")]
    print("Pilot: median seconds per candidate fit, seconds of the first (original) fit, fits, failures")
    per_dataset = pilot.groupby(["model", "dataset", "rows"]).agg(
        columns=("columns", "min"), seconds=("seconds", lambda x: x.iloc[1:].median()),
        first_fit_s=("seconds", "first"), fits=("seconds", "size"), failed=("error", lambda e: (e != "").sum()),
        device=("device", "first")).reset_index()
    print(per_dataset.to_string(index=False))
    print("\nLabel scale (improvement of the candidates): share >= MIN_IMPROVEMENT, median |improvement|")
    labels = ok.dropna(subset=["improvement"])
    print(labels.groupby("model")["improvement"].agg(
        share_above_min=lambda x: (x >= config.MIN_IMPROVEMENT).mean(), median_abs=lambda x: x.abs().median(),
        max="max").to_string())

    sizes = dataset_sizes()
    max_rows = experiment.max_label_rows or np.inf
    sizes["label_rows"] = np.minimum(sizes["train_rows"], max_rows)
    sizes["labels"] = np.minimum(sizes["candidates"], experiment.max_candidates or np.inf) + 1
    estimates = []
    for model_key, group in ok.groupby("model"):
        g = group.groupby("dataset").agg(seconds=("seconds", "median"), cells=("rows", "first"),
                                         columns=("columns", "first"))
        g["cells"] *= g["columns"]
        if len(g) >= 2 and g["cells"].nunique() >= 2:
            slope, intercept = np.polyfit(np.log(g["cells"]), np.log(g["seconds"]), 1)
        else:  # one dataset: no size dependence can be estimated, constant time (a lower bound)
            print(f"[Warning] {model_key}: pilot on one dataset only, assuming the same time per fit everywhere")
            slope, intercept = 0.0, float(np.log(g["seconds"].iloc[0]))
        per_fit = lambda rows, cols: np.exp(intercept + slope * np.log(rows * cols))
        label_s = (sizes["labels"] * per_fit(sizes["label_rows"], sizes["columns"])).sum()
        eval_s = (len(config.FOLDS) * EVALUATION_FITS_PER_FOLD
                  * per_fit(sizes["train_rows"], sizes["columns"])).sum()
        largest = sizes.loc[(sizes["labels"] * per_fit(sizes["label_rows"], sizes["columns"])).idxmax()]
        estimates.append({"model": model_key, "device": group["device"].iloc[0], "slope": round(slope, 2),
                          "labels": int(sizes["labels"].sum()), "core_matrix_h": round(label_s / 3600, 1),
                          "largest_dataset": int(largest["dataset"]),
                          "largest_dataset_h": round(largest["labels"] * per_fit(largest["label_rows"],
                                                                                 largest["columns"]) / 3600, 1),
                          "evaluation_h": round(eval_s / 3600, 1)})
    estimates = pd.DataFrame(estimates)
    print(f"\nExtrapolation to all {len(sizes)} datasets (max_candidates={experiment.max_candidates}, "
          f"max_label_rows={experiment.max_label_rows}); hours of one job (8 CPUs, plus 1 GPU for GPU models):")
    print(estimates.to_string(index=False))
    print("largest_dataset_h = time of the slowest core-matrix job (one dataset); jobs can be resubmitted to continue."
          "\nChange max_candidates / max_label_rows in metafe/config.py to fit the budget (labels scale linearly).")
    estimates.to_csv(os.path.join(experiment.pilot_dir, "estimate.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="Time the label fits of one dataset and model")
    config.add_experiment_argument(p)
    p.add_argument("--dataset", type=int, required=True)
    p.add_argument("--model", required=True, help="AutoGluon model key")
    p.add_argument("--n_candidates", type=int, default=10)
    p = sub.add_parser("summary", help="Summarize the pilot and extrapolate the compute")
    config.add_experiment_argument(p)
    args = parser.parse_args()
    experiment = config.EXPERIMENTS[args.experiment]
    if args.command == "run":
        check_setup(experiment, [args.model])
        run(args.dataset, args.model, experiment, args.n_candidates)
    else:
        summary(experiment)


if __name__ == "__main__":
    main()
