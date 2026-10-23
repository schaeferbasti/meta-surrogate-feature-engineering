import argparse
import glob
import os
import re
import time

import pandas as pd
from collections import defaultdict

import pyarrow

from src.utils.get_data import split_data_original, get_openml_dataset_split_and_metadata, get_openfe_data, \
    concat_data, get_tabprep_data
from src.utils.run_models import get_model_score_origin_classification, get_model_score_origin_regression

# All methods (incl. Original) are evaluated on the original OpenML train/test split.
# Results computed with the old random re-split live in test_results/ and must not be reused as cache.
RESULTS_DIR = "test_results_same_split"

# Datasets on which OpenFE did not finish
NO_OPENFE_DATASETS = [2073, 359930, 359931, 359935, 359938, 359959, 359962, 359983]


def clean_column_names(X):
    return X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', str(x)))


def get_scores(X_train, y_train, X_test, y_test, task_type, dataset_id, origin):
    # AutoGluon adds the target column to the passed frames, so always hand over copies
    X_train, y_train, X_test, y_test = X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy()
    if task_type == "Supervised Classification":
        results = get_model_score_origin_classification(X_train, y_train, X_test, y_test, dataset_id, origin)
    else:
        results = get_model_score_origin_regression(X_train, y_train, X_test, y_test, dataset_id, origin)
    return results[results['model'] == "LightGBM_BAG_L1"]


def get_baseline_fe_results(name, get_fe_data, dataset_id, fold, X_train, y_train, X_test, y_test, task_type):
    # FE methods that are run directly by this script (OpenFE, TabPrep)
    result_path = f"{RESULTS_DIR}/{name}_Result_{dataset_id}_{fold}.parquet"
    try:
        results = pd.read_parquet(result_path)
        print(f"{name} Results loaded.")
        return results
    except FileNotFoundError:
        pass
    data_path = f"test_data/FE_{dataset_id}_{name}.parquet"
    try:
        data = pd.read_parquet(data_path)
        X_fe_train, y_fe_train, X_fe_test, y_fe_test = split_data_original(data, "target", X_train, X_test)
    except FileNotFoundError:
        start = time.time()
        X_fe_train, y_fe_train, X_fe_test, y_fe_test = get_fe_data(X_train.copy(), y_train.copy(), X_test.copy(),
                                                                   y_test.copy())
        end = time.time()
        print(f"{name} Time: {end - start}")
        data = concat_data(X_fe_train, y_fe_train, X_fe_test, y_fe_test, "target")
        data.to_parquet(data_path)
    X_fe_train = clean_column_names(X_fe_train)
    X_fe_test = clean_column_names(X_fe_test)
    results = get_scores(X_fe_train, y_fe_train, X_fe_test, y_fe_test, task_type, dataset_id, name)
    results.to_parquet(result_path)
    print(f"{name} Results calculated.")
    return results


def main(fold):
    print(fold)
    target_label = 'target'
    os.makedirs(RESULTS_DIR, exist_ok=True)

    result_files = glob.glob("test_data/*FE_*.parquet")
    result_files.sort()

    dataset_files = defaultdict(list)
    for f in result_files:
        dataset_id = f.split('FE_')[1].split('_')[0]
        dataset_files[dataset_id].append(f)

    for dataset_id, files in dataset_files.items():
        print(f"\nProcessing Dataset ID: {dataset_id} and files: {files}")
        X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(int(dataset_id))
        task_type = dataset_metadata["task_type"]
        print(task_type)

        # === ORIGINAL RESULTS ===
        original_path = f"{RESULTS_DIR}/Original_Result_{dataset_id}_{fold}.parquet"
        try:
            original_results = pd.read_parquet(original_path)
        except FileNotFoundError:
            original_results = get_scores(X_train, y_train, X_test, y_test, task_type, dataset_id, "Original")
            original_results.to_parquet(original_path)
        print("Original Results loaded.")
        combined_results = [original_results]

        # === OPENFE RESULTS ===
        if int(dataset_id) not in NO_OPENFE_DATASETS:
            combined_results.append(get_baseline_fe_results(
                "OpenFE", get_openfe_data, dataset_id, fold, X_train, y_train, X_test, y_test, task_type))
        else:
            print("No OpenFE")

        # === TabPrep RESULTS ===
        get_tabprep = lambda X_tr, y_tr, X_te, y_te: get_tabprep_data(X_tr, y_tr, X_te, y_te, task_type)
        combined_results.append(get_baseline_fe_results(
            "TabPrep", get_tabprep, dataset_id, fold, X_train, y_train, X_test, y_test, task_type))

        # === METHOD RESULTS (Random/pandas/MFE/d2v) ===
        best_random_result = None
        best_score = float('-inf')

        for data_file in files:
            # OpenFE and TabPrep are handled above
            if "OpenFE" in data_file or "TabPrep" in data_file:
                print("Skip")
            else:
                print(f"  Processing file: {data_file}")
                name = data_file.split(f'FE_{dataset_id}_')[1]

                if "fold" in name:
                    random_fold = name.split("fold_")[1].split(".")[0]
                    method = name.split('_')[0] + f"_{random_fold}"
                    is_random = True
                    result_path = f"{RESULTS_DIR}/{method}_Result_{dataset_id}_{fold}.parquet"
                elif "MFE" in name:
                    version = name.split('.parquet')[0].split("_")[-1]
                    name_info = name.replace("info_theory", "info-theory")
                    category = name_info.split("MFE_")[1].split("_")[0]
                    method = name.split('_')[0] + f"_{category}_{version}"
                    is_random = False
                    result_path = f"{RESULTS_DIR}/{method}_Result_{dataset_id}_{fold}.parquet"
                else:
                    prefix = data_file.split('/')[1].split('_')[0]
                    if prefix.isdigit():
                        time_limit = prefix
                        base_name = name.split('_')[0]
                        method = f"{time_limit}_{base_name}"
                    else:
                        version = name.split('.parquet')[0].split("_")[-1]
                        method = name.split('_')[0] + "_" + version
                    is_random = False
                    result_path = f"{RESULTS_DIR}/{method}_Result_{dataset_id}_{fold}.parquet"

                try:
                    results = pd.read_parquet(result_path)
                except (FileNotFoundError, pyarrow.lib.ArrowInvalid):
                    df = pd.read_parquet(data_file)
                    try:
                        Xf_train, yf_train, Xf_test, yf_test = split_data_original(df, target_label, X_train, X_test)
                    except ValueError as e:
                        print(f"[Warning] Skipping {data_file}: {e}")
                        continue
                    results = get_scores(Xf_train, yf_train, Xf_test, yf_test, task_type, dataset_id, method)
                    results.to_parquet(result_path)

                if is_random:
                    score = results["score_test"].values[0]
                    if score > best_score:
                        best_score = score
                        best_random_result = results
                else:
                    combined_results.append(results)

        # Append the best random result
        if best_random_result is not None:
            combined_results.append(best_random_result)

        all_results = pd.concat(combined_results, ignore_index=True).drop_duplicates()
        all_results.to_parquet(f"{RESULTS_DIR}/Result_{dataset_id}_{fold}.parquet")
        print(f"Saved combined results for dataset {dataset_id} and fold {fold}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None, help="Run a single fold (default: all 10 folds)")
    args = parser.parse_args()
    folds = range(10) if args.fold is None else [args.fold]
    for fold in folds:
        main(fold)
