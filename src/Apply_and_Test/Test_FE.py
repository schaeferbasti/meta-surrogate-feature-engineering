import glob

import openml
import pandas as pd
import re
from collections import defaultdict

import pyarrow

from src.Apply_and_Test.Random_Model import run_random_surrogate_model
from src.Apply_and_Test.analyse_results import analyse_results
from src.utils.get_data import get_openfe_data
from src.utils.get_data import split_data, get_openml_dataset_split_and_metadata
from src.utils.run_models import get_model_score_origin_classification, get_model_score_origin_regression


def main():
    target_label = 'target'

    result_files = glob.glob("test_data/FE_*.parquet")
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
        original_path = f"test_results/Original_Result_{dataset_id}.parquet"
        try:
            original_results = pd.read_parquet(original_path)
        except FileNotFoundError:
            if task_type == "Supervised Classification":
                original_results = get_model_score_origin_classification(X_train, y_train, X_test, y_test, dataset_id, "Original")
            else:
                original_results = get_model_score_origin_regression(X_train, y_train, X_test, y_test, dataset_id, "Original")
            original_results = original_results[original_results['model'] == "LightGBM_BAG_L1"]
            original_results.to_parquet(original_path)
        print("Original Results loaded.")

        # === OPENFE RESULTS ===
        if int(dataset_id) != 359983:
            openfe_path = f"test_results/OpenFE_Result_{dataset_id}.parquet"
            try:
                openfe_results = pd.read_parquet(openfe_path)
                print("OpenFE Results loaded.")
            except FileNotFoundError:
                X_openfe_train, y_openfe_train, X_openfe_test, y_openfe_test = get_openfe_data(X_train, y_train, X_test, y_test)
                X_openfe_train = X_openfe_train.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
                X_openfe_test = X_openfe_test.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
                if task_type == "Supervised Classification":
                    openfe_results = get_model_score_origin_classification(X_openfe_train, y_openfe_train, X_openfe_test, y_openfe_test, dataset_id, "OpenFE")
                else:
                    openfe_results = get_model_score_origin_regression(X_openfe_train, y_openfe_train, X_openfe_test, y_openfe_test, dataset_id, "OpenFE")
                openfe_results = openfe_results[openfe_results['model'] == "LightGBM_BAG_L1"]
                openfe_results.to_parquet(openfe_path)
                print("OpenFE Results calculated.")
        else:
            continue

        # === METHOD RESULTS (Random/pandas/MFE/d2v) ===
        combined_results = [original_results, openfe_results]
        best_random_result = None
        best_score = float('-inf')

        for data_file in files:
            print(f"  Processing file: {data_file}")
            name = data_file.split(f'FE_{dataset_id}_')[1]

            if "fold" in name:
                fold = name.split("fold_")[1].split(".")[0]
                method = name.split('_')[0] + f"_{fold}"
                is_random = True
                result_path = f"test_results/{method}_Result_{dataset_id}.parquet"
            elif "MFE" in name:
                category = name.split("MFE_")[1].split("_")[0]
                method = name.split('_')[0]
                is_random = False  # e.g., pandas
                result_path = f"test_results/{method}_{category}_Result_{dataset_id}.parquet"
            else:
                version = name.split('.parquet')[0].split("_")[-1]
                method = name.split('_')[0] + "_" + version
                is_random = False  # e.g., pandas
                result_path = f"test_results/{method}_Result_{dataset_id}.parquet"
            try:
                results = pd.read_parquet(result_path)
            except (FileNotFoundError, pyarrow.lib.ArrowInvalid):
                df = pd.read_parquet(data_file)
                Xf_train, yf_train, Xf_test, yf_test = split_data(df, target_label)
                if task_type == "Supervised Classification":
                    results = get_model_score_origin_classification(Xf_train, yf_train, Xf_test, yf_test, dataset_id, method)
                else:
                    results = get_model_score_origin_regression(Xf_train, yf_train, Xf_test, yf_test, dataset_id, method)
                results = results[results['model'] == "LightGBM_BAG_L1"]
                results.to_parquet(result_path)

            if is_random:
                score = results["score_test"].values[0]
                if score > best_score:
                    best_score = score
                    best_random_result = results
            else:
                combined_results.append(results)  # Only non-random (e.g. pandas)

            # Append the best random result
        if best_random_result is not None:
            combined_results.append(best_random_result)

        all_results = pd.concat(combined_results, ignore_index=True).drop_duplicates()
        all_results.to_parquet(f"test_results/Result_{dataset_id}.parquet")
        print(f"Saved combined results for dataset {dataset_id}.")

    analyse_results()


if __name__ == "__main__":
    main()
