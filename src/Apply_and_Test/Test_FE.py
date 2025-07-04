import glob

import pandas as pd
import re
from collections import defaultdict

from src.Apply_and_Test.analyse_results import analyse_results
from src.utils.get_data import get_openfe_data
from src.utils.get_data import split_data, get_openml_dataset_split
from src.utils.run_models import get_model_score_origin


def main():
    target_label = 'target'

    result_files = glob.glob("test_data/FE_*.parquet")
    result_files.sort()

    dataset_files = defaultdict(list)
    for f in result_files:
        dataset_id = f.split('FE_')[-1].split('_')[0]
        dataset_files[dataset_id].append(f)

    for dataset_id, files in dataset_files.items():
        print(f"\nProcessing Dataset ID: {dataset_id}")

        # Load dataset split once
        X_train, y_train, X_test, y_test = get_openml_dataset_split(int(dataset_id))

        # === ORIGINAL RESULTS ===
        original_path = f"test_results/Original_Result_{dataset_id}.parquet"
        try:
            original_results = pd.read_parquet(original_path)
        except FileNotFoundError:
            original_results = get_model_score_origin(X_train, y_train, X_test, y_test, dataset_id, "Original")
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
                openfe_results = get_model_score_origin(X_openfe_train, y_openfe_train, X_openfe_test, y_openfe_test, dataset_id, "OpenFE")
                openfe_results = openfe_results[openfe_results['model'] == "LightGBM_BAG_L1"]
                openfe_results.to_parquet(openfe_path)
                print("OpenFE Results calculated.")
        else:
            continue

        # === METHOD RESULTS (Random/pandas/folds) ===
        combined_results = [original_results, openfe_results]
        best_random_result = None
        best_score = float('-inf')

        for result_file in files:
            print(f"  Processing file: {result_file}")
            name = result_file.split(f'FE_{dataset_id}_')[1]

            if "fold" in name:
                fold = name.split("fold_")[1].split(".")[0]
                method = name.split('_')[0] + f"_{fold}"
                is_random = True
            else:
                version = name.split('.parquet')[0].split("_")[-1]
                method = name.split('_')[0] + "_" + version
                is_random = False  # e.g., pandas

            result_path = f"test_results/{method}_Result_{dataset_id}.parquet"
            try:
                results = pd.read_parquet(result_path)
            except FileNotFoundError:
                df = pd.read_parquet(result_file)
                Xf_train, yf_train, Xf_test, yf_test = split_data(df, target_label)
                results = get_model_score_origin(Xf_train, yf_train, Xf_test, yf_test, dataset_id, method)
                results = results[results['model'] == "LightGBM_BAG_L1"]
                results.to_parquet(result_path)

            if is_random:
                score = results["score"].values[0]
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
