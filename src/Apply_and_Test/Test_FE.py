import glob

import pandas as pd

from src.utils.get_data import get_openfe_data
from src.utils.get_data import split_data, get_openml_dataset_split
from src.utils.run_models import get_model_score_origin


def main():
    target_label = 'target'
    dataset_id = 359963
    result = pd.DataFrame(columns=['origin', 'dataset', 'model', 'score'])
    X_train, y_train, X_test, y_test = get_openml_dataset_split(int(dataset_id))
    # Original
    try:
        original_results = pd.read_parquet("test_results/Original_Result_" + str(dataset_id) + ".parquet")
    except FileNotFoundError:
        original_results = get_model_score_origin(X_train, y_train, X_test, y_test, dataset_id, "Original")
        result.to_parquet("test_results/Original_Result_" + str(dataset_id) + ".parquet")
    result = pd.concat([result, original_results], ignore_index=True)
    result.to_parquet("test_results/Result_" + str(dataset_id) + ".parquet")
    print("Original Results: " + str(original_results))
    result_files = glob.glob("FE_*.parquet")
    result_files.sort()
    for result_file in result_files:
        print("Test Feature Engineering for " + str(result_file))
        dataset_id = result_file.split('FE_')[-1].split('_')[0]
        name = result_file.split('FE_' + str(dataset_id) + '_')[1]
        if "fold" in name:
            fold = name.split("fold_")[1].split(".")[0]
            origin = name.split('_')[0]
            origin = origin + "_" + str(fold)
        else:
            origin = name.split('_')[0]
        data = pd.read_parquet(result_file)
        try:
            results = pd.read_parquet("test_results/" + str(origin) + "_Result_" + str(dataset_id) + ".parquet")
        except FileNotFoundError:
            X_train, y_train, X_test, y_test = split_data(data, target_label)
            results = get_model_score_origin(X_train, y_train, X_test, y_test, dataset_id, origin)
        result = pd.concat([result, results], ignore_index=True)
        result.to_parquet("test_results/Result_" + str(dataset_id) + ".parquet")
        if origin == "Random":
            print("Random Results: " + str(results))
        if origin == "pandas":
            print("Pandas Results: " + str(results))
        results.to_parquet("test_results/" + str(origin) + "_Result_" + str(dataset_id) + ".parquet")
    # OpenFE
    try:
        openfe_results = pd.read_parquet("test_results/OpenFE_Result_" + str(dataset_id) + ".parquet")
    except FileNotFoundError:
        X_train, y_train, X_test, y_test = get_openfe_data(X_train, y_train, X_test, y_test)
        openfe_results = get_model_score_origin(X_train, y_train, X_test, y_test, dataset_id, "OpenFE")
        openfe_results.to_parquet("test_results/OpenFE_Result_" + str(dataset_id) + ".parquet")
    result = pd.concat([result, openfe_results], ignore_index=True)
    result.to_parquet("test_results/Result_" + str(dataset_id) + ".parquet")
    print("OpenFE Results: " + str(openfe_results))


if __name__ == "__main__":
    main()
