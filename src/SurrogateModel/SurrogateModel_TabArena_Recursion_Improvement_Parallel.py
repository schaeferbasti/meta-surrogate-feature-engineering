from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
import numpy as np
import pandas as pd
import psutil
from pymfe.mfe import MFE

from src.Apply_and_Test.Apply_FE import execute_feature_engineering_recursive
from src.Metadata.d2v.Add_d2v_Metafeatures import add_d2v_metadata_columns
from src.Metadata.pandas.Add_Pandas_Metafeatures import add_pandas_metadata_columns
from src.Metadata.tabpfn.Add_TabPFN_Metafeatures import add_tabpfn_metadata_columns
from src.Metadata.mfe.Add_MFE_Metafeatures import add_mfe_metadata_columns
from autogluon.tabular.models import CatBoostModel

from src.utils.create_feature_and_featurename import create_featurenames, extract_operation_and_original_features
from src.utils.get_data import get_openml_dataset_split_and_metadata, concat_data
from src.utils.get_matrix import get_matrix_core_columns

import warnings
warnings.filterwarnings('ignore')


def create_empty_core_matrix_for_dataset(X_train, model, dataset_id) -> pd.DataFrame:
    columns = get_matrix_core_columns()
    comparison_result_matrix = pd.DataFrame(columns=columns)
    for feature1 in X_train.columns:
        featurename = "without - " + str(feature1)
        columns = get_matrix_core_columns()
        new_rows = pd.DataFrame(columns=columns)
        operator = "delete"
        new_rows.loc[len(new_rows)] = [
            dataset_id,
            featurename,
            operator,
            model,
            0
        ]
        comparison_result_matrix = pd.concat([comparison_result_matrix, pd.DataFrame(new_rows)], ignore_index=True)
    columns = get_matrix_core_columns()
    new_rows = pd.DataFrame(columns=columns)
    featurenames = create_featurenames(X_train.columns)
    for i in range(len(featurenames)):
        operator, _ = extract_operation_and_original_features(featurenames[i])
        new_rows.loc[len(new_rows)] = [
            dataset_id,
            featurenames[i],
            operator,
            model,
            0
        ]
    comparison_result_matrix = pd.concat([comparison_result_matrix, pd.DataFrame(new_rows)], ignore_index=True)
    return comparison_result_matrix


def get_mfe_categories():
    X_dummy = np.array([[0, 1], [1, 0]])
    y_dummy = np.array([0, 1])

    # Initialize result dictionary
    groups = [
        "general",
        "statistical",
        "info-theory",
        "landmarking",
        "complexity",
        "clustering",
        "concept",
        "itemset"
    ]

    # This will hold your result like:
    # [dataset_metadata_general_names, dataset_metadata_statistical_names, ...]
    group_feature_lists = []

    for group in groups:
        mfe = MFE(groups=[group])
        mfe.fit(X_dummy, y_dummy)
        feature_names, _ = mfe.extract()
        group_feature_lists.append(feature_names)
    return group_feature_lists


def add_method_metadata(result_matrix, dataset_metadata, X_predict, y_predict, method):
    if method == "d2v":
        result_matrix = add_d2v_metadata_columns(dataset_metadata, X_predict, result_matrix)
    elif method == "pandas":
        result_matrix = add_pandas_metadata_columns(dataset_metadata, X_predict, result_matrix)
    elif method == "tabpfn":
        result_matrix = add_tabpfn_metadata_columns(X_predict, y_predict, result_matrix)
    return result_matrix


def recursive_feature_addition(X, y, X_test, y_test, model, method, dataset_metadata, category_to_drop, wanted_min_relative_improvement, dataset_id):
    # Reload base matrix
    if method == "pandas":
        result_matrix = pd.read_parquet("src/Metadata/pandas/Pandas_Matrix_Complete.parquet")
    elif method == "tabpfn":
        result_matrix = pd.read_parquet("src/Metadata/tabpfn/TabFPN_Matrix_Complete.parquet")
    else:
        result_matrix = pd.read_parquet("src/Metadata/d2v/D2V_Matrix_Complete.parquet")
        method = "d2v"    # Create comparison matrix for new dataset
    datasets = pd.unique(result_matrix["dataset - id"]).tolist()
    print("Datasets in " + str(method) + " Matrix: " + str(datasets))

    if dataset_id in datasets:
        result_matrix = result_matrix[result_matrix["dataset - id"] != dataset_id]

    start = time.time()
    comparison_result_matrix = create_empty_core_matrix_for_dataset(X, model, dataset_id)
    comparison_result_matrix = add_method_metadata(comparison_result_matrix, dataset_metadata, X, y, method)
    end = time.time()
    print("Time for creating Comparison Result Matrix: " + str(end - start))
    comparison_result_matrix.to_parquet("Comparison_Result_Matrix.parquet")
    # Predict and split again
    start = time.time()
    X_new, y_new = predict_improvement(result_matrix, comparison_result_matrix, method, X, y, wanted_min_relative_improvement)
    end = time.time()
    print("Time for Predicting Improvement using CatBoost: " + str(end - start))
    if X_new.equals(X):  # if X_new.shape == X.shape
        data = concat_data(X, y, X_test, y_test, "target")
        data.to_parquet("FE_" + str(dataset_id) + "_" + str(method) + "_CatBoost_recursion.parquet")
        return X, y
    else:
        data = concat_data(X_new, y_new, X_test, y_test, "target")
        data.to_parquet("FE_" + str(dataset_id) + "_" + str(method) + "_CatBoost_recursion.parquet")
        return recursive_feature_addition(X_new, y_new, X_test, y_test, model, method, dataset_metadata, category_to_drop, wanted_min_relative_improvement, dataset_id)


def recursive_feature_addition_mfe(X, y, X_test, y_test, model, method, dataset_metadata, category_to_drop, wanted_min_relative_improvement, dataset_id):
    # Reload base matrix
    result_matrix = pd.read_parquet("src/Metadata/mfe/MFE_Matrix_Complete.parquet")
    # Create comparison matrix for new dataset
    comparison_result_matrix = create_empty_core_matrix_for_dataset(X, model, dataset_id)
    comparison_result_matrix, _, _, _, _, _, _, _, _ = add_mfe_metadata_columns(X, y, comparison_result_matrix)
    comparison_result_matrix, general, statistical, info_theory, landmarking, complexity, clustering, concept, itemset = add_mfe_metadata_columns(X, y, comparison_result_matrix)
    # Drop no category, single category or all categories but one
    comparison_result_matrix_copy = comparison_result_matrix.drop(columns=category_to_drop, errors='ignore')
    result_matrix_copy = result_matrix.drop(columns=category_to_drop, errors='ignore')
    # Predict and split again
    X_new, y_new = predict_improvement(result_matrix_copy, comparison_result_matrix_copy, "all", X, y, wanted_min_relative_improvement)
    # Recurse
    if X_new.equals(X):  # if X_new.shape == X.shape
        data = concat_data(X, y, X_test, y_test, "target")
        data.to_parquet("FE_" + str(dataset_id) + "_" + str(method) + "_CatBoost_recursion.parquet")
        return X, y
    else:
        data = concat_data(X_new, y_new, X_test, y_test, "target")
        data.to_parquet("FE_" + str(dataset_id) + "_" + str(method) + "_CatBoost_recursion.parquet")
        return recursive_feature_addition(X_new, y_new, X_test, y_test, model, method, dataset_metadata, category_to_drop, wanted_min_relative_improvement, dataset_id)


def predict_improvement(result_matrix, comparison_result_matrix, category_or_method, X_train, y_train, wanted_min_relative_improvement):
    y_result = result_matrix["improvement"]
    result_matrix = result_matrix.drop("improvement", axis=1)
    y_comparison = comparison_result_matrix["improvement"]
    comparison_result_matrix = comparison_result_matrix.drop("improvement", axis=1)
    print("Old columns: " + str(X_train.columns))
    # Train TabArena Model
    # clf = RealMLPModel()
    # clf = TabDPTModel()
    clf = CatBoostModel()
    clf.fit(X=result_matrix, y=y_result)

    # Predict and score
    prediction = clf.predict(X=comparison_result_matrix)
    prediction_df = pd.DataFrame(prediction, columns=["predicted_improvement"])
    prediction_concat_df = pd.concat([comparison_result_matrix[["dataset - id", "feature - name", "model"]], prediction_df], axis=1)
    prediction_concat_df.to_parquet("Prediction_" + str(category_or_method) + ".parquet")
    best_operation = prediction_concat_df.nlargest(n=1, columns="predicted_improvement", keep="first")
    if best_operation["predicted_improvement"].values[0] < wanted_min_relative_improvement:
        print("Predicted improvement of best operation: " + str(best_operation["predicted_improvement"].values[0]) + " - not good enough")
        return X_train, y_train
    else:
        print("Predicted improvement of best operation: " + str(best_operation["predicted_improvement"].values[0]) + " - execute feature engineering")
        X, y, _, _ = execute_feature_engineering_recursive(best_operation, X_train, y_train)
        print("New columns: " + str(X.columns))
    return X, y


def main(dataset_id, wanted_min_relative_improvement, method):
    model = "LightGBM_BAG_L1"
    X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset_id)
    start = time.time()
    X_train, y_train = recursive_feature_addition(X_train, y_train, X_test, y_test, model, method, dataset_metadata, None, wanted_min_relative_improvement, dataset_id)
    end = time.time()
    print("Time for creating Comparison Result Matrix: " + str(end - start))
    data = concat_data(X_train, y_train, X_test, y_test, "target")
    data.to_parquet("FE_" + str(dataset_id) + "_" + str(method) + "_CatBoost_recursion.parquet")


def run_with_resource_limits(target_func, mem_limit_mb, time_limit_sec, check_interval=5):
    process = multiprocessing.Process(target=target_func)
    process.start()
    pid = process.pid
    start_time = time.time()

    while process.is_alive():
        try:
            mem = psutil.Process(pid).memory_info().rss / (1024 * 1024)  # MB
            elapsed_time = time.time() - start_time

            if mem > mem_limit_mb:
                print(f"[Monitor] Memory exceeded: {mem:.2f} MB > {mem_limit_mb} MB. Terminating.")
                process.terminate()
                break

            if elapsed_time > time_limit_sec:
                print(f"[Monitor] Time limit exceeded: {elapsed_time:.1f} sec > {time_limit_sec} sec. Terminating.")
                process.terminate()
                break

        except psutil.NoSuchProcess:
            break
        time.sleep(check_interval)

    process.join()
    return process.exitcode


def main_wrapper():
    parser = argparse.ArgumentParser(description='Run CatBoost Surrogate Model with Metadata from Method')
    # parser.add_argument('--mf_method', required=True, help='Metafeature Method')
    parser.add_argument('--dataset', required=True, help='Metafeature Method')
    args = parser.parse_args()
    method = "d2v"
    wanted_min_relative_improvement = 0.1
    main(int(args.dataset), wanted_min_relative_improvement, method)


if __name__ == '__main__':
    memory_limit_mb = 64000     # 64 GB
    time_limit_sec = 3600       # 1h
    exit_code = run_with_resource_limits(main_wrapper, memory_limit_mb, time_limit_sec)
    if exit_code != 0:
        print(f"Process exited with code {exit_code}")
        sys.exit(exit_code)
