from __future__ import annotations

import argparse
import multiprocessing
import time
import numpy as np
import pandas as pd
import psutil
from pymfe.mfe import MFE

from src.Apply_and_Test.Apply_FE import execute_feature_engineering_recursive
from src.Metadata.d2v.Add_d2v_Metafeatures import add_d2v_metadata_columns
from src.Metadata.pandas.Add_Pandas_Metafeatures import add_pandas_metadata_columns
from src.Metadata.tabpfn.Add_TabPFN_Metafeatures import add_tabpfn_metadata_columns
from src.Metadata.mfe.Add_MFE_Metafeatures import add_mfe_metadata_columns_group
from autogluon.tabular.models import CatBoostModel

from src.utils.create_feature_and_featurename import create_featurenames, extract_operation_and_original_features
from src.utils.get_data import get_openml_dataset_split_and_metadata, concat_data, split_data
from src.utils.get_matrix import get_matrix_core_columns
from multiprocessing import Value
import ctypes

import warnings
warnings.filterwarnings('ignore')

last_reset_time = Value(ctypes.c_double, time.time())

merge_keys = ["dataset - id", "feature - name", "operator", "model", "improvement"]

def safe_merge(left, right):
    return pd.merge(left, right, on=merge_keys, how="inner")


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
    if method == "pandas":
        result_matrix = pd.read_parquet("src/Metadata/pandas/Pandas_Matrix_Complete.parquet")
    elif method == "tabpfn":
        result_matrix = pd.read_parquet("src/Metadata/tabpfn/TabFPN_Matrix_Complete.parquet")
    else:
        result_matrix = pd.read_parquet("src/Metadata/d2v/D2V_Matrix_Complete.parquet")
        method = "d2v"
    datasets = pd.unique(result_matrix["dataset - id"]).tolist()
    print("Datasets in " + str(method) + " Matrix: " + str(datasets))

    if dataset_id in datasets:
        result_matrix = result_matrix[result_matrix["dataset - id"] != dataset_id]

    start = time.time()
    comparison_result_matrix = create_empty_core_matrix_for_dataset(X, model, dataset_id)
    comparison_result_matrix = add_method_metadata(comparison_result_matrix, dataset_metadata, X, y, method)
    end = time.time()
    print("Time for creating Comparison Result Matrix: " + str(end - start))
    # Predict and split again
    start = time.time()
    X_new, y_new = predict_improvement(result_matrix, comparison_result_matrix, method, X, y, wanted_min_relative_improvement)
    end = time.time()
    print("Time for Predicting Improvement using CatBoost: " + str(end - start))
    if X_new.equals(X):  # if X_new.shape == X.shape
        try:
            y_list = y['target'].tolist()
            y_series = pd.Series(y_list)
            y = y_series
        except KeyError:
            print("")
        data = concat_data(X, y, X_test, y_test, "target")
        data.to_parquet("FE_" + str(dataset_id) + "_" + str(method) + f"_MWI_{wanted_min_relative_improvement}_CatBoost_recursion.parquet")
        return X, y
    else:
        try:
            y_list = y_new['target'].tolist()
            y_series = pd.Series(y_list)
            y_new = y_series
        except KeyError:
            print("")
        data = concat_data(X_new, y_new, X_test, y_test, "target")
        data.to_parquet("FE_" + str(dataset_id) + "_" + str(method) + f"_MWI_{wanted_min_relative_improvement}_CatBoost_recursion.parquet")
        return recursive_feature_addition(X_new, y_new, X_test, y_test, model, method, dataset_metadata, category_to_drop, wanted_min_relative_improvement, dataset_id)


def recursive_feature_addition_mfe_group(X_train, y_train, X_test, y_test, model, method, dataset_id, group, wanted_min_relative_improvement):
    # Reload base matrix
    if group == "info-theory":
        filename = "info_theory"
    else:
        filename = group
    result_matrix = pd.read_parquet("src/Metadata/mfe/MFE_" + str(filename).title() + "_Matrix_Complete.parquet")
    if group == "info_theory":
        groupname = "info-theory"
    else:
        groupname = group
    datasets = pd.unique(result_matrix["dataset - id"]).tolist()
    datasets = [int(x) for x in datasets]
    print("Datasets in MFE_" + str(group).title() + "_Matrix_Complete.parquet: " + str(datasets))

    if dataset_id in datasets:
        result_matrix = result_matrix[result_matrix["dataset - id"] != dataset_id]
    # Create comparison matrix for new dataset
    try:
        data = pd.read_parquet("FE_" + str(dataset_id) + "_" + str(method) + "_" + str(groupname) + "_CatBoost_recursion.parquet")
        X_train, y_train, X_test, y_test = split_data(data, "target")
        return X_train, y_train, X_test, y_test
    except FileNotFoundError:
        start = time.time()
        comparison_result_matrix = create_empty_core_matrix_for_dataset(X_train, model, dataset_id)
        comparison_result_matrix = add_mfe_metadata_columns_group(X_train, y_train, comparison_result_matrix, group)
        end = time.time()
        print("Time for creating Comparison Result Matrix: " + str(end - start))
        # Predict and split again
        start = time.time()
        X_new, y_new = predict_improvement(result_matrix, comparison_result_matrix, method, X_train, y_train, wanted_min_relative_improvement)
        end = time.time()
        print("Time for Predicting Improvement using CatBoost: " + str(end - start))
        if X_new.equals(X_train):  # if X_new.shape == X.shape
            try:
                y_list = y_train['target'].tolist()
                y_series = pd.Series(y_list)
                y_train = y_series
            except KeyError:
                print("")
            data = concat_data(X_train, y_train, X_test, y_test, "target")
            data.to_parquet(f"FE_{dataset_id}_{method}_{groupname}_CatBoost_recursion.parquet")
            return X_train, y_train
        else:
            try:
                y_list = y_new['target'].tolist()
                y_series = pd.Series(y_list)
                y_new = y_series
            except KeyError:
                print("")
            data = concat_data(X_new, y_new, X_test, y_test, "target")
            data.to_parquet(f"FE_{dataset_id}_{method}_{groupname}_CatBoost_recursion.parquet")
            return recursive_feature_addition_mfe_group(X_new, y_new, X_test, y_test, model, method, dataset_id, group, wanted_min_relative_improvement)


def recursive_feature_addition_mfe_groups(X_train, y_train, X_test, y_test, model, method, dataset_id, groups, wanted_min_relative_improvement):
    # Reload base matrix
    group_set = set(groups)
    if group_set == {"general", "statistical"}:
        groupname = "Without_Info_Theory"
        groupname1, groupname2 = "general", "statistical"
        groupname3 = None
    elif group_set == {"general", "info_theory"}:
        groupname = "Without_Statistical"
        groupname1, groupname2 = "general", "info-theory"
        groupname3 = None
    elif group_set == {"statistical", "info_theory"}:
        groupname = "Without_General"
        groupname1, groupname2 = "statistical", "info-theory"
        groupname3 = None
    elif group_set == {"general", "statistical", "info_theory"}:
        groupname = "All"
        groupname1, groupname2, groupname3 = "general", "statistical", "info-theory"
    else:
        raise ValueError(f"Unknown group combination: {groups}")
    result_matrix = pd.read_parquet("src/Metadata/mfe/MFE_" + str(groupname) + "_Matrix_Complete.parquet")
    datasets = pd.unique(result_matrix["dataset - id"]).tolist()
    print("Datasets in " + str(method) + " Matrix: " + str(datasets))

    if dataset_id in datasets:
        result_matrix = result_matrix[result_matrix["dataset - id"] != dataset_id]
    # Create comparison matrix for new dataset
    try:
        data = pd.read_parquet("FE_" + str(dataset_id) + "_" + str(method) + "_" + str(groupname) + "_CatBoost_recursion.parquet")
        X_train, y_train, X_test, y_test = split_data(data, "target")
        return X_train, y_train, X_test, y_test
    except FileNotFoundError:
        start = time.time()
        empty_comparison_result_matrix = create_empty_core_matrix_for_dataset(X_train, model, dataset_id)
        comparison_result_matrix_1 = add_mfe_metadata_columns_group(X_train, y_train, empty_comparison_result_matrix, groupname1)
        comparison_result_matrix_2 = add_mfe_metadata_columns_group(X_train, y_train, empty_comparison_result_matrix, groupname2)
        if groupname3 is None:
            comparison_result_matrix = safe_merge(comparison_result_matrix_1, comparison_result_matrix_2)
        else:
            comparison_result_matrix_3 = add_mfe_metadata_columns_group(X_train, y_train, empty_comparison_result_matrix, groupname3)
            comparison_result_matrix = safe_merge(safe_merge(comparison_result_matrix_1, comparison_result_matrix_2), comparison_result_matrix_3)
        end = time.time()
        print("Time for creating Comparison Result Matrix: " + str(end - start))
        # comparison_result_matrix.to_parquet("Comparison_Result_Matrix.parquet")
        # Predict and split again
        start = time.time()
        X_new, y_new = predict_improvement(result_matrix, comparison_result_matrix, method, X_train, y_train, wanted_min_relative_improvement)
        end = time.time()
        print("Time for Predicting Improvement using CatBoost: " + str(end - start))
        if X_new.equals(X_train):  # if X_new.shape == X.shape
            try:
                y_list = y_train['target'].tolist()
                y_series = pd.Series(y_list)
                y_train = y_series
            except KeyError:
                print("")
            data = concat_data(X_train, y_train, X_test, y_test, "target")
            data.to_parquet(f"FE_{dataset_id}_{method}_{str(groups)}_CatBoost_recursion.parquet")
            return X_train, y_train
        else:
            try:
                y_list = y_new['target'].tolist()
                y_series = pd.Series(y_list)
                y_new = y_series
            except KeyError:
                print("")
            data = concat_data(X_new, y_new, X_test, y_test, "target")
            data.to_parquet(f"FE_{dataset_id}_{method}_{str(groups)}_CatBoost_recursion.parquet")
            return recursive_feature_addition_mfe_group(X_new, y_new, X_test, y_test, model, method, dataset_id, groups, wanted_min_relative_improvement)


def predict_improvement(result_matrix, comparison_result_matrix, category_or_method, X_train, y_train, wanted_min_relative_improvement):
    y_result = result_matrix["improvement"]
    result_matrix = result_matrix.drop("improvement", axis=1)
    comparison_result_matrix = comparison_result_matrix.drop("improvement", axis=1)
    # clf = RealMLPModel()
    # clf = TabDPTModel()
    clf = CatBoostModel()
    clf.fit(X=result_matrix, y=y_result)

    # Predict and score
    comparison_result_matrix = comparison_result_matrix[result_matrix.columns]
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
    return X, y


def process_group(dataset_id, method, group, model, wanted_min_relative_improvement, last_reset_time):
    last_reset_time.value = time.time()
    if group == "info_theory":
        groupname = "info-theory"
    else:
        groupname = group
    print(f"[Processing Group] {groupname}")
    X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset_id)
    X_train, y_train, X_test, y_test = recursive_feature_addition_mfe_group(X_train, y_train, X_test, y_test, model, method, dataset_id, groupname, wanted_min_relative_improvement)
    data = concat_data(X_train, y_train, X_test, y_test, "target")
    data.to_parquet(f"FE_{dataset_id}_{method}_{groupname}_MWI_{wanted_min_relative_improvement}_CatBoost_recursion.parquet")


def process_groups(dataset_id, method, groups, model, wanted_min_relative_improvement, last_reset_time):
    last_reset_time.value = time.time()
    X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset_id)
    X_train, y_train, X_test, y_test = recursive_feature_addition_mfe_groups(X_train, y_train, X_test, y_test, model, method, dataset_id, groups, wanted_min_relative_improvement)
    data = concat_data(X_train, y_train, X_test, y_test, "target")
    data.to_parquet(f"FE_{dataset_id}_{method}_{str(groups)}_MWI_{wanted_min_relative_improvement}_CatBoost_recursion.parquet")


def process_method(dataset_id, method, model, wanted_min_relative_improvement, last_reset_time):
    last_reset_time.value = time.time()
    X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset_id)
    start = time.time()
    X_train, y_train = recursive_feature_addition(X_train, y_train, X_test, y_test, model, method, dataset_metadata, None, wanted_min_relative_improvement, dataset_id)
    end = time.time()
    print("Time for creating Comparison Result Matrix: " + str(end - start))
    data = concat_data(X_train, y_train, X_test, y_test, "target")
    data.to_parquet("FE_" + str(dataset_id) + "_" + str(method) + f"_MWI_{wanted_min_relative_improvement}_CatBoost_recursion.parquet")


def main(dataset_id, method, wanted_min_relative_improvement):
    print("Method: " + str(method) + ", Dataset: " + str(dataset_id) + ", Model: " + str("CatBoost"))
    model = "LightGBM_BAG_L1"
    if method.startswith("MFE"):

        groups = ["general", "statistical", "info_theory"]
        for group in groups:
            print(f"\n=== Starting group: {group} ===")
            process_func = lambda: process_group(dataset_id, method, group, model, wanted_min_relative_improvement, last_reset_time)
            exit_code = run_with_resource_limits(process_func, mem_limit_mb=64000, time_limit_sec=7200)
            if exit_code != 0:
                print(f"[Warning] Group {group} failed or was terminated. Skipping.\n")
                continue

        groupnames = {groups[0], groups[1]}
        last_reset_time.value = time.time()
        print(f"\n=== Starting groups: {groupnames} ===")
        process_func = lambda: process_groups(dataset_id, method, groupnames, model, wanted_min_relative_improvement, last_reset_time)
        exit_code = run_with_resource_limits(process_func, mem_limit_mb=64000, time_limit_sec=7200)
        if exit_code != 0:
            print(f"[Warning] Groups {groupnames} failed or was terminated. Skipping.\n")

        groupnames = {groups[1], groups[2]}
        last_reset_time.value = time.time()
        print(f"\n=== Starting groups: {groupnames} ===")
        process_func = lambda: process_groups(dataset_id, method, groupnames, model, wanted_min_relative_improvement, last_reset_time)
        exit_code = run_with_resource_limits(process_func, mem_limit_mb=64000, time_limit_sec=7200)
        if exit_code != 0:
            print(f"[Warning] Groups {groupnames} failed or was terminated. Skipping.\n")

        groupnames = {groups[0], groups[2]}
        last_reset_time.value = time.time()
        print(f"\n=== Starting groups: {groupnames} ===")
        process_func = lambda: process_groups(dataset_id, method, groupnames, model, wanted_min_relative_improvement, last_reset_time)
        exit_code = run_with_resource_limits(process_func, mem_limit_mb=64000, time_limit_sec=7200)
        if exit_code != 0:
            print(f"[Warning] Groups {groupnames} failed or was terminated. Skipping.\n")

        groupnames = {groups[0], groups[1], groups[2]}
        last_reset_time.value = time.time()
        print(f"\n=== Starting groups: {groupnames} ===")
        process_func = lambda: process_groups(dataset_id, method, groupnames, model, wanted_min_relative_improvement, last_reset_time)
        exit_code = run_with_resource_limits(process_func, mem_limit_mb=64000, time_limit_sec=7200)
        if exit_code != 0:
            print(f"[Warning] Groups {groupnames} failed or was terminated. Skipping.\n")
    else:
        print(f"\n=== Starting Method: {method} ===")
        process_func = lambda: process_method(dataset_id, method, model, wanted_min_relative_improvement, last_reset_time)
        exit_code = run_with_resource_limits(process_func, mem_limit_mb=64000, time_limit_sec=7200)
        if exit_code != 0:
            print(f"[Warning] Method {method} failed or was terminated. Skipping.\n")


def run_with_resource_limits(target_func, mem_limit_mb, time_limit_sec, check_interval=5):
    process = multiprocessing.Process(target=target_func)
    process.start()
    pid = process.pid

    while process.is_alive():
        try:
            mem = psutil.Process(pid).memory_info().rss / (1024 * 1024)  # MB
            elapsed_time = time.time() - last_reset_time.value
            print("Elapsed Time: " + str(elapsed_time))

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
    methods = ["pandas", "d2v", "MFE"]
    wanted_min_relative_improvements = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
    for wanted_min_relative_improvement in wanted_min_relative_improvements:
        for method in methods:
            main(int(args.dataset), method, wanted_min_relative_improvement)


if __name__ == '__main__':
    last_reset_time = Value(ctypes.c_double, time.time())
    main_wrapper()
