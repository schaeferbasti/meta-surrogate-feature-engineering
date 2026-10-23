from __future__ import annotations

import argparse
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch

from src.Apply_and_Test.Apply_FE import execute_feature_engineering
from src.Metadata.pandas.Add_Pandas_Metafeatures import add_pandas_metadata_columns
from src.SurrogateModel.Pretrain_PFN import CustomFeatureSelectionPFN
from src.utils.create_feature_and_featurename import create_featurenames, extract_operation_and_original_features
from src.utils.get_data import get_openml_dataset_split_and_metadata, concat_data
from src.utils.get_matrix import get_matrix_core_columns

import warnings

warnings.filterwarnings('ignore')

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


def add_method_metadata(result_matrix, dataset_metadata, X_predict, y_predict, method):
    return add_pandas_metadata_columns(dataset_metadata, X_predict, result_matrix)


def feature_addition(X_train, y_train, X_test, y_test, model, method, dataset_metadata, dataset_id, number_of_features):
    start = time.time()
    comparison_result_matrix = create_empty_core_matrix_for_dataset(X_train, model, dataset_id)
    comparison_result_matrix = add_method_metadata(comparison_result_matrix, dataset_metadata, X_train, y_train, method)
    end = time.time()
    print("Time for creating Comparison Result Matrix: " + str(end - start))

    start = time.time()
    X_new, y_new = predict_improvement(comparison_result_matrix, number_of_features)
    end = time.time()
    print("Time for Predicting Improvement using Custom PFN: " + str(end - start))

    try:
        y_list = y_new['target'].tolist()
        y_series = pd.Series(y_list)
        y_new = y_series
    except KeyError:
        pass

    data = concat_data(X_new, y_new, X_test, y_test, "target")
    data.to_parquet("FE_" + str(dataset_id) + "_" + str(method) + "_fePFN.parquet")
    return X_new, y_new, X_test, y_test


@torch.no_grad()
def predict_improvement(comparison_result_matrix, number_of_features):
    # Notice: We no longer load or pass `result_matrix` (the 147k rows)!
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    identifier_cols = ["dataset - id", "feature - name", "operator", "model", "improvement"]

    # 1. Prepare the candidate operations for the NEW dataset (The Queries)
    X_test_numeric = comparison_result_matrix.drop(columns=[c for c in identifier_cols if c in comparison_result_matrix.columns])
    X_test_numeric = X_test_numeric.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_test_numeric = X_test_numeric.apply(pd.to_numeric, errors='coerce')
    X_test_numeric = X_test_numeric.fillna(0.0)
    X_test_numeric = X_test_numeric.replace([np.inf, -np.inf], 0.0)
    scaler = joblib.load('pfn_scaler.pkl')
    X_test_numeric = scaler.transform(X_test_numeric.values)

    # Shape: (1, N_queries, Features)
    x_qry = torch.tensor(X_test_numeric, dtype=torch.float32).unsqueeze(0).to(device)

    # For a purely zero-shot forward pass (no prior evaluations on the new dataset),
    # the context is empty.
    x_ctx = torch.zeros((1, 1, X_test_numeric.shape[1]), dtype=torch.float32).to(device)
    y_ctx = torch.zeros((1, 1, 1), dtype=torch.float32).to(device)
    # 2. Instantiate Architecture and Load Pre-trained Parameters
    clf = CustomFeatureSelectionPFN(num_features=X_test_numeric.shape[1]).to(device)
    clf.load_state_dict(torch.load('pfn_weights.pt', map_location=device))
    clf.eval()

    for name, param in clf.named_parameters():
        if torch.isnan(param).any():
            print(f"CRITICAL ERROR: Weight '{name}' contains NaNs!")
            break

    # 3. SINGLE FORWARD PASS
    # The pre-trained weights do all the heavy lifting.
    predicted_deltas = clf(x_ctx, y_ctx, x_qry)
    prediction = predicted_deltas.squeeze(0).cpu().numpy()

    # Extract top operations
    prediction_df = pd.DataFrame(prediction, columns=["predicted_improvement"])
    original_cols_df = comparison_result_matrix[["dataset - id", "feature - name", "model"]].reset_index(drop=True)
    prediction_concat_df = pd.concat([original_cols_df, prediction_df], axis=1)

    best_operation = prediction_concat_df.nlargest(n=number_of_features, columns="predicted_improvement", keep="first")
    best_operation = best_operation.sort_values(key=lambda s: s.str.startswith("without - "), by="feature - name")

    X, y, _, _ = execute_feature_engineering(best_operation)
    return X, y


def process_method(dataset_id, method, model, number_of_features):
    file_name = f"FE_{dataset_id}_{method}_fePFN.parquet"
    if os.path.exists(file_name):
        print(f"File {file_name} already exists. Skipping dataset {dataset_id}.")
        return
    X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset_id)
    start = time.time()
    X_train, y_train, X_test, y_test = feature_addition(X_train, y_train, X_test, y_test, model, method, dataset_metadata, dataset_id, number_of_features)
    end = time.time()
    print("Total Time SM: " + str(end - start))
    data = concat_data(X_train, y_train, X_test, y_test, "target")
    data.to_parquet(file_name)


def main(dataset_id, method, number_of_features):
    print("Method: " + str(method) + ", Dataset: " + str(dataset_id) + ", Model: " + str("Custom PFN Surrogate"))
    model = "LightGBM_BAG_L1"

    print(f"\n=== Starting Method: {method} ===")
    process_method(dataset_id, method, model, number_of_features)


def main_wrapper():
    parser = argparse.ArgumentParser(
        description='Run Surrogate Model with Metadata from Method locally using Custom PFN')
    parser.add_argument('--dataset', default=2073, help='Dataset ID')
    args = parser.parse_args()

    methods = ["pandas"]
    number_of_features = 100

    datasets = [2073]

    for dataset in datasets:
        for method in methods:
            main(dataset, method, number_of_features)


if __name__ == '__main__':
    main_wrapper()
