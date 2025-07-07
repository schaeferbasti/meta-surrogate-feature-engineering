import os

import pandas as pd

from src.utils.create_feature_and_featurename import create_feature_and_featurename
from src.utils.get_data import get_openml_dataset
from src.utils.create_feature_and_featurename import extract_operation_and_original_features

import warnings
warnings.filterwarnings("ignore")


def get_additional_features(X, y, prediction_result):
    additional_feature_list = prediction_result['feature - name']
    for additional_feature in additional_feature_list:
        operation, original_features = extract_operation_and_original_features(additional_feature)
        if len(original_features) == 2:
            feature, featurename = create_feature_and_featurename(X[original_features[0]], X[original_features[1]], operation)
        else:
            feature, featurename = create_feature_and_featurename(X[original_features[0]], None, operation)
        if feature is not None:
            feature = pd.Series(feature).to_frame(additional_feature)
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)
            feature = feature.reset_index(drop=True)
            X = pd.concat([X, feature], axis=1)
        else:
            X = X.drop(featurename, axis=1)
    return X, y


def execute_feature_engineering(prediction_result):
    dataset_id = int(prediction_result["dataset - id"].values[0])
    model = prediction_result["model"].values[0]
    X, y = get_openml_dataset(dataset_id)
    target_label = "target"
    y = y.to_frame(target_label)
    X, y = get_additional_features(X, y, prediction_result)
    return X, y, dataset_id, model


def execute_feature_engineering_recursive(prediction_result, X, y):
    dataset_id = int(prediction_result["dataset - id"].values[0])
    model = prediction_result["model"].values[0]
    print("execute_feature_engineering_recursive")
    X, y = get_additional_features(X, y, prediction_result)
    return X, y, dataset_id, model



def main():
    print("Read Prediction Results")
    path = "../SurrogateModel/"
    files = os.listdir(path)
    core_files = []
    for file in files:
        if file.startswith("Best") and file.endswith('_0.parquet') or file.startswith("Best") and file.endswith('_1.parquet'):
            core_files.append(file)
    core_files.sort()
    for core_file in core_files:
        prediction_result = pd.read_parquet(path + core_file)
        print("Execute Feature Engineering")
        X, y, dataset_id, model = execute_feature_engineering(prediction_result)
        data = pd.concat([X, y], axis=1)
        data.to_parquet("FE_Dataset_" + str(dataset_id) + "_" + str(core_file))


if __name__ == "__main__":
    main()
