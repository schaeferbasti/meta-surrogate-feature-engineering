import pandas as pd
import re

from src.utils.create_feature_and_featurename import create_feature_and_featurename
from src.utils.get_dataset import get_openml_dataset, split_data
from src.utils.run_autogluon import test_fe_for_model


def extract_operation_and_original_features(s):
    match = re.match(r"([^\s(]+)\s*\(([^)]+)\)", s)  # Capture operation and features
    if match:
        operation = match.group(1)  # The operation (before brackets)
        features = match.group(2).split(", ")  # The features inside brackets
        return operation, features
    return None, []


def get_additional_features(data, prediction_result):
    additional_feature_list = prediction_result['feature - name']
    for additional_feature in additional_feature_list:
        operation, original_features = extract_operation_and_original_features(additional_feature)
        if len(original_features) == 2:
            feature, _ = create_feature_and_featurename(original_features[0], original_features[1], operation)
        else:
            feature, _ = create_feature_and_featurename(original_features[0], None, operation)
        data._append(feature)
    return data


def execute_feature_engineering(prediction_result):
    dataset_id = prediction_result["dataset - id"].values[0]
    X, y = get_openml_dataset(dataset_id)
    target_label = y.columns[0]
    data = X.append(y, ignore_index=True, axis=1)
    data = get_additional_features(data, prediction_result)
    X_train, y_train, X_test, y_test = split_data(data, target_label)
    return X_train, y_train, X_test, y_test


def get_feature_engineered_data(prediction_result):
    X_train, y_train, X_test, y_test = execute_feature_engineering(prediction_result)
    return X_train, y_train, X_test, y_test


def test_feature_engineered_data_performance(X_train, y_train, X_test, y_test, model):
    train_data = X_train._append(y_train)
    target_label = y_train.columns[0]
    lb = test_fe_for_model(train_data, X_test, target_label, model)
    return lb.score()


def main():
    prediction_result = pd.read_parquet("Best_Prediction.parquet")
    X_train, y_train, X_test, y_test = get_feature_engineered_data(prediction_result)
    model = prediction_result["model"].values[0]
    results = test_feature_engineered_data_performance(X_train, y_train, X_test, y_test, model)
    print(results)


if __name__ == '__main__':
    main()
