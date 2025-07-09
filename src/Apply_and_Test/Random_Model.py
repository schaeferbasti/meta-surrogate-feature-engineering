import random
import pandas as pd

from src.utils.create_feature_and_featurename import create_featurenames, extract_operation_and_original_features, create_feature_and_featurename
from src.utils.get_data import get_openml_dataset_split_and_metadata, split_data, concat_data, get_openml_dataset


def get_additional_features(data, prediction_result):
    additional_feature_list = prediction_result
    for additional_feature in additional_feature_list:
        print("Add Feature: " + str(additional_feature))
        operation, original_features = extract_operation_and_original_features(additional_feature)
        if len(original_features) == 2:
            feature, _ = create_feature_and_featurename(data[original_features[0]], data[original_features[1]],
                                                        operation)
        else:
            feature, _ = create_feature_and_featurename(data[original_features[0]], None, operation)
        feature = pd.Series(feature).to_frame(additional_feature)
        data = pd.concat([data, feature], axis=1)
    return data


def execute_feature_engineering(prediction_result, model, dataset_id):
    X, y = get_openml_dataset(dataset_id)
    target_label = "target"
    y = y.to_frame(target_label)
    data = pd.concat([X, y], axis=1)
    print("Add Features to Data")
    data = get_additional_features(data, prediction_result)
    return data, dataset_id, model


def feature_addition_random(X_train, n_features_to_add, model, dataset_id):
    # Predict and split again
    feature_transformations = create_featurenames(X_train.columns)
    random_operations = random.sample(feature_transformations, n_features_to_add)
    data, _, _ = execute_feature_engineering(random_operations, model, dataset_id)
    X_train, y_train, X_test, y_test = split_data(data, "target")
    return X_train, y_train, X_test, y_test


def run_random_surrogate_model(dataset_id):
    method = "Random"
    print("Method: " + str(method) + ", Dataset: " + str(dataset_id))
    model = "LightGBM_BAG_L1"
    n_features_to_add = 10
    folds = 10
    for i in range(folds):
        X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset_id)
        X_train, y_train, X_test, y_test = feature_addition_random(X_train, n_features_to_add, model, dataset_id)
        data = concat_data(X_train, y_train, X_test, y_test, "target")
        data.to_parquet("test_data/FE_" + str(dataset_id) + "_" + str(method) + "_fold_" + str(i) + ".parquet")


if __name__ == '__main__':
    dataset_ids = [359983, 359987, 359992, 359993]
    for dataset_id in dataset_ids:
        run_random_surrogate_model(dataset_id)
