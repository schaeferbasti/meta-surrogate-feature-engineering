import pandas as pd

from src.utils.create_feature_and_featurename import create_feature_and_featurename
from src.utils.get_data import get_openml_dataset, split_data, concat_data
from src.utils.preprocess_data import factorize_dataset, factorize_transformed_dataset
from src.utils.create_feature_and_featurename import extract_operation_and_original_features

import warnings
warnings.filterwarnings("ignore")


def get_additional_features(data, prediction_result):
    additional_feature_list = prediction_result['feature - name']
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


def execute_feature_engineering(prediction_result):
    dataset_id = int(prediction_result["dataset - id"].values[0])
    model = prediction_result["model"].values[0]
    print("Get original Dataset")
    X, y = get_openml_dataset(dataset_id)
    target_label = "target"
    y = y.to_frame(target_label)
    data = pd.concat([X, y], axis=1)
    print("Factorize Data")
    X_train, y_train, X_test, y_test = split_data(data, target_label)
    X_train, y_train, X_test, y_test = factorize_dataset(X_train, y_train, X_test, y_test)
    data = concat_data(X_train, y_train, X_test, y_test, target_label)
    print("Add Features to Data")
    try:
        data = pd.read_parquet("FE_Data_" + str(dataset_id) + ".parquet")
    except FileNotFoundError:
        data = get_additional_features(data, prediction_result)
    return data, dataset_id, model



def main():
    print("Read Prediction Results")
    prediction_result = pd.read_parquet("../SurrogateModel/Best_Operations.parquet")
    print("Execute Feature Engineering")
    data, dataset_id, model = execute_feature_engineering(prediction_result)
    data.to_parquet("FE_Data_" + str(dataset_id) + ".parquet")


if __name__ == "__main__":
    main()
