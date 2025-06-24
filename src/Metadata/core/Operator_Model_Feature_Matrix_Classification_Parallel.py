import argparse

import numpy as np
import pandas as pd

from src.utils.create_feature_and_featurename import create_feature_and_featurename, extract_operation_and_original_features
from src.utils.get_data import get_openml_dataset_split_and_metadata, get_openml_dataset
from src.utils.get_matrix import get_matrix_core_columns
from src.utils.get_operators import get_operators
from src.utils.run_models import get_model_score, run_autogluon_lgbm_classification


def check_if_complete(result_matrix):
    check = False
    dataset_id = result_matrix["dataset - id"].values[0]
    X, y = get_openml_dataset(int(dataset_id))
    n_features = len(X.columns)
    n_lines_calculated = n_features * n_features * 13 + n_features * 10 + n_features
    n_lines_present = result_matrix.shape[0]
    if n_lines_present == n_lines_calculated:
        check = True
    return check


def check_if_operator_is_there(result_matrix, operator):
    check = False
    dataset_id = result_matrix["dataset - id"].values[0]
    X, y = get_openml_dataset(int(dataset_id))
    n_features = len(X.columns)
    n_lines_calculated = n_features
    operator_count = result_matrix['operator'].value_counts()
    try:
        n_lines_present = operator_count[operator]
        if n_lines_present == n_lines_calculated:
            check = True
    except KeyError:
        return False
    return check


def get_core_result_feature_generation_classification(X_train, y_train, X_test, y_test, dataset_metadata, train_feature, test_feature, featurename, original_results):
    operator, _ = extract_operation_and_original_features(featurename)
    operator = str(operator)
    X_train_new = X_train.copy()
    X_train_new[featurename] = train_feature
    X_test_new = X_test.copy()
    X_test_new[featurename] = test_feature
    lb = run_autogluon_lgbm_classification(X_train_new, y_train, X_test_new, y_test)
    models = lb["model"]
    columns = get_matrix_core_columns()
    new_results = pd.DataFrame(columns=columns)
    for model in models:
        dataset = dataset_metadata["task_id"]
        original_score = np.abs(
            original_results.query("dataset == @dataset and model == @model", )['score'].values[0])
        modified_score = np.abs(lb.loc[lb['model'] == model, 'score_val'].values[0])
        relative_improvement = calc_relative_improvement(original_score, modified_score)
        new_results.loc[len(new_results)] = [
            dataset_metadata["task_id"],
            featurename,
            operator,
            model,
            relative_improvement
        ]
    print("Result for " + featurename + ": " + str(new_results))
    return new_results


def get_core_result_feature_selection_classification(X_train, y_train, X_test, y_test, dataset_metadata, featurename, original_results):
    operator = "delete"
    lb = run_autogluon_lgbm_classification(X_train, y_train, X_test, y_test)
    models = lb["model"]
    columns = get_matrix_core_columns()
    new_results = pd.DataFrame(columns=columns)
    for model in models:
        dataset = dataset_metadata["task_id"]
        original_score = np.abs(original_results.query("dataset == @dataset and model == @model", )['score'].values[0])
        modified_score = np.abs(lb.loc[lb['model'] == model, 'score_val'].values[0])
        relative_improvement = calc_relative_improvement(original_score, modified_score)
        new_results.loc[len(new_results)] = [
            dataset_metadata["task_id"],
            featurename,
            operator,
            model,
            relative_improvement
        ]
    print("Result for " + featurename + ": " + str(new_results))
    return new_results


def calc_relative_improvement(original_score, modified_score):
    if np.isclose(original_score, modified_score):
        return 0.0
    elif original_score > modified_score:
        return (original_score - modified_score) / original_score
    else:
        return (original_score - modified_score) / modified_score


def continue_calculating_improvement_classification(result_matrix, dataset, path):
    unary_operators, binary_operators = get_operators()
    X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset)
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()
    original_results = get_model_score(X_train, y_train, X_test, y_test, dataset)
    if check_if_operator_is_there(result_matrix, "delete"):
        print("Result matrix contains all delete operations")
    else:
        for feature1 in X_train_copy.columns:
            X_train_reduced = X_train_copy.drop(feature1, axis=1)
            X_test_reduced = X_test_copy.drop(feature1, axis=1)
            featurename = "without - " + str(feature1)
            new_rows = get_core_result_feature_selection_classification(X_train_reduced, y_train, X_test_reduced, y_test,
                                                                        dataset_metadata, featurename, original_results)
            result_matrix = pd.concat([result_matrix, pd.DataFrame(new_rows)], ignore_index=True)
            result_matrix.to_parquet(path + "Operator_Model_Feature_Matrix_Core" + str(dataset) + ".parquet")
    missing_binary_operators = []
    for binary_operator in binary_operators:
        if binary_operator == "+":
            binary_operator = "add"
        if binary_operator == "-":
            binary_operator = "subtract"
        if binary_operator == "*":
            binary_operator = "multiply"
        if binary_operator == "/":
            binary_operator = "/"
        if check_if_operator_is_there(result_matrix, binary_operator):
            print("Result matrix contains all delete operations")
        else:
            missing_binary_operators.append(binary_operator)
    for feature1 in X_train_copy.columns:
        for feature2 in X_train_copy.columns:
            for operator in missing_binary_operators:
                train_feature, featurename = create_feature_and_featurename(feature1=X_train[feature1], feature2=X_train[feature2], operator=operator)
                test_feature, featurename = create_feature_and_featurename(feature1=X_test[feature1], feature2=X_test[feature2], operator=operator)
                new_rows = get_core_result_feature_generation_classification(X_train, y_train, X_test, y_test, dataset_metadata, train_feature, test_feature, featurename, original_results)
                result_matrix = pd.concat([result_matrix, pd.DataFrame(new_rows)], ignore_index=True)
                result_matrix.to_parquet(path + "Operator_Model_Feature_Matrix_Core" + str(dataset) + ".parquet")
    missing_unary_operators = []
    for unary_operator in unary_operators:
        if check_if_operator_is_there(result_matrix, unary_operator):
            print("Result matrix contains all delete operations")
        else:
            missing_unary_operators.append(unary_operator)
    for feature1 in X_train_copy.columns:
        for operator in missing_unary_operators:
            train_feature, featurename = create_feature_and_featurename(feature1=X_train[feature1], feature2=None, operator=operator)
            test_feature, featurename = create_feature_and_featurename(feature1=X_test[feature1], feature2=None, operator=operator)
            new_rows = get_core_result_feature_generation_classification(X_train, y_train, X_test, y_test, dataset_metadata, train_feature, test_feature, featurename, original_results)
            result_matrix = pd.concat([result_matrix, pd.DataFrame(new_rows)], ignore_index=True)
            result_matrix.to_parquet(path + "Operator_Model_Feature_Matrix_Core" + str(dataset) + ".parquet")
        result_matrix.to_parquet(path + "Operator_Model_Feature_Matrix_Core" + str(dataset) + ".parquet")
    result_matrix.to_parquet(path + "Operator_Model_Feature_Matrix_Core" + str(dataset) + ".parquet")


def main(dataset):
    print("Classification Dataset: " + str(dataset))
    path = "src/Metadata/core/core_submatrices/"
    try:
        result_matrix = pd.read_parquet(path + "Operator_Model_Feature_Matrix_Core" + str(dataset) + ".parquet")
        check = check_if_complete(result_matrix)
        if check:
            print("Dataset " + str(dataset) + " is complete")
        else:
            print("Dataset " + str(dataset) + " is NOT complete")
            continue_calculating_improvement_classification(result_matrix, dataset, path)
    except FileNotFoundError:
        print("Dataset " + str(dataset) + " is NOT existant")
        columns = get_matrix_core_columns()
        result_matrix = pd.DataFrame(columns=columns)
        unary_operators, binary_operators = get_operators()
        X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset)
        X_train_copy = X_train.copy()
        X_test_copy = X_test.copy()
        original_results = get_model_score(X_train, y_train, X_test, y_test, dataset)
        for feature1 in X_train_copy.columns:
            X_train_reduced = X_train_copy.drop(feature1, axis=1)
            X_test_reduced = X_test_copy.drop(feature1, axis=1)
            featurename = "without - " + str(feature1)
            new_rows = get_core_result_feature_selection_classification(X_train_reduced, y_train, X_test_reduced, y_test, dataset_metadata, featurename, original_results)
            result_matrix = pd.concat([result_matrix, pd.DataFrame(new_rows)], ignore_index=True)
            result_matrix.to_parquet(path + "Operator_Model_Feature_Matrix_Core" + str(dataset) + ".parquet")
        for feature1 in X_train_copy.columns:
            for feature2 in X_train_copy.columns:
                for operator in binary_operators:
                    train_feature, featurename = create_feature_and_featurename(feature1=X_train[feature1], feature2=X_train[feature2], operator=operator)
                    test_feature, featurename = create_feature_and_featurename(feature1=X_test[feature1], feature2=X_test[feature2],  operator=operator)
                    new_rows = get_core_result_feature_generation_classification(X_train, y_train, X_test, y_test, dataset_metadata, train_feature, test_feature, featurename, original_results)
                    result_matrix = pd.concat([result_matrix, pd.DataFrame(new_rows)], ignore_index=True)
                    result_matrix.to_parquet(path + "Operator_Model_Feature_Matrix_Core" + str(dataset) + ".parquet")
        for feature1 in X_train_copy.columns:
            for operator in unary_operators:
                train_feature, featurename = create_feature_and_featurename(feature1=X_train[feature1], feature2=None, operator=operator)
                test_feature, featurename = create_feature_and_featurename(feature1=X_test[feature1], feature2=None, operator=operator)
                new_rows = get_core_result_feature_generation_classification(X_train, y_train, X_test, y_test, dataset_metadata, train_feature, test_feature, featurename, original_results)
                result_matrix = pd.concat([result_matrix, pd.DataFrame(new_rows)], ignore_index=True)
                result_matrix.to_parquet(path + "Operator_Model_Feature_Matrix_Core" + str(dataset) + ".parquet")
            result_matrix.to_parquet(path + "Operator_Model_Feature_Matrix_Core" + str(dataset) + ".parquet")
        result_matrix.to_parquet(path + "Operator_Model_Feature_Matrix_Core" + str(dataset) + ".parquet")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Metadata Classification')
    parser.add_argument('--dataset_classification', type=int, required=True, help='AMLB Classification Dataset')
    args = parser.parse_args()
    main(args.dataset_classification)
