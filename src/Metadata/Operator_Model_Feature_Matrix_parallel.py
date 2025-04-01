import argparse

import numpy as np
import pandas as pd

from src.utils.create_feature_and_featurename import create_feature_and_featurename
from src.utils.get_dataset import get_openml_dataset_split_and_metadata, get_all_amlb_dataset_ids
from src.utils.get_matrix import get_matrix_columns
from src.utils.get_metafeatures import get_numeric_pandas_metafeatures, get_categorical_pandas_metafeatures
from src.utils.get_operators import get_operators
from src.utils.run_models import run_autogluon_lgbm


def get_result(X_train, y_train, dataset_metadata, feature, featurename, original_results):
    feature_df = pd.DataFrame(feature, columns=[featurename])
    feature_datatype = feature_df[featurename].dtype
    if feature_datatype == np.number:
        feature_metadata_numeric = get_numeric_pandas_metafeatures(feature_df, featurename)
        X_train_new = X_train.copy()
        X_train_new[feature_metadata_numeric["feature - name"]] = feature
        print("Run Autogluon with new Feature")
        lb = run_autogluon_lgbm(X_train_new, y_train)
        models = lb["model"]
        columns = get_matrix_columns()
        new_results = pd.DataFrame(columns=columns)
        for model in models:
            dataset = dataset_metadata["task_id"]
            original_score = np.abs(
                original_results.query("dataset == @dataset and model == @model", )['score'].values[0])
            score_val = np.abs(lb.loc[lb['model'] == model, 'score_val'].values[0])
            improvement = score_val - original_score
            new_results.loc[len(new_results)] = [
                dataset_metadata["task_id"], dataset_metadata["task_type"],
                dataset_metadata["number_of_classes"],
                feature_metadata_numeric["feature - name"],
                str(feature_datatype),
                int(feature_metadata_numeric["feature - count"]),
                feature_metadata_numeric["feature - mean"],
                feature_metadata_numeric["feature - std"],
                feature_metadata_numeric["feature - min"],
                feature_metadata_numeric["feature - max"],
                feature_metadata_numeric["feature - lower percentile"],
                feature_metadata_numeric["feature - 50 percentile"],
                feature_metadata_numeric["feature - upper percentile"],
                None,
                None,
                None,
                model,
                improvement
            ]
    else:
        feature_metadata_categorical = get_categorical_pandas_metafeatures(feature_df, featurename)
        X_train_new = X_train.copy()
        X_train_new[feature_metadata_categorical["feature - name"]] = feature
        print("Run Autogluon with new Feature")
        lb = run_autogluon_lgbm(X_train_new, y_train)
        models = lb["model"]
        columns = get_matrix_columns()
        new_results = pd.DataFrame(columns=columns)
        for model in models:
            dataset = dataset_metadata["task_id"]
            original_score = np.abs(
                original_results.query("dataset == @dataset and model == @model", )['score'].values[0])
            score_val = np.abs(lb.loc[lb['model'] == model, 'score_val'].values[0])
            improvement = score_val - original_score
            new_results.loc[len(new_results)] = [
                dataset_metadata["task_id"], dataset_metadata["task_type"],
                dataset_metadata["number_of_classes"],
                feature_metadata_categorical["feature - name"], str(feature_datatype),
                int(feature_metadata_categorical["feature - count"]),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                feature_metadata_categorical["feature - unique"],
                feature_metadata_categorical["feature - top"],
                feature_metadata_categorical["feature - freq"],
                model,
                improvement
            ]
    print("Result for " + featurename + ": " + str(new_results))
    return new_results


def get_original_result(X_train, y_train, dataset_id):
    print("Run Autogluon with new Feature")
    lb = run_autogluon_lgbm(X_train, y_train)
    models = lb["model"]
    new_results = pd.DataFrame(columns=['dataset', 'model', 'score'])
    for model in models:
        score_val = lb.loc[lb['model'] == model, 'score_val'].values[0]
        new_results.loc[len(new_results)] = [dataset_id, model,
                                             score_val]  # None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    print("Result for original dataset: " + str(new_results))
    return new_results


def main(dataset):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    columns = get_matrix_columns()
    result_matrix = pd.DataFrame(columns=columns)
    print("Result Matrix created")
    unary_operators, binary_operators = get_operators()
    print("Iterate over Datasets")

    X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset)
    original_results = get_original_result(X_train, y_train, dataset)
    for feature1 in X_train.columns:
        for feature2 in X_train.columns:
            for operator in binary_operators:
                feature, featurename = create_feature_and_featurename(feature1=X_train[feature1],
                                                                      feature2=X_train[feature2], operator=operator)
                new_rows = get_result(X_train, y_train, dataset_metadata, feature, featurename, original_results)
                result_matrix = pd.concat([result_matrix, pd.DataFrame(new_rows)], ignore_index=True)
                result_matrix.to_parquet("Operator_Model_Feature_Matrix_2_" + str(dataset) + ".parquet")
    for feature1 in X_train.columns:
        for operator in unary_operators:
            feature, featurename = create_feature_and_featurename(feature1=X_train[feature1], feature2=None,
                                                                  operator=operator)
            new_rows = get_result(X_train, y_train, dataset_metadata, feature, featurename, original_results)
            result_matrix = pd.concat([result_matrix, pd.DataFrame(new_rows)], ignore_index=True)
            result_matrix.to_parquet("Operator_Model_Feature_Matrix_2_" + str(dataset) + ".parquet")
    result_matrix.to_parquet("Operator_Model_Feature_Matrix_2_" + str(dataset) + ".parquet")
    result_matrix.to_parquet("Operator_Model_Feature_Matrix_2.parquet")
    print("Final Result: \n" + str(result_matrix))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Metadata')
    parser.add_argument('--dataset', type=int, required=True, help='Metadata dataset')
    args = parser.parse_args()
    main(args.dataset)
