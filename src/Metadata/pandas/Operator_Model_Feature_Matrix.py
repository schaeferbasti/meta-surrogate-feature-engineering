import numpy as np
import pandas as pd

from src.utils.create_feature_and_featurename import extract_operation_and_original_features
from src.utils.create_feature_and_featurename import create_feature_and_featurename
from src.utils.get_data import get_openml_dataset_split_and_metadata, get_all_amlb_dataset_ids
from src.utils.get_matrix import get_matrix_columns
from src.utils.get_metafeatures import get_numeric_pandas_metafeatures, get_categorical_pandas_metafeatures
from src.utils.get_operators import get_operators
from src.utils.run_models import run_autogluon_lgbm, get_model_score


def get_result(X_train, y_train, X_test, y_test, dataset_metadata, train_feature, test_feature, featurename, original_results):
    operator, _ = extract_operation_and_original_features(featurename)
    operator = str(operator)
    train_feature_df = pd.DataFrame(train_feature, columns=[featurename])
    feature_datatype = train_feature_df[featurename].dtype
    if pd.api.types.is_numeric_dtype(train_feature_df[featurename]):
        feature_metadata_numeric = get_numeric_pandas_metafeatures(train_feature_df, featurename)
        X_train_new = X_train.copy()
        X_train_new[feature_metadata_numeric["feature - name"]] = train_feature
        X_test_new = X_test.copy()
        X_test_new[feature_metadata_numeric["feature - name"]] = test_feature
        print("Run Autogluon with new Feature")
        lb = run_autogluon_lgbm(X_train_new, y_train, X_test_new, y_test)
        models = lb["model"]
        columns = get_matrix_columns()
        new_results = pd.DataFrame(columns=columns)
        for model in models:
            dataset = dataset_metadata["task_id"]
            original_score = np.abs(
                original_results.query("dataset == @dataset and model == @model", )['score'].values[0])
            modified_score = np.abs(lb.loc[lb['model'] == model, 'score_val'].values[0])
            relative_improvement = calc_relative_improvement(original_score, modified_score)
            new_results.loc[len(new_results)] = [
                dataset_metadata["task_id"],
                dataset_metadata["task_type"],
                dataset_metadata["number_of_classes"],
                feature_metadata_numeric["feature - name"],
                operator,
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
                relative_improvement
            ]
    else:
        feature_metadata_categorical = get_categorical_pandas_metafeatures(train_feature_df, featurename)
        X_train_new = X_train.copy()
        X_train_new[feature_metadata_categorical["feature - name"]] = train_feature
        X_test_new = X_test.copy()
        X_test_new[feature_metadata_categorical["feature - name"]] = test_feature
        print("Run Autogluon with new Feature")
        lb = run_autogluon_lgbm(X_train_new, y_train, X_test_new, y_test)
        models = lb["model"]
        columns = get_matrix_columns()
        new_results = pd.DataFrame(columns=columns)
        for model in models:
            dataset = dataset_metadata["task_id"]
            original_score = np.abs(
                original_results.query("dataset == @dataset and model == @model", )['score'].values[0])
            modified_score = np.abs(lb.loc[lb['model'] == model, 'score_val'].values[0])
            relative_improvement = calc_relative_improvement(original_score, modified_score)
            new_results.loc[len(new_results)] = [
                dataset_metadata["task_id"],
                dataset_metadata["task_type"],
                dataset_metadata["number_of_classes"],
                feature_metadata_categorical["feature - name"],
                operator,
                str(feature_datatype),
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


def main():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    columns = get_matrix_columns()
    result_matrix = pd.DataFrame(columns=columns)
    print("Result Matrix created")
    datasets = get_all_amlb_dataset_ids()
    unary_operators, binary_operators = get_operators()
    print("Iterate over Datasets")
    for dataset in datasets:
        X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset)
        original_results = get_model_score(X_train, y_train, X_test, y_test, dataset)
        for feature1 in X_train.columns:
            for feature2 in X_train.columns:
                for operator in binary_operators:
                    train_feature, featurename = create_feature_and_featurename(feature1=X_train[feature1],
                                                                          feature2=X_train[feature2], operator=operator)
                    test_feature, featurename = create_feature_and_featurename(feature1=X_test[feature1],
                                                                          feature2=X_test[feature2], operator=operator)
                    new_rows = get_result(X_train, y_train, X_test, y_test, dataset_metadata, train_feature, test_feature, featurename, original_results)
                    result_matrix = pd.concat([result_matrix, pd.DataFrame(new_rows)], ignore_index=True)
                    result_matrix.to_parquet("Operator_Model_Feature_Matrix_2_" + str(dataset) + ".parquet")
        for feature1 in X_train.columns:
            for operator in unary_operators:
                train_feature, featurename = create_feature_and_featurename(feature1=X_train[feature1], feature2=None,
                                                                      operator=operator)
                test_feature, featurename = create_feature_and_featurename(feature1=X_test[feature1], feature2=None,
                                                                            operator=operator)
                new_rows = get_result(X_train, y_train, X_test, y_test, dataset_metadata, train_feature, test_feature, featurename, original_results)
                result_matrix = pd.concat([result_matrix, pd.DataFrame(new_rows)], ignore_index=True)
                result_matrix.to_parquet("Operator_Model_Feature_Matrix_2_" + str(dataset) + ".parquet")
        result_matrix.to_parquet("Operator_Model_Feature_Matrix_2_" + str(dataset) + ".parquet")
    result_matrix.to_parquet("Operator_Model_Feature_Matrix_2.parquet")
    print("Final Result: \n" + str(result_matrix))


if __name__ == '__main__':
    main()
