import argparse

import pandas as pd

from src.Metadata.Operator_Model_Feature_Matrix import get_result
from src.utils.create_feature_and_featurename import create_feature_and_featurename
from src.utils.get_data import get_openml_dataset_split_and_metadata
from src.utils.get_matrix import get_matrix_columns
from src.utils.get_operators import get_operators
from src.utils.run_models import get_original_result


def main(dataset):
    columns = get_matrix_columns()
    result_matrix = pd.DataFrame(columns=columns)
    unary_operators, binary_operators = get_operators()
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Metadata')
    parser.add_argument('--dataset', type=int, required=True, help='Metadata dataset')
    args = parser.parse_args()
    main(args.dataset)
