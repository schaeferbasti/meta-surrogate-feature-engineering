import time

import pandas as pd

from src.utils.create_feature_and_featurename import create_feature
from src.utils.get_data import get_openml_dataset_split_and_metadata
from src.utils.get_matrix import get_additional_pandas_columns
from src.utils.get_metafeatures import get_pandas_metafeatures


def add_pandas_metadata_columns(dataset_metadata, X_train, result_matrix):
    columns = get_additional_pandas_columns()
    new_columns = pd.DataFrame(index=result_matrix.index, columns=columns)
    for row in result_matrix.iterrows():
        dataset = row[1][0]
        featurename = row[1][1]
        X_train_copy = X_train.copy()
        if featurename.startswith("without"):
            feature_to_delete = featurename.split(" - ")[1]
            X_train_copy = X_train_copy.drop(feature_to_delete, axis=1)
        else:
            features = featurename.split("(")[1].replace(")", "").replace(" ", "")
            if "," in features:
                featurename1 = features.split(",")[0]
                feature1 = X_train_copy[featurename1]
                featurename2 = features.split(",")[1]
                feature2 = X_train_copy[featurename2]
            else:
                featurename1 = features
                feature1 = X_train_copy[featurename1]
                feature2 = None
            new_feature = create_feature(feature1, feature2, featurename)
            new_feature_df = pd.DataFrame(new_feature, columns=[featurename])
            X_train_copy = pd.concat([X_train_copy, new_feature_df])
        try:
            feature = pd.DataFrame(X_train_copy[featurename])
            feature_metadata = get_pandas_metafeatures(feature, featurename)
            new_row = pd.DataFrame(columns=columns)
            new_row.loc[len(result_matrix)] = [
                dataset_metadata["task_type"],
                feature_metadata["feature - count"],
                feature_metadata["feature - unique"],
                feature_metadata["feature - top"],
                feature_metadata["feature - freq"],
                feature_metadata["feature - mean"],
                feature_metadata["feature - std"],
                feature_metadata["feature - min"],
                feature_metadata["feature - 25"],
                feature_metadata["feature - 50"],
                feature_metadata["feature - 75"],
                feature_metadata["feature - max"],
            ]
            matching_indices = result_matrix[result_matrix["feature - name"] == str(featurename)].index
            for idx in matching_indices:
                new_columns.loc[idx] = new_row.iloc[0]
        except KeyError:
            feature = pd.DataFrame(X_train[feature_to_delete])
            feature_metadata = get_pandas_metafeatures(feature, feature_to_delete)
            new_row = pd.DataFrame(columns=columns)
            new_row.loc[len(result_matrix)] = [
                dataset_metadata["task_type"],
                feature_metadata["feature - count"],
                feature_metadata["feature - unique"],
                feature_metadata["feature - top"],
                feature_metadata["feature - freq"],
                feature_metadata["feature - mean"],
                feature_metadata["feature - std"],
                feature_metadata["feature - min"],
                feature_metadata["feature - 25"],
                feature_metadata["feature - 50"],
                feature_metadata["feature - 75"],
                feature_metadata["feature - max"],
            ]
            matching_indices = result_matrix[result_matrix["feature - name"] == str(featurename)].index
            for idx in matching_indices:
                new_columns.loc[idx] = new_row.iloc[0]
    insert_position = result_matrix.shape[1] - 2
    result_matrix = pd.concat([result_matrix.iloc[:, :insert_position], new_columns, result_matrix.iloc[:, insert_position:]], axis=1)
    return result_matrix


def main():
    result_matrix = pd.read_parquet("src/Metadata/core/Core_Matrix_Complete.parquet")
    columns = get_additional_pandas_columns()
    result_matrix_columns = result_matrix.columns.values.tolist()
    columns = columns + result_matrix_columns
    result_matrix_pandas = pd.DataFrame(columns=columns)
    start = time.time()
    counter = 0
    for dataset, _ in result_matrix.groupby('dataset - id'):
        print("Dataset: " + str(dataset))
        try:
            pd.read_parquet("src/Metadata/pandas/Pandas_Matrix_Complete" + str(dataset) + ".parquet")
        except FileNotFoundError:
            counter += 1
            start_dataset = time.time()
            X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(int(str(dataset)))
            result_matrix_dataset = result_matrix[result_matrix['dataset - id'] == dataset]
            result_matrix_dataset = add_pandas_metadata_columns(dataset_metadata, X_train, result_matrix_dataset)
            result_matrix_pandas.columns = result_matrix_dataset.columns
            result_matrix_pandas = pd.concat([result_matrix_pandas, result_matrix_dataset], axis=0)
            result_matrix_pandas.to_parquet("src/Metadata/pandas/Pandas_Matrix_Complete" + str(dataset) + ".parquet")
            end_dataset = time.time()
            print("Time for Pandas on Dataset " + str(dataset) + ": " + str(end_dataset - start_dataset))
    result_matrix_pandas.to_parquet("src/Metadata/pandas/Pandas_Matrix_Complete.parquet")
    end = time.time()
    print("Time for complete Pandas MF: " + str(end - start) + " on " + str(counter) + " datasets.")


if __name__ == '__main__':
    main()
