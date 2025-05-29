import pandas as pd

from src.utils.get_data import get_openml_dataset_split_and_metadata
from src.utils.get_matrix import get_additional_numerical_columns, get_additional_categorical_columns
from src.utils.get_metafeatures import get_numeric_pandas_metafeatures, get_categorical_pandas_metafeatures


def add_pandas_metadata_columns(X_train, result_matrix):
    for row in result_matrix.iterrows():
        dataset = row[1][0]
        featurename = row[1][1]
        try:
            feature = pd.DataFrame(X_train[featurename])
            feature_datatype = feature.dtype
            if pd.api.types.is_numeric_dtype(X_train[featurename]):
                feature_metadata_numeric = get_numeric_pandas_metafeatures(feature, featurename)
                columns = get_additional_numerical_columns(str(dataset), featurename)
                new_columns = pd.DataFrame(index=result_matrix.index, columns=columns)
                new_row = pd.DataFrame(columns=columns)
                new_row.loc[len(result_matrix)] = [
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
                ]
                matching_indices = result_matrix[result_matrix["dataset - id"] == int(dataset)].index
                for idx in matching_indices:
                    new_columns.loc[idx] = new_row.iloc[0]
            else:
                feature_metadata_categorical = get_categorical_pandas_metafeatures(feature, featurename)
                columns = get_additional_categorical_columns(str(dataset), featurename)
                new_columns = pd.DataFrame(index=result_matrix.index, columns=columns)
                new_row = pd.DataFrame(columns=columns)
                new_row.loc[len(result_matrix)] = [
                    feature_metadata_categorical["feature - name"],
                    str(feature_datatype),
                    int(feature_metadata_categorical["feature - count"]),
                    feature_metadata_categorical["feature - unique"],
                    feature_metadata_categorical["feature - top"],
                    feature_metadata_categorical["feature - freq"],
                ]
                matching_indices = result_matrix[result_matrix["dataset - id"] == int(dataset)].index
                for idx in matching_indices:
                    new_columns.loc[idx] = new_row.iloc[0]
        except KeyError:
            columns = get_additional_categorical_columns(str(dataset), featurename)
            new_columns = pd.DataFrame(index=result_matrix.index, columns=columns)
            new_row = pd.DataFrame(columns=columns)
            new_row.loc[len(result_matrix)] = [
                None,
                None,
                None,
                None,
                None,
                None,
            ]
            matching_indices = result_matrix[result_matrix["dataset - id"] == int(dataset)].index
            for idx in matching_indices:
                new_columns.loc[idx] = new_row.iloc[0]
    insert_position = result_matrix.shape[1] - 2
    result_matrix = pd.concat([result_matrix.iloc[:, :insert_position], new_columns, result_matrix.iloc[:, insert_position:]], axis=1)
    return result_matrix


def main():
    result_matrix = pd.read_parquet("../core/Core_Matrix_Example.parquet")
    for dataset, _ in result_matrix.groupby('dataset - id'):
        print("Dataset: " + str(dataset))
        X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset)
        result_matrix = add_pandas_metadata_columns(X_train, result_matrix)
        result_matrix.to_parquet("Pandas_Matrix_4_" + str(dataset) + ".parquet")
    result_matrix.to_parquet("Pandas_Matrix_4.parquet")


if __name__ == '__main__':
    main()
