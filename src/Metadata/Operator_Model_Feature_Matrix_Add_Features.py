import pandas as pd

from src.utils.get_data import get_openml_dataset_split_and_metadata
from src.utils.get_matrix import get_additional_numerical_columns, get_additional_categorical_columns
from src.utils.get_metafeatures import get_numeric_pandas_metafeatures, get_categorical_pandas_metafeatures


def add_columns(dataset_metadata, train_feature, feature, featurename, result_matrix):
    train_feature_df = pd.DataFrame(train_feature, columns=[featurename])
    feature_datatype = train_feature_df[featurename].dtype
    dataset_id = dataset_metadata["task_id"]
    if pd.api.types.is_numeric_dtype(train_feature_df[featurename]):
        feature_metadata_numeric = get_numeric_pandas_metafeatures(feature, featurename)
        columns = get_additional_numerical_columns(str(dataset_id), featurename)
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
        # new_columns = pd.DataFrame(columns=columns)
        # for row in result_matrix.iterrows():
            # if row[1].values[0] == int(dataset_id):
            #     new_columns = pd.concat([new_columns, pd.DataFrame(new_row)], ignore_index=True)
            # else:
            #     new_columns._append(pd.Series(), ignore_index=True)
        # Create an empty DataFrame with the same index as result_matrix and the new columns
        new_columns = pd.DataFrame(index=result_matrix.index, columns=columns)
        # Find the correct rows to fill
        matching_indices = result_matrix[result_matrix["dataset - id"] == int(dataset_id)].index
        # Fill only the matching row(s)
        for idx in matching_indices:
            new_columns.loc[idx] = new_row.iloc[0]
    else:
        feature_metadata_categorical = get_categorical_pandas_metafeatures(feature, featurename)
        columns = get_additional_categorical_columns(str(dataset_id), featurename)
        new_row = pd.DataFrame(columns=columns)
        new_row.loc[len(result_matrix)] = [
            feature_metadata_categorical["feature - name"],
            str(feature_datatype),
            int(feature_metadata_categorical["feature - count"]),
            feature_metadata_categorical["feature - unique"],
            feature_metadata_categorical["feature - top"],
            feature_metadata_categorical["feature - freq"],
        ]
        # new_columns = pd.DataFrame(columns=columns)
        # for row in result_matrix.iterrows():
            # if row[1].values[0] == int(dataset_id):
            #     new_columns = pd.concat([new_columns, pd.DataFrame(new_row)], ignore_index=True)
            # else:
            #     new_columns._append(pd.Series(), ignore_index=True)
        new_columns = pd.DataFrame(index=result_matrix.index, columns=columns)
        # Find the correct rows to fill
        matching_indices = result_matrix[result_matrix["dataset - id"] == int(dataset_id)].index
        # Fill only the matching row(s)
        for idx in matching_indices:
            new_columns.loc[idx] = new_row.iloc[0]
    insert_position = result_matrix.shape[1] - 2
    result_matrix = pd.concat([result_matrix.iloc[:, :insert_position], new_columns, result_matrix.iloc[:, insert_position:]], axis=1)
    return result_matrix


def main():
    result_matrix = pd.read_parquet("Operator_Model_Feature_Matrix_2.parquet")
    for dataset, _ in result_matrix.groupby('dataset - id'):
        print("Dataset: " + str(dataset))
        X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset)
        for featurename in X_train.columns:
            print("Feature: " + str(featurename))
            feature = pd.DataFrame(X_train[featurename])
            result_matrix = add_columns(dataset_metadata, X_train, feature, featurename, result_matrix)
            result_matrix.to_parquet("Operator_Model_Feature_Matrix_3_" + str(dataset) + ".parquet")
        result_matrix.to_parquet("Operator_Model_Feature_Matrix_3.parquet")
    print("Final Result: \n" + str(result_matrix))


if __name__ == '__main__':
    main()
