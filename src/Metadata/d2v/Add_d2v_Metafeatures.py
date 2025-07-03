import time

import pandas as pd

from src.utils.get_data import get_openml_dataset_split_and_metadata, get_name_and_split_and_save_dataset
from src.utils.get_matrix import get_additional_d2v_columns
from src.utils.get_metafeatures import get_d2v_metafeatures


def add_d2v_metadata_columns(dataset_metadata, X_train, result_matrix):
    columns = get_additional_d2v_columns()
    metafeatures = get_d2v_metafeatures(dataset_metadata["task_id"])
    new_columns = pd.DataFrame(index=result_matrix.index, columns=columns)
    for row in result_matrix.iterrows():
        featurename = row[1][1]
        matching_indices = result_matrix[result_matrix["feature - name"] == str(featurename)].index
        for idx in matching_indices:
            new_columns.loc[idx] = metafeatures.iloc[0]
    insert_position = result_matrix.shape[1] - 2
    result_matrix = pd.concat([result_matrix.iloc[:, :insert_position], new_columns, result_matrix.iloc[:, insert_position:]], axis=1)
    return result_matrix


def main():
    result_matrix = pd.read_parquet("src/Metadata/core/Core_Matrix_Complete.parquet")
    columns = get_additional_d2v_columns()
    result_matrix_columns = result_matrix.columns.values.tolist()
    columns = columns + result_matrix_columns
    result_matrix_pandas = pd.DataFrame(columns=columns)
    start = time.time()
    counter = 0
    for dataset, _ in result_matrix.groupby('dataset - id'):
        get_name_and_split_and_save_dataset(dataset)
        print("Dataset: " + str(dataset))
        try:
            pd.read_parquet("src/Metadata/d2v/D2V_Matrix_Complete" + str(dataset) + ".parquet")
        except FileNotFoundError:
            try:
                counter += 1
                start_dataset = time.time()
                X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(int(str(dataset)))
                result_matrix_dataset = result_matrix[result_matrix['dataset - id'] == dataset]
                result_matrix_dataset = add_d2v_metadata_columns(dataset_metadata, X_train, result_matrix_dataset)
                result_matrix_pandas.columns = result_matrix_dataset.columns
                result_matrix_pandas = pd.concat([result_matrix_pandas, result_matrix_dataset], axis=0)
                result_matrix_pandas.to_parquet("src/Metadata/d2v/D2V_Matrix_Complete" + str(dataset) + ".parquet")
                end_dataset = time.time()
                print("Time for d2v on Dataset " + str(dataset) + ": " + str(end_dataset - start_dataset))
            except TypeError:
                continue
    result_matrix_pandas.to_parquet("src/Metadata/d2v/D2V_Matrix_Complete.parquet")
    end = time.time()
    print("Time for complete d2v MF: " + str(end - start) + " on " + str(counter) + " datasets.")


if __name__ == '__main__':
    main()
