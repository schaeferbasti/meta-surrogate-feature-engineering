import os

import pandas as pd

from src.utils.get_operators import get_operators


def calc_indexes_to_delete(result_matrix_len, number_features, number_unary_operators):
    indexes = []
    index = result_matrix_len - 1
    for i in range(number_features):
        indexes.append(index)
        index = index - number_unary_operators
    return indexes


def main():
    files = os.listdir()
    core_files = []
    for file in files:
        if file.endswith('.parquet'):
            core_files.append(file)

    for core_file in core_files:
        result_matrix = pd.read_parquet(core_file)
        result_matrix_copy = result_matrix.copy()
        try:
            result_matrix_len = result_matrix.shape[0]
            print(result_matrix_len)
            result_matrix.count("columns")
            number_features = result_matrix.groupby('operator').count()["dataset - id"]
            number_features = number_features.loc["delete"]
            print(number_features)
            unary_operators, binary_operators = get_operators()
            number_unary_operators = len(unary_operators) + 1
            print(number_unary_operators)
            indexes = calc_indexes_to_delete(result_matrix_len, number_features, number_unary_operators)
            print(indexes)
            for index in indexes:
                result_matrix = result_matrix.drop(index, axis=0)
            result_matrix = result_matrix.reset_index(drop=True)
            result_matrix.to_parquet(core_file)
        except KeyError:
            result_matrix_copy.to_parquet(core_file)



if __name__ == "__main__":
    main()
