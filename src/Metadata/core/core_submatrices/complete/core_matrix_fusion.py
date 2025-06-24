import os
import pandas as pd

from src.utils.get_matrix import get_matrix_core_columns


def core_matrix_fusion(path1, path2):
    files = os.listdir(path1)
    core_files = []
    for file in files:
        if file.endswith('.parquet'):
            core_files.append(path1 + file)
    core_files.sort()

    columns = get_matrix_core_columns()
    result_matrix = pd.DataFrame(columns=columns)
    line_counter = 0
    for core_file in core_files:
        result_matrix_new = pd.read_parquet(core_file)
        result_matrix = pd.concat([result_matrix_new, result_matrix], ignore_index=True)
    result_matrix_old = pd.read_parquet(path2 + "Core_Matrix_Complete.parquet")
    print("Old Matrix Shape: " + str(result_matrix_old.shape))
    result_matrix.to_parquet(path2 + "Core_Matrix_Complete.parquet")
    print("New Matrix Shape: " + str(result_matrix.shape))


def main():
    core_matrix_fusion("../complete/", "../")


if __name__ == "__main__":
    main()
