import os
import pandas as pd

from src.utils.get_matrix import get_matrix_core_columns


def core_matrix_fusion():
    files = os.listdir()
    core_files = []
    for file in files:
        if file.endswith('.parquet'):
            core_files.append(file)
    core_files.sort()

    columns = get_matrix_core_columns()
    result_matrix = pd.DataFrame(columns=columns)
    line_counter = 0
    for core_file in core_files:
        result_matrix_new = pd.read_parquet(core_file)
        result_matrix = pd.concat([result_matrix_new, result_matrix], ignore_index=True)
    result_matrix.to_parquet("Core_Matrix_Complete.parquet")


def main():
    core_matrix_fusion()


if __name__ == "__main__":
    main()