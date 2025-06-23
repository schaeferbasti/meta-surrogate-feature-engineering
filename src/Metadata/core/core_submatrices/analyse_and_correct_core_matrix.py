import os
import pandas as pd

from src.utils.get_data import get_openml_dataset
from src.utils.get_operators import get_operators


def remove_duplicate_columns():
    files = os.listdir()
    core_files = []
    for file in files:
        if file.endswith('.parquet'):
            core_files.append(file)

    for core_file in core_files:
        print(core_file)
        result_matrix = pd.read_parquet(core_file)
        result_matrix.drop_duplicates(inplace=True)
        result_matrix.to_parquet(core_file)


def list_incomplete_df():
    files = os.listdir()
    core_files = []
    for file in files:
        if file.endswith('.parquet'):
            core_files.append(file)
    core_files.sort()

    list_of_incomplete_df = []
    for core_file in core_files:
        print(core_file)
        result_matrix = pd.read_parquet(core_file)
        dataset_id = core_file.split('.')[0].split('Core')[-1]
        unary_operators, binary_operators = get_operators()
        list_of_operators = unary_operators + binary_operators + ["delete", "add", "subtract", "multiply", "divide"]
        list_of_operators.remove("+")
        list_of_operators.remove("-")
        list_of_operators.remove("*")
        list_of_operators.remove("/")
        print("Present number of lines: " + str(result_matrix.shape[0]))
        X, y = get_openml_dataset(int(dataset_id))
        n_features = len(X.columns)
        print("Calculated number of lines of matrix: " + str(n_features * n_features * 13 + n_features * 10 + n_features))
        if set(list_of_operators).issubset(set(result_matrix['operator'])):
            print("✅ All required operators are present.")
            result_matrix.to_parquet("complete/" + core_file)
        else:
            print("❌ Some required operators are missing.")
            list_of_incomplete_df.append(dataset_id)
    print("Number of Saved Files: " + str(len(core_files)))
    print("Number of Incomplete Files: " + str(len(list_of_incomplete_df)))
    print("Number of Complete Files: " + str(len(core_files) - len(list_of_incomplete_df)))
    print("List of Incomplete Files: " + str(list_of_incomplete_df))

def main():
    remove_duplicate_columns()
    list_incomplete_df()


if __name__ == "__main__":
    main()
