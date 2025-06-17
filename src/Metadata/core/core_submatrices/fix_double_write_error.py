import os
import pandas as pd


def main():
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


if __name__ == "__main__":
    main()
