import glob

import pandas as pd

from src.utils.get_data import get_openfe_data
from src.utils.get_data import split_data, get_openml_dataset_split
from src.utils.run_models import get_result


def main():
    pattern = r'FE_Data_*.parquet'
    test_files = glob.glob(pattern)
    target_label = 'target'
    for test_file in test_files:
        result = pd.DataFrame(columns=['dataset', 'model', 'score'])
        print("Test Feature Engineering for " + str(test_file))
        dataset_id_and_model = test_file.split('FE_Data_')[-1].split('.')[0]
        dataset_id = dataset_id_and_model.split('_')[0]
        X_train, y_train, X_test, y_test = get_openml_dataset_split(int(dataset_id))
        original_results = get_result(X_train, y_train, dataset_id)
        result = pd.concat([result, original_results], ignore_index=True)
        print("Original Results: " + str(original_results))
        X_train, y_train, X_test, y_test = get_openfe_data(X_train, y_train, X_test, y_test)
        openfe_results = get_result(X_train, y_train, dataset_id)
        result = pd.concat([result, openfe_results], ignore_index=True)
        print("OpenFE Results: " + str(openfe_results))
        data = pd.read_parquet(test_file)
        X_train, y_train, X_test, y_test = split_data(data, target_label)
        my_fe_results = get_result(X_train, y_train, dataset_id)
        result = pd.concat([result, my_fe_results], ignore_index=True)
        print("My Results: " + str(my_fe_results))
        result.to_parquet("Result_" + test_file)


if __name__ == "__main__":
    main()
