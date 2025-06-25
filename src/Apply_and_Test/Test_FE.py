import glob

import pandas as pd

from src.utils.get_data import get_openfe_data
from src.utils.get_data import split_data, get_openml_dataset_split
from src.utils.run_models import get_model_score_origin


def main():
    #pattern = r'FE_*.parquet'
    #test_files = glob.glob(pattern)
    target_label = 'target'
    #for test_file in test_files:
        #try:
        #    result = pd.read_parquet("Result_" + test_file)
        #    print(result)
        #except FileNotFoundError:
    dataset_id = 0
    result = pd.DataFrame(columns=['origin', 'dataset', 'model', 'score'])
    for result_file in glob.glob("FE_*.parquet"):
        print("Test Feature Engineering for " + str(result_file))
        dataset_id = result_file.split('FE_')[-1].split('_')[0]
        origin = result_file.split('FE_' + str(dataset_id) + '_')[0].split('_')[0]
        data = pd.read_parquet(result_file)
        X_train, y_train, X_test, y_test = split_data(data, target_label)
        my_fe_results = get_model_score_origin(X_train, y_train, X_test, y_test, dataset_id, origin)
        # my_fe_results['score'] = -my_fe_results['score']
        result = pd.concat([result, my_fe_results], ignore_index=True)
        print("My Results: " + str(my_fe_results))
        result.to_parquet("Result_" + result_file)
    # Original
    X_train, y_train, X_test, y_test = get_openml_dataset_split(int(dataset_id))
    original_results = get_model_score_origin(X_train, y_train, X_test, y_test, dataset_id, "Original")
    result = pd.concat([result, original_results], ignore_index=True)
    result.to_parquet("Result_" + result_file)
    print("Original Results: " + str(original_results))
    # OpenFE
    X_train, y_train, X_test, y_test = get_openfe_data(X_train, y_train, X_test, y_test)
    openfe_results = get_model_score_origin(X_train, y_train, X_test, y_test, dataset_id, "OpenFE")
    # openfe_results['score'] = -openfe_results['score']
    result = pd.concat([result, openfe_results], ignore_index=True)
    result.to_parquet("Result_" + result_file)
    print("OpenFE Results: " + str(openfe_results))


if __name__ == "__main__":
    main()
