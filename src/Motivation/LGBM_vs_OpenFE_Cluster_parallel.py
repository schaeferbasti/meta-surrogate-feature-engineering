import argparse
import os

import psutil

import pandas as pd

from OpenFE.openfe_parallel import OpenFE
from src.utils.get_dataset import get_openml_dataset_split
from src.utils.run_models import run_lgbm, run_autogluon_lgbm_cluster


def get_openfe_data(X_train, y_train, X_test, y_test, name):
    openFE = OpenFE()
    features = openFE.fit(data=X_train, label=y_train, n_jobs=1, name=name)  # generate new features
    X_train_openfe, X_test_openfe = openFE.transform(X_train, X_test, features, n_jobs=1)
    return X_train_openfe, y_train, X_test_openfe, y_test


def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 * 1024)  # Memory usage in MB
    print(f"Memory usage: {memory_usage:.2f} MB")


def main(args):
    log_memory_usage()
    dataset_id = args.dataset
    df = pd.DataFrame(columns=['Dataset', 'LGBM', 'OpenFE + LGBM', 'Autogluon LGBM', 'OpenFE + Autogluon LGBM', 'Tuned Autogluon LGBM', 'OpenFE + Tuned Autogluon LGBM'])
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        f.write("Dataset: " + str(dataset_id) + "\n")
    log_memory_usage()
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        f.write("Get OpenML Data\n")
    X_train, y_train, X_test, y_test = get_openml_dataset_split(dataset_id)
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        #pd.set_option('display.max_columns', None)
        f.write(str(X_train))
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        f.write("Factorize Data\n")
    X_train, y_train, X_test, y_test = factorize_data_old(X_train, y_train, X_test, y_test)
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        #pd.set_option('display.max_columns', None)
        f.write(str(X_train))
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        f.write("Use OpenFE\n")
    X_train_openfe, y_train_openfe, X_test_openfe, y_test_openfe = get_openfe_data(X_train, y_train, X_test, y_test,                                                                               str(dataset_id))
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        #pd.set_option('display.max_columns', None)
        f.write(str(X_train))
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        f.write("Start Experiments\n")
    log_memory_usage()
    lgbm_results = run_lgbm(X_train, y_train, X_test, y_test)
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        f.write("LGBM Results: " + str(lgbm_results) + "\n")
    log_memory_usage()
    lgbm_openfe_results = run_lgbm(X_train_openfe, y_train_openfe, X_test_openfe, y_test_openfe)
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        f.write("LGBM OpenFE Results: " + str(lgbm_openfe_results) + "\n")
    log_memory_usage()
    autogluon_lgbm_results = run_autogluon_lgbm_cluster(X_train, y_train, X_test, y_test, dataset_id, zeroshot=False)
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        f.write("Autogluon LGBM Results: " + str(autogluon_lgbm_results) + "\n")
    log_memory_usage()
    autogluon_lgbm_openfe_results = run_autogluon_lgbm_cluster(X_train_openfe, y_train_openfe, X_test_openfe, y_test_openfe, dataset_id,
                                                       zeroshot=False)
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        f.write("Autogluon LGBM OpenFE Results: " + str(autogluon_lgbm_openfe_results) + "\n")
    log_memory_usage()
    tuned_autogluon_lgbm_results = run_autogluon_lgbm_cluster(X_train, y_train, X_test, y_test, dataset_id, zeroshot=True)
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        f.write("Tuned Autogluon LGBM Results: " + str(tuned_autogluon_lgbm_results) + "\n")
    log_memory_usage()
    tuned_autogluon_lgbm_openfe_results = run_autogluon_lgbm_cluster(X_train_openfe, y_train_openfe, X_test_openfe,
                                                             y_test_openfe, dataset_id, zeroshot=True)
    with open("results_" + str(dataset_id) + ".txt", "a") as f:
        f.write("Tuned Autogluon LGBM OpenFE Results: " + str(tuned_autogluon_lgbm_openfe_results) + "\n")
    results_of_dataset = pd.Series({'Dataset': dataset_id, 'LGBM': lgbm_results, 'OpenFE + LGBM': lgbm_openfe_results, 'Autogluon LGBM': autogluon_lgbm_results, 'OpenFE + Autogluon LGBM': autogluon_lgbm_openfe_results, 'Tuned Autogluon LGBM': tuned_autogluon_lgbm_results, 'OpenFE + Tuned Autogluon LGBM': tuned_autogluon_lgbm_openfe_results})
    df.loc[len(df)] = results_of_dataset
    parquet_path = "results_" + str(dataset_id) + ".parquet"
    df.to_parquet(path=parquet_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Motivation')
    parser.add_argument('--dataset', type=int, required=True, help='Motivation dataset')
    args = parser.parse_args()
    main(args)
