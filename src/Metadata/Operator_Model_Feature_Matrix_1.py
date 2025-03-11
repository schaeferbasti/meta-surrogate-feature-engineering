import math

import requests
import yaml

import numpy as np
import pandas as pd

import openml
from pymfe.mfe import MFE

from autogluon.tabular.src.autogluon.tabular import TabularPredictor
from tabrepo_2024_custom import zeroshot2024


def get_all_amlb_dataset_ids():
    # Code from https://github.com/openml/automlbenchmark/blob/2c2a93dc3fc65fc3d6a77fe97ec9df1108551075/scripts/find_matching_datasets.py
    small_config_url = "https://raw.githubusercontent.com/openml/automlbenchmark/2c2a93dc3fc65fc3d6a77fe97ec9df1108551075/resources/benchmarks/small.yaml"
    #medium_config_url = "https://raw.githubusercontent.com/openml/automlbenchmark/2c2a93dc3fc65fc3d6a77fe97ec9df1108551075/resources/benchmarks/medium.yaml"
    #large_config_url = "https://raw.githubusercontent.com/openml/automlbenchmark/2c2a93dc3fc65fc3d6a77fe97ec9df1108551075/resources/benchmarks/large.yaml"
    small_configuration = yaml.load(requests.get(small_config_url).text, Loader=yaml.Loader)
    #medium_configuration = yaml.load(requests.get(medium_config_url).text, Loader=yaml.Loader)
    #large_configuration = yaml.load(requests.get(large_config_url).text, Loader=yaml.Loader)
    benchmark_tids = set(
        [problem.get("openml_task_id") for problem in small_configuration]
        # + [problem.get("openml_task_id") for problem in medium_configuration]
        # + [problem.get("openml_task_id") for problem in large_configuration]
    )
    return benchmark_tids


def get_openml_dataset(openml_task_id: int) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    task = openml.tasks.get_task(
        openml_task_id,
        download_splits=True,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    train_idx, test_idx = task.get_train_test_split_indices()
    X, y = task.get_X_and_y(dataset_format="dataframe")  # type: ignore
    train_x, train_y = X.iloc[train_idx], y.iloc[train_idx]
    test_x, test_y = X.iloc[test_idx], y.iloc[test_idx]
    return train_x, train_y, test_x, test_y


def get_pymfe_metafeatures(feature):
    pymfe = MFE()
    pymfe.fit(np.array(feature))
    metafeatures = pymfe.extract()
    return metafeatures


def create_feature_and_featurename(feature1, feature2, operator, ):
    if feature2 is None:
        feature, featurename = create_unary_feature_and_featurename(feature1, operator)
    else:
        feature, featurename = create_binary_feature_and_featurename(feature1, feature2, operator)
    return feature, featurename


def create_unary_feature_and_featurename(feature1, operator):
    feature1_int_list = [int(x) for x in feature1]
    if operator == "min":
        feature = feature1.apply(lambda x: min(feature1_int_list))
        featurename = "min(" + str(feature1.name) + ")"
    if operator == "max":
        feature = feature1.apply(lambda x: max(feature1_int_list))
        featurename = "max(" + str(feature1.name) + ")"
    if operator == "freq":
        feature = feature1.apply(lambda x: feature1_int_list.count(int(x)))
        featurename = "freq(" + str(feature1.name) + ")"
    if operator == "abs":
        feature = feature1.apply(lambda x: abs(float(x)))
        featurename = "abs(" + str(feature1.name) + ")"
    if operator == "log":
        feature = feature1.apply(lambda x: np.log(float(x)))
        featurename = "log(" + str(feature1.name) + ")"
    if operator == "sqrt":
        feature = feature1.apply(lambda x: np.sqrt(float(x)))
        featurename = "sqrt(" + str(feature1.name) + ")"
    if operator == "square":
        feature = feature1.apply(lambda x: int(x) ^ 2)
        featurename = "square(" + str(feature1.name) + ")"
    if operator == "sigmoid":
        feature = feature1.apply(lambda x: 1 / (1 + math.exp(-float(x))))
        featurename = "sigmoid(" + str(feature1.name) + ")"
    return feature, featurename


def create_binary_feature_and_featurename(feature1, feature2, operator):
    feature1_int_list = [int(x) for x in feature1]
    feature2_int_list = [int(x) for x in feature2]
    if operator == "+":
        feature = [f1 + f2 for f1, f2 in zip(feature1_int_list, feature2_int_list)]
        featurename = "add(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    if operator == "-":
        feature = [f1 - f2 for f1, f2 in zip(feature1_int_list, feature2_int_list)]
        featurename = "subtract(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    if operator == "*":
        feature = [f1 * f2 for f1, f2 in zip(feature1_int_list, feature2_int_list)]
        featurename = "multiply(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    if operator == "/":
        feature = [f1 / f2 if f2 != 0 else f1 for f1, f2 in zip(feature1_int_list, feature2_int_list)]
        featurename = "divide(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    return feature, featurename


def run_autogluon_lgbm(X_train, y_train, X_test, y_test, zeroshot=False):
    label = "target"
    X_train["target"] = y_train

    allowed_models = [
        "GBM",
    ]

    for k in list(zeroshot2024.keys()):
        if k not in allowed_models:
            del zeroshot2024[k]
        else:
            if not zeroshot:
                zeroshot2024[k] = zeroshot2024[k][:1]

    # -- Run AutoGluon
    predictor = TabularPredictor(
        label=label,
        eval_metric="log_loss",  # roc_auc (binary), log_loss (multiclass)
        problem_type="multiclass",  # binary, multiclass
        verbosity=-1,
    )

    predictor.fit(
        time_limit=int(60 * 60 * 4),
        memory_limit=48 * 1024 * 1024,
        num_cpus=8,
        num_gpus=0,
        train_data=X_train,
        presets="best_quality",
        dynamic_stacking=False,
        hyperparameters=zeroshot2024,
        # Validation Protocol
        num_bag_folds=8,
        num_bag_sets=1,
        num_stack_levels=0,
    )
    predictor.fit_summary(verbosity=-1)
    data = pd.concat([X_train, y_train], axis=1)
    lb = predictor.leaderboard(data)
    return lb


def get_result(X_train, y_train, X_test, y_test, dataset, feature, featurename, operator):
    print("Create new Feature: " + str(featurename))
    X_train_new = X_train.copy()
    X_train_new[featurename] = feature
    print("Run Autogluon with new Feature")
    lb = run_autogluon_lgbm(X_train_new, y_train, X_test, y_test)
    print("Leaderboard: " + str(lb))
    models = lb["model"]
    new_results = pd.DataFrame(columns=['dataset', 'feature', 'operator', 'model', 'score'])
    for model in models:
        score_val = lb.loc[lb['model'] == model, 'score_val'].values[0]
        new_results.loc[len(new_results)] = [dataset, featurename, operator, model, score_val]
    return new_results


def main():
    result_matrix = pd.DataFrame(columns=['dataset', 'feature', 'operator', 'model', 'score'])
    print("Result Matrix created")
    datasets = [146818]  # get_all_amlb_dataset_ids()
    unary_operators = ["min", "max", "freq", "abs", "log",
                        "sqrt", "square", "sigmoid"]  # , "round", "residual"]  # Unary OpenFE Operators
    binary_operators = ["+", "-", "*", "/", ]
                        # "GroupByThenMin", "GroupByThenMax", "GroupByThenMean",
                        # "GroupByThenMedian", "GroupByThenStd", "GroupByThenRank",
                        # "Combine", "CombineThenFreq", "GroupByThenNUnique"]  # Binary OpenFE Operators
    print("Iterate over Datasets")
    for dataset in datasets:
        X_train, y_train, X_test, y_test = get_openml_dataset(dataset)
        # X_train = X_train.drop(["A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14"], axis=1)
        # X_test = X_test.drop(["A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14"], axis=1)
        X_train_original = X_train.copy()
        for feature1 in X_train.columns:
            for feature2 in X_train.columns:
                for operator in binary_operators:
                    feature, featurename = create_feature_and_featurename(feature1=X_train[feature1], feature2=X_train[feature2], operator=operator)
                    new_rows = get_result(X_train, y_train, X_test, y_test, dataset, feature, featurename, operator)
                    result_matrix = pd.concat([result_matrix, pd.DataFrame(new_rows)], ignore_index=True)
                    print("Result for " + featurename + ": " + str(new_rows))
        for feature1 in X_train_original.columns:
            for operator in unary_operators:
                feature, featurename = create_feature_and_featurename(feature1=X_train[feature1], feature2=None, operator=operator)
                new_rows = get_result(X_train_original, y_train, X_test, y_test, dataset, feature, featurename, operator)
                result_matrix = pd.concat([result_matrix, pd.DataFrame(new_rows)], ignore_index=True)
                print("Result for " + featurename + ": " + str(new_rows))
    result_matrix.to_parquet("Operator_Model_Feature_Matrix_1.parquet")
    print("Final Result: \n" + str(result_matrix))


if __name__ == '__main__':
    main()
