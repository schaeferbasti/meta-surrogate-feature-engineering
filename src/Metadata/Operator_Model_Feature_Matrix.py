import numpy as np
import pandas as pd
import requests
import yaml
import openml
from pymfe.mfe import MFE

from autogluon.tabular import TabularPredictor
from tabrepo_2024_custom import zeroshot2024


def get_matrix_columns():
    return ['dataset - id', 'dataset - task type', 'dataset - number of classes', 'feature - name', 'feature - count', 'feature - mean', 'feature - std', 'model', 'score']


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
    dict
]:
    task = openml.tasks.get_task(
        openml_task_id,
        download_splits=True,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    dataset_metadata = {"task_id": task.task_id, "task_type": task.task_type, "number_of_classes": len(task.class_labels) if task.class_labels else 'N/A'}
    train_idx, test_idx = task.get_train_test_split_indices()
    X, y = task.get_X_and_y(dataset_format="dataframe")  # type: ignore
    train_x, train_y = X.iloc[train_idx], y.iloc[train_idx]
    test_x, test_y = X.iloc[test_idx], y.iloc[test_idx]
    return train_x, train_y, test_x, test_y, dataset_metadata


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
    elif operator == "max":
        feature = feature1.apply(lambda x: max(feature1_int_list))
        featurename = "max(" + str(feature1.name) + ")"
    elif operator == "freq":
        feature = feature1.apply(lambda x: feature1_int_list.count(int(x)))
        featurename = "freq(" + str(feature1.name) + ")"
    elif operator == "abs":
        feature = feature1.apply(lambda x: abs(float(x)))
        featurename = "abs(" + str(feature1.name) + ")"
    elif operator == "log":
        feature = feature1.apply(lambda x: np.log(np.abs(float(x).replace(0, np.nan))))
        featurename = "log(" + str(feature1.name) + ")"
    elif operator == "sqrt":
        feature = feature1.apply(lambda x: np.sqrt(np.abs(float(x))))
        featurename = "sqrt(" + str(feature1.name) + ")"
    elif operator == "square":
        feature = feature1.apply(lambda x: np.square(float(x)))
        featurename = "square(" + str(feature1.name) + ")"
    elif operator == "sigmoid":
        feature = feature1.apply(lambda x: 1 / (1 + np.exp(-float(x))))
        featurename = "sigmoid(" + str(feature1.name) + ")"
    elif operator == "round":
        feature = feature1.apply(lambda x: np.floor(float(x)))
        featurename = "round(" + str(feature1.name) + ")"
    elif operator == "residual":
        feature = feature1.apply(lambda x: float(x) - np.floor(float(x)))
        featurename = "residual(" + str(feature1.name) + ")"
    else:
        raise NotImplementedError(f"Unrecognized operator {operator}.")
    return feature, featurename


def create_binary_feature_and_featurename(feature1, feature2, operator):
    feature1_int_list = [int(x) for x in feature1]
    feature2_int_list = [int(x) for x in feature2]
    if operator == "+":
        feature = [f1 + f2 for f1, f2 in zip(feature1_int_list, feature2_int_list)]
        featurename = "add(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "-":
        feature = [f1 - f2 for f1, f2 in zip(feature1_int_list, feature2_int_list)]
        featurename = "subtract(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "*":
        feature = [f1 * f2 for f1, f2 in zip(feature1_int_list, feature2_int_list)]
        featurename = "multiply(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "/":
        feature = [f1 / f2 if f2 != 0 else f1 for f1, f2 in zip(feature1_int_list, feature2_int_list)]
        featurename = "divide(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "GroupByThenMin":
        temp = feature1.groupby(feature2).min()
        temp.loc[np.nan] = np.nan
        feature = feature2.apply(lambda x: temp.loc[x])
        featurename = "GroupByThenMin(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "GroupByThenMax":
        temp = feature1.groupby(feature2).max()
        temp.loc[np.nan] = np.nan
        feature = feature2.apply(lambda x: temp.loc[x])
        featurename = "GroupByThenMax(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "GroupByThenMean":
        feature1 = pd.to_numeric(feature1, errors='coerce')
        temp = feature1.groupby(feature2)
        temp = temp.mean()
        temp.loc[np.nan] = np.nan
        feature = feature2.apply(lambda x: temp.loc[x])
        featurename = "GroupByThenMean(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "GroupByThenMedian":
        feature1 = pd.to_numeric(feature1, errors='coerce')
        temp = feature1.groupby(feature2).median()
        temp.loc[np.nan] = np.nan
        feature = feature2.apply(lambda x: temp.loc[x])
        featurename = "GroupByThenMedian(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "GroupByThenStd":
        feature1 = pd.to_numeric(feature1, errors='coerce')
        temp = feature1.groupby(feature2).std()
        temp.loc[np.nan] = np.nan
        feature = feature2.apply(lambda x: temp.loc[x])
        featurename = "GroupByThenStd(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == 'GroupByThenRank':
        feature1 = pd.to_numeric(feature1, errors='coerce')
        feature = feature1.groupby(feature2).rank(ascending=True, pct=True)
        featurename = "GroupByThenRank(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "GroupByThenFreq":
        def _f(x):
            value_counts = x.value_counts()
            value_counts.loc[np.nan] = np.nan
            return x.apply(lambda x: value_counts.loc[x])
        feature = feature1.groupby(feature2).apply(_f)
        featurename = "GroupByThenFreq(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "GroupByThenNUnique":
        nunique = feature1.groupby(feature2).nunique()
        nunique.loc[np.nan] = np.nan
        feature = feature2.apply(lambda x: nunique.loc[x])
        featurename = "GroupByThenNUnique(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "Combine":
        temp = feature1.astype(str) + '_' + feature2.astype(str)
        temp[feature1.isna() | feature2.isna()] = np.nan
        temp, _ = temp.factorize()
        feature = pd.Series(temp, index=feature1.index).astype("float64")
        featurename = "Combine(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "CombineThenFreq":
        temp = feature1.astype(str) + '_' + feature2.astype(str)
        temp[feature1.isna() | feature2.isna()] = np.nan
        value_counts = temp.value_counts()
        value_counts.loc[np.nan] = np.nan
        feature = temp.apply(lambda x: value_counts.loc[x])
        featurename = "CombineThenFreq(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    else:
        raise NotImplementedError(f"Unrecognized operator {operator}.")
    return feature, featurename


def run_autogluon_lgbm(X_train, y_train, zeroshot=False):
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


def get_result(X_train, y_train, dataset_metadata, feature, featurename):
    feature_pandas_description = pd.DataFrame(feature, columns=[featurename]).describe()
    feature_metadata = {"feature - name": featurename,
                        "feature - count": feature_pandas_description.iloc[0],
                        "feature - mean/unique": feature_pandas_description.iloc[1],
                        "feature - std/top": feature_pandas_description.iloc[2]}
    print("Create new Feature: " + str(feature_metadata["feature - name"]))
    X_train_new = X_train.copy()
    X_train_new[feature_metadata["feature - name"]] = feature
    print("Run Autogluon with new Feature")
    lb = run_autogluon_lgbm(X_train_new, y_train)
    models = lb["model"]
    columns = get_matrix_columns()
    new_results = pd.DataFrame(columns=columns)
    for model in models:
        score_val = lb.loc[lb['model'] == model, 'score_val'].values[0]
        new_results.loc[len(new_results)] = [dataset_metadata["task_id"], dataset_metadata["task_type"], dataset_metadata["number_of_classes"], feature_metadata["feature - name"], feature_metadata["feature - count"], feature_metadata["feature - mean/unique"], feature_metadata["feature - std/top"], model, score_val]
    print("Result for " + featurename + ": " + str(new_results))
    return new_results


def main():
    columns = get_matrix_columns()
    result_matrix = pd.DataFrame(columns=columns)
    print("Result Matrix created")
    datasets = [146818, 146820, 168911, 190411]  # get_all_amlb_dataset_ids()
    unary_operators = ["min", "max", "freq", "abs", "log", "sqrt", "square", "sigmoid", "round", "residual"]  # Unary OpenFE Operators
    binary_operators = ["+", "-", "*", "/", "GroupByThenMin", "GroupByThenMax", "GroupByThenMean", "GroupByThenMedian", "GroupByThenStd", "GroupByThenRank", "Combine", "CombineThenFreq", "GroupByThenNUnique"]  # Binary OpenFE Operators
    print("Iterate over Datasets")
    for dataset in datasets:
        X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset(dataset)
        # X_train = X_train.drop(["A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14"], axis=1)
        # X_test = X_test.drop(["A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14"], axis=1)

        for feature1 in X_train.columns:
            for feature2 in X_train.columns:
                for operator in binary_operators:
                    feature, featurename = create_feature_and_featurename(feature1=X_train[feature1], feature2=X_train[feature2], operator=operator)
                    new_rows = get_result(X_train, y_train, dataset_metadata, feature, featurename)
                    result_matrix = pd.concat([result_matrix, pd.DataFrame(new_rows)], ignore_index=True)
        for feature1 in X_train.columns:
            for operator in unary_operators:
                feature, featurename = create_feature_and_featurename(feature1=X_train[feature1], feature2=None, operator=operator)
                new_rows = get_result(X_train, y_train, dataset_metadata, feature, featurename)
                result_matrix = pd.concat([result_matrix, pd.DataFrame(new_rows)], ignore_index=True)
        result_matrix.to_parquet("Operator_Model_Feature_Matrix_2.parquet")
    result_matrix.to_parquet("Operator_Model_Feature_Matrix_2.parquet")
    print("Final Result: \n" + str(result_matrix))


if __name__ == '__main__':
    main()
