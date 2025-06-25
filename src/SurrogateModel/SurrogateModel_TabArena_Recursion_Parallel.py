from __future__ import annotations

import argparse

from autogluon.core.data import LabelCleaner
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular.models import CatBoostModel
from sklearn.metrics import roc_auc_score

from src.Apply_and_Test.Apply_FE import execute_feature_engineering
from src.Metadata.d2v.Add_d2v_Metafeatures import add_d2v_metadata_columns
from src.Metadata.pandas.Add_Pandas_Metafeatures import add_pandas_metadata_columns
from src.Metadata.tabpfn.Add_TabPFN_Metafeatures import add_tabpfn_metadata_columns
# Import a TabArena model
from tabrepo.benchmark.models.ag.realmlp.realmlp_model import RealMLPModel
from tabrepo.benchmark.models.ag.tabdpt.tabdpt_model import TabDPTModel


import numpy as np
import pandas as pd
from pymfe.mfe import MFE

from src.utils.create_feature_and_featurename import create_featurenames, extract_operation_and_original_features
from src.utils.get_data import get_openml_dataset_and_metadata, get_openml_dataset_split_and_metadata, split_data, \
    concat_data
from src.utils.get_matrix import get_matrix_core_columns
# from src.Metadata.d2v.Add_d2v_Metafeatures import add_d2v_metadata_columns
from src.Metadata.mfe.Add_MFE_Metafeatures import add_mfe_metadata_columns
# from src.Metadata.pandas.Add_Pandas_Metafeatures import add_pandas_metadata_columns
# from src.Metadata.tabpfn.Add_TabPFN_Metafeatures import add_tabpfn_metadata_columns

import warnings
warnings.filterwarnings('ignore')


def create_empty_core_matrix_for_dataset(X_train, model) -> pd.DataFrame:
    columns = get_matrix_core_columns()
    comparison_result_matrix = pd.DataFrame(columns=columns)
    for feature1 in X_train.columns:
        featurename = "without - " + str(feature1)
        columns = get_matrix_core_columns()
        new_rows = pd.DataFrame(columns=columns)
        operator = "delete"
        new_rows.loc[len(new_rows)] = [
            dataset_id,
            featurename,
            operator,
            model,
            0
        ]
        comparison_result_matrix = pd.concat([comparison_result_matrix, pd.DataFrame(new_rows)], ignore_index=True)
    columns = get_matrix_core_columns()
    new_rows = pd.DataFrame(columns=columns)
    featurenames = create_featurenames(X_train.columns)
    for i in range(len(featurenames)):
        operator, _ = extract_operation_and_original_features(featurenames[i])
        new_rows.loc[len(new_rows)] = [
            dataset_id,
            featurenames[i],
            operator,
            model,
            0
        ]
    comparison_result_matrix = pd.concat([comparison_result_matrix, pd.DataFrame(new_rows)], ignore_index=True)
    return comparison_result_matrix


def get_mfe_categories():
    X_dummy = np.array([[0, 1], [1, 0]])
    y_dummy = np.array([0, 1])

    # Initialize result dictionary
    groups = [
        "general",
        "statistical",
        "info-theory",
        "landmarking",
        "complexity",
        "clustering",
        "concept",
        "itemset"
    ]

    # This will hold your result like:
    # [dataset_metadata_general_names, dataset_metadata_statistical_names, ...]
    group_feature_lists = []

    for group in groups:
        mfe = MFE(groups=[group])
        mfe.fit(X_dummy, y_dummy)
        feature_names, _ = mfe.extract()
        group_feature_lists.append(feature_names)
    return group_feature_lists


def get_mfe_category(category):
    X_dummy = np.array([[0, 1], [1, 0]])
    y_dummy = np.array([0, 1])

    # Initialize result dictionary
    groups = [
        category
    ]

    # This will hold your result like:
    # [dataset_metadata_general_names, dataset_metadata_statistical_names, ...]
    group_feature_lists = []

    for group in groups:
        mfe = MFE(groups=[group])
        mfe.fit(X_dummy, y_dummy)
        feature_names, _ = mfe.extract()
        group_feature_lists.append(feature_names)
    return group_feature_lists


def add_method_metadata(result_matrix, dataset_metadata, X_predict, y_predict, method):
    if method == "d2v":
        result_matrix = add_d2v_metadata_columns(dataset_metadata, X_predict, result_matrix)
    elif method == "pandas":
        result_matrix = add_pandas_metadata_columns(dataset_metadata, X_predict, result_matrix)
    elif method == "tabpfn":
        result_matrix = add_tabpfn_metadata_columns(X_predict, y_predict, result_matrix)
    return result_matrix


def recursive_feature_addition(i, n_features_to_add, X_train, y_train, X_test, y_test, model, method, dataset_metadata, category_to_drop):
    if i >= n_features_to_add:
        return X_train, y_train, X_test, y_test
    # Reload base matrix
    result_matrix = pd.read_parquet("src/Metadata/core/Core_Matrix_Complete.parquet")
    # Create comparison matrix for new dataset
    comparison_result_matrix = create_empty_core_matrix_for_dataset(X_train, model)
    comparison_result_matrix = add_method_metadata(comparison_result_matrix, dataset_metadata, X_train, y_train, method)
    comparison_result_matrix, general, statistical, info_theory, landmarking, complexity, clustering, concept, itemset = add_mfe_metadata_columns(X_train, y_train, comparison_result_matrix)
    # Drop category
    comparison_result_matrix_copy = comparison_result_matrix.drop(columns=category_to_drop, errors='ignore')
    result_matrix_copy = result_matrix.drop(columns=category_to_drop, errors='ignore')
    # Predict and split again
    data = predict_improvement(result_matrix_copy, comparison_result_matrix_copy, "all")
    X_train, y_train, X_test, y_test = split_data(data, "target")
    # Recurse
    recursive_feature_addition_mfe(i + 1, n_features_to_add, X_train, y_train, X_test, y_test, model, method, dataset_metadata, category_to_drop)
    return X_train, y_train, X_test, y_test


def recursive_feature_addition_mfe(i, n_features_to_add, X_train, y_train, X_test, y_test, model, method, dataset_metadata, category_to_drop):
    if i >= n_features_to_add:
        return X_train, y_train, X_test, y_test
    # Reload base matrix
    result_matrix = pd.read_parquet("src/Metadata/core/Core_Matrix_Complete.parquet")
    # Create comparison matrix for new dataset
    comparison_result_matrix = create_empty_core_matrix_for_dataset(X_train, model)
    comparison_result_matrix = result_matrix, _, _, _, _, _, _, _, _ = add_mfe_metadata_columns(X_train, y_train, comparison_result_matrix)
    comparison_result_matrix, general, statistical, info_theory, landmarking, complexity, clustering, concept, itemset = add_mfe_metadata_columns(X_train, y_train, comparison_result_matrix)
    # Drop no category, single category or all categories but one
    comparison_result_matrix_copy = comparison_result_matrix.drop(columns=category_to_drop, errors='ignore')
    result_matrix_copy = result_matrix.drop(columns=category_to_drop, errors='ignore')
    # Predict and split again
    data = predict_improvement(result_matrix_copy, comparison_result_matrix_copy, "all")
    X_train, y_train, X_test, y_test = split_data(data, "target")
    # Recurse
    recursive_feature_addition_mfe(i + 1, n_features_to_add, X_train, y_train, X_test, y_test, model, method, dataset_metadata, category_to_drop)
    return X_train, y_train, X_test, y_test


def predict_improvement(result_matrix, comparison_result_matrix, category_or_method):
    y_result = result_matrix["improvement"]
    result_matrix = result_matrix.drop("improvement", axis=1)
    y_comparison = comparison_result_matrix["improvement"]
    comparison_result_matrix = comparison_result_matrix.drop("improvement", axis=1)
    # Single-predictor (improvement given all possible operations on features)
    feature_generator, label_cleaner = (
        AutoMLPipelineFeatureGenerator(),
        LabelCleaner.construct(problem_type="regression", y=y_result),
    )
    result_matrix, y_result = (
        feature_generator.fit_transform(result_matrix),
        label_cleaner.transform(y_result),
    )
    comparison_result_matrix, y_comparison = feature_generator.transform(comparison_result_matrix), label_cleaner.transform(y_comparison)

    # Train TabArena Model
    # clf = RealMLPModel() # Catboost TabDPT -> tabarena hugging face
    clf = TabDPTModel()
    # clf = CatBoostModel()
    clf.fit(X=result_matrix, y=y_result)

    # Predict and score
    prediction = clf.predict(X=comparison_result_matrix)
    prediction_df = pd.concat([comparison_result_matrix[["dataset - id", "feature - name", "model"]], prediction], axis=1)
    best_operation = prediction_df.nlargest(n=1, columns="predicted_improvement", keep="first")
    data, _, _ = execute_feature_engineering(best_operation)
    return data


def main(method, dataset_id):
    print("Method: " + str(method) + ", Dataset: " + str(dataset_id))
    model = "LightGBM_BAG_L1"
    n_features_to_add = 10
    j = 0
    category = "No_Category"
    if method.startswith("mfe"):
        categories = get_mfe_categories()
        # Keep all categories
        if "all" in method:
            X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset_id)
            X_train, y_train, X_test, y_test = recursive_feature_addition_mfe(j, n_features_to_add, X_train, y_train, X_test, y_test, model, method, dataset_metadata, None)
            data = concat_data(X_train, y_train, X_test, y_test, "target")
            data.to_parquet("FE_" + str(dataset_id) + "_" + str(method) + "_" + category + "_recursion.parquet")

        # Remove one category completely
        if "without" in method:
            X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset_id)
            category = method.split("mfe_")[1]
            category_name = category.split("_")[1]
            category_to_remove = get_mfe_category(category_name)
            X_train, y_train, X_test, y_test = recursive_feature_addition_mfe(j, n_features_to_add, X_train, y_train, X_test, y_test, model, method, dataset_metadata, category_to_remove)
            data = concat_data(X_train, y_train, X_test, y_test, "target")
            data.to_parquet("FE_" + str(dataset_id) + "_" + str(method) + "_" + category + "_recursion.parquet")

        # Remove all categories completely but one
        if "only" in method:
            X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata( dataset_id)
            category = method.split("mfe_")[1]
            category_name = category.split("_")[1]
            category_to_keeo = get_mfe_category(category_name)
            categories_to_remove = categories.remove(category_to_keeo)
            X_train, y_train, X_test, y_test = recursive_feature_addition_mfe(j, n_features_to_add, X_train, y_train, X_test, y_test, model, method, dataset_metadata, categories_to_remove)
            data = concat_data(X_train, y_train, X_test, y_test, "target")
            data.to_parquet("FE_" + str(dataset_id) + "_" + str(method) + "_" + category + "_recursion.parquet")
    else:
        X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset_id)
        X_train, y_train, X_test, y_test = recursive_feature_addition(j, n_features_to_add, X_train, y_train, X_test, y_test, model, method, dataset_metadata, None)
        data = concat_data(X_train, y_train, X_test, y_test, "target")
        data.to_parquet("FE_" + str(dataset_id) + "_" + str(method) + "_" + category + "_recursion.parquet")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Surrogate Model with Metadata from Method')
    parser.add_argument('--mf_method', required=True, help='Metafeature Method')
    args = parser.parse_args()
    dataset_id = 190411
    main(args.mf_method, dataset_id)
