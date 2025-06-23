from __future__ import annotations

from autogluon.core.data import LabelCleaner
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from sklearn.metrics import roc_auc_score

# Import a TabArena model
from tabrepo.benchmark.models.ag.realmlp.realmlp_model import RealMLPModel

import numpy as np
import pandas as pd
from pymfe.mfe import MFE

from src.utils.create_feature_and_featurename import create_featurenames, extract_operation_and_original_features
from src.utils.get_data import get_openml_dataset_and_metadata, get_openml_dataset_split_and_metadata
from src.utils.get_matrix import get_matrix_core_columns
# from src.Metadata.d2v.Add_d2v_Metafeatures import add_d2v_metadata_columns
from src.Metadata.mfe.Add_MFE_Metafeatures import add_mfe_metadata_columns
# from src.Metadata.pandas.Add_Pandas_Metafeatures import add_pandas_metadata_columns
# from src.Metadata.tabpfn.Add_TabPFN_Metafeatures import add_tabpfn_metadata_columns

import warnings
warnings.filterwarnings('ignore')


def create_empty_core_matrix_for_new_dataset(dataset, model) -> pd.DataFrame:
    columns = get_matrix_core_columns()
    comparison_result_matrix = pd.DataFrame(columns=columns)
    X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset)
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


def add_method_metadata(result_matrix, dataset_metadata, X_predict, y_predict, method):
    if method == "d2v":
        result_matrix = add_d2v_metadata_columns(dataset_metadata, X_predict, result_matrix)
    elif method == "mfe":
        result_matrix, _, _, _, _, _, _, _, _ = add_mfe_metadata_columns(X_predict, y_predict, result_matrix)
    elif method == "pandas":
        result_matrix = add_pandas_metadata_columns(dataset_metadata, X_predict, result_matrix)
    elif method == "tabpfn":
        result_matrix = add_tabpfn_metadata_columns(X_predict, y_predict, result_matrix)
    return result_matrix


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
    clf = RealMLPModel()
    clf.fit(X=result_matrix, y=y_result)

    # Predict and score
    prediction_probabilities = clf.predict_proba(X=comparison_result_matrix)
    # roc_auc = roc_auc_score(y_comparison, prediction_probabilities)
    # print("ROC AUC: " + str(roc_auc))
    prediction_probabilities_df = pd.DataFrame({'predicted_improvement': prediction_probabilities})
    prediction_probabilities_df.to_parquet("Prediction_" + str(category_or_method) + ".parquet")
    best_operations = prediction_probabilities_df.nlargest(n=20, columns="predicted_improvement", keep="first")
    best_operations.to_parquet("Best_Operations_" + str(category_or_method) + ".parquet")


def get_dummy_mfe_metafeatures():
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


def main(dataset_id, model):
    if model == "GBM":
        model = "LightGBM_BAG_L1"
    X_predict, y_predict, dataset_metadata = get_openml_dataset_and_metadata(dataset_id)
    methods = ["mfe"]
    # methods = ["d2v", "mfe", "pandas", "tabpfn"]
    for method in methods:
        print("Read Matrices")
        result_matrix = pd.read_parquet("src/Metadata/" + method + "/" + method + "_metafeatures.parquet")
        comparison_result_matrix = result_matrix
        # comparison_result_matrix = create_empty_core_matrix_for_new_dataset(dataset_id, model)
        # comparison_result_matrix = pd.read_parquet("../Metadata/core/Core_Matrix.parquet")
        # comparison_result_matrix = add_method_metadata(comparison_result_matrix, dataset_metadata, X_predict, y_predict, method)
        print("Add Metadata to comparison_result_matrix")
        # comparison_result_matrix, dataset_metadata_general_names, dataset_metadata_statistical_names, dataset_metadata_info_theory_names, dataset_metadata_landmarking_names, dataset_metadata_complexity_names, dataset_metadata_clustering_names, dataset_metadata_concept_names, dataset_metadata_itemset_names = add_mfe_metadata_columns(X_predict, y_predict, comparison_result_matrix)
        # categories = [dataset_metadata_general_names, dataset_metadata_statistical_names, dataset_metadata_info_theory_names, dataset_metadata_landmarking_names, dataset_metadata_complexity_names, dataset_metadata_clustering_names, dataset_metadata_concept_names, dataset_metadata_itemset_names]
        categories = get_dummy_mfe_metafeatures()
        category_names = ["general", "statistical", "info_theory", "landmarking", "complexity", "clustering", "concept", "itemset"]
        # Keep all categories
        print("Keep all categories")
        comparison_result_matrix_copy = comparison_result_matrix.copy()
        result_matrix_copy = result_matrix.copy()
        predict_improvement(result_matrix_copy, comparison_result_matrix_copy, "all")
        # Remove one category completely and sample 5 metafeatures from the other categories
        print("Remove one category completely")
        for i in range(len(categories)):
            comparison_result_matrix_copy = comparison_result_matrix.copy()
            result_matrix_copy = result_matrix.copy()
            comparison_result_matrix_copy = comparison_result_matrix_copy.drop(categories[i], axis=1)
            result_matrix_copy = result_matrix_copy.drop(categories[i], axis=1)
            predict_improvement(result_matrix_copy, comparison_result_matrix_copy, "without_" + category_names[i])
        # Remove all categories completely but one
        print("Remove all categories completely but one")
        for i in range(len(categories)):
            comparison_result_matrix_copy = comparison_result_matrix.copy()
            result_matrix_copy = result_matrix.copy()
            keep_core = get_matrix_core_columns()
            keep = keep_core + categories[i]
            comparison_result_matrix_copy = comparison_result_matrix_copy[keep]
            result_matrix_copy = result_matrix_copy[keep]
            predict_improvement(result_matrix_copy, comparison_result_matrix_copy, category_names[i])


if __name__ == '__main__':
    dataset_id = 190411
    models = "GBM"
    main(dataset_id, models)
