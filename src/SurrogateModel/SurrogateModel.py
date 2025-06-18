import random

import pandas as pd

from src.utils.get_data import get_openml_dataset_and_metadata
from src.utils.get_matrix import get_matrix_core_columns
from src.utils.get_metafeatures import get_mfe_feature_metadata
from src.utils.run_models import multi_predict_autogluon_lgbm, predict_autogluon_lgbm
from src.Metadata.d2v.Add_d2v_Metafeatures import add_d2v_metadata_columns
from src.Metadata.mfe.Add_MFE_Metafeatures import add_mfe_metadata_columns
from src.Metadata.pandas.Add_Pandas_Metafeatures import add_pandas_metadata_columns
from src.Metadata.tabpfn.Add_TabPFN_Metafeatures import add_tabpfn_metadata_columns

import warnings
warnings.filterwarnings('ignore')

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


def predict_improvment(comparison_result_matrix, dataset_metadata, result_matrix, category_or_method, fold):
    # Single-predictor (improvement given all possible operations on features)
    prediction = predict_autogluon_lgbm(result_matrix, comparison_result_matrix)
    prediction.to_parquet("Prediction_" + category_or_method + "_" + fold + ".parquet")
    #  evaluation, prediction, best_operations = predict_operators_for_models(X, y, X_predict, y_predict, models=models, zeroshot=False)
    #  evaluation.to_parquet('Evaluation.parquet')
    prediction_result = pd.read_parquet("Prediction_" + category_or_method + "_" + fold + ".parquet")
    best_operations = prediction_result.nlargest(n=20, columns="predicted_improvement", keep="first")
    best_operations.to_parquet("Best_Operations_" + category_or_method + "_" + fold + ".parquet")
    # Multi-predictor (features + operator & improvement)
    multi_prediction = multi_predict_autogluon_lgbm(result_matrix, comparison_result_matrix)
    multi_prediction.to_parquet("Multi_Prediction_" + category_or_method + "_" + fold + ".parquet")
    #  multi_evaluation, multi_prediction = multi_predict_operators_for_models(X, y, X_predict, y_predict, models=models, zeroshot=False)
    #  multi_evaluation.to_parquet('Multi_Evaluation.parquet')
    multi_prediction_result = pd.read_parquet("Multi_Prediction_" + category_or_method + "_" + fold + ".parquet")
    multi_best_operations = multi_prediction_result.nlargest(n=20, columns="predicted_improvement", keep="first")
    multi_best_operations.to_parquet("Multi_Best_Operations_" + category_or_method + "_" + fold + ".parquet")


def main(dataset_id, models):
    # Select Models
    if models is None:
        models = {"GBM": {}, "RF": {}, "KNN": {}, "CAT": {}, "XGB": {}, "LR": {}, "FASTAI": {}, "AG_AUTOMM": {},
                  "NN_TORCH": {}}
    else:
        models = {"LightGBM_BAG_L1": {}}

    X_predict, y_predict, dataset_metadata = get_openml_dataset_and_metadata(dataset_id)
    methods = ["mfe"]
    folds = 10
    # methods = ["d2v", "mfe", "pandas", "tabpfn"]
    for fold in range(folds):
        for method in methods:
            print("Read Matrices")
            result_matrix = pd.read_parquet("../Metadata/" + method + "/" + method + "_metafeatures.parquet")
            # Need to read from another matrix later?
            comparison_result_matrix = pd.read_parquet("../Metadata/core/Core_Matrix_Example.parquet")
            # Only want data from our dataset to test
            comparison_result_matrix = comparison_result_matrix.loc[comparison_result_matrix['dataset - id'] == dataset_id]
            # comparison_result_matrix = add_method_metadata(comparison_result_matrix, dataset_metadata, X_predict, y_predict, method)
            print("Add Metadata to comparison_result_matrix")
            comparison_result_matrix, dataset_metadata_general_names, dataset_metadata_statistical_names, dataset_metadata_info_theory_names, dataset_metadata_landmarking_names, dataset_metadata_complexity_names, dataset_metadata_clustering_names, dataset_metadata_concept_names, dataset_metadata_itemset_names = add_mfe_metadata_columns(X_predict, y_predict, comparison_result_matrix)
            categories = [dataset_metadata_general_names, dataset_metadata_statistical_names, dataset_metadata_info_theory_names, dataset_metadata_landmarking_names, dataset_metadata_complexity_names, dataset_metadata_clustering_names, dataset_metadata_concept_names, dataset_metadata_itemset_names]
            category_names = ["general", "statistical", "info_theory", "landmarking", "complexity", "clustering", "concept", "itemset"]
            # Keep all categories, sample 5 metafeatures from all categories
            print("Keep all categories, sample 5 metafeatures from all categories")
            for i in range(len(categories)):
                comparison_result_matrix_copy = comparison_result_matrix.copy()
                category = categories[i]
                if len(category) > 5:
                    keep = random.sample(category, 5)
                    drop = list(set(category) - set(keep))
                    comparison_result_matrix_copy = comparison_result_matrix_copy.drop(drop, axis=1)
                print(str(comparison_result_matrix_copy.columns))
                print("Predict Imrpovement")
                predict_improvment(comparison_result_matrix_copy, dataset_metadata, result_matrix, "all", fold)
            # Remove one category completely and sample 5 metafeatures from the other categories
            print("Remove one category completely and sample 5 metafeatures from the other categories")
            for i in range(len(categories)):
                comparison_result_matrix_copy = comparison_result_matrix.copy()
                # Drop selected category
                comparison_result_matrix_copy = comparison_result_matrix_copy.drop(categories[i], axis=1)
                # Randomly sample metafeatures from the other categories
                for j in range(len(categories)):
                    if j != i:
                        category = categories[j]
                        if len(category) > 5:
                            keep = random.sample(category, 5)
                            drop = list(set(category) - set(keep))
                            comparison_result_matrix_copy = comparison_result_matrix_copy.drop(drop, axis=1)
                print(str(comparison_result_matrix_copy. columns))
                print("Predict Imrpovement")
                predict_improvment(comparison_result_matrix_copy, dataset_metadata, result_matrix, "without_" + category_names[i], fold)
            # Remove all categories completely but one and sample 5 metafeatures from this category
            print("Remove all categories completely but one and sample 5 metafeatures from this category")
            for i in range(len(categories)):
                comparison_result_matrix_copy = comparison_result_matrix.copy()
                # Drop all but selected category
                keep_core = get_matrix_core_columns()
                keep = keep_core + categories[i]
                comparison_result_matrix_copy = comparison_result_matrix_copy[keep]
                # Randomly sample metafeatures from the kept category
                category = categories[i]
                if len(category) > 5:
                    keep_core = get_matrix_core_columns()
                    keep = random.sample(category, 5) + keep_core
                    drop = list(set(category) - set(keep))
                    comparison_result_matrix_copy = comparison_result_matrix_copy.drop(drop, axis=1)
                print(str(comparison_result_matrix_copy.columns))
                print("Predict Imrpovement")
                predict_improvment(comparison_result_matrix_copy, dataset_metadata, result_matrix, category_names[i], fold)


if __name__ == '__main__':
    dataset_id = 2073
    models = "GBM"
    main(dataset_id, models)
