import time

import numpy as np
import pandas as pd

from src.Metadata.pandas.Add_Pandas_Metafeatures import get_operator_count, split_top_level_args
from src.utils.create_feature_and_featurename import create_feature
from src.utils.get_data import get_openml_dataset_split_and_metadata
from src.utils.get_matrix import get_additional_mfe_columns, get_additional_mfe_columns_group
from src.utils.get_metafeatures import get_mfe_dataset_metadata
from src.utils.get_operators import get_operators


def add_mfe_metadata_columns_group(X_train, y_train, result_matrix, group):
    columns = get_additional_mfe_columns_group(group)
    new_columns = pd.DataFrame(index=result_matrix.index, columns=columns)
    unary_operators, binary_operators = get_operators()
    operators = unary_operators + binary_operators + ["without"]
    for row in result_matrix.iterrows():
        featurename = row[1][1]
        X_train_copy = X_train.copy()
        operator_count = get_operator_count(featurename, operators)
        if featurename.startswith("without"):
            feature_to_delete = featurename.split(" - ")[1]
            X_train_copy = X_train_copy.drop(feature_to_delete, axis=1)
        elif operator_count > 1:
            features = featurename.split("(")[1].replace(")", "").replace(" ", "")
            inner = featurename.split("(", 1)[1].rsplit(")", 1)[0]
            args = split_top_level_args(inner)
            featurename1 = args[0]
            featurename2 = args[1] if len(args) > 1 else None
            if "(" in featurename1:
                feature1 = X_train_copy[featurename1]
            else:
                featurename1 = features.split(",")[0]
                feature1 = X_train_copy[featurename1]
            if featurename2 is not None:
                if "(" in featurename2:
                    feature2 = X_train_copy[featurename2]
                else:
                    feature2 = X_train_copy[featurename2]
            else:
                feature2 = None
            new_feature = create_feature(feature1, feature2, featurename)
            X_train_copy = X_train_copy.reset_index(drop=True)
            new_feature_df = pd.DataFrame(new_feature, columns=[featurename])
            new_feature_df = new_feature_df.reset_index(drop=True)
            X_train_copy = pd.concat([X_train_copy, new_feature_df], axis=1, ignore_index=False)
        else:
            features = featurename.split("(")[1].replace(")", "").replace(" ", "")
            if "," in features:
                featurename1 = features.split(",")[0]
                feature1 = X_train_copy[featurename1]
                featurename2 = features.split(",")[1]
                feature2 = X_train_copy[featurename2]
            else:
                featurename1 = features
                feature1 = X_train_copy[featurename1]
                feature2 = None
            new_feature = create_feature(feature1, feature2, featurename)
            X_train_copy = X_train_copy.reset_index(drop=True)
            new_feature_df = pd.DataFrame(new_feature, columns=[featurename])
            new_feature_df = new_feature_df.reset_index(drop=True)
            X_train_copy = pd.concat([X_train_copy, new_feature_df], axis=1)
        X_train_copy = X_train_copy.apply(lambda col: col.astype(str) if isinstance(col.dtype, pd.CategoricalDtype) else col)
        X = X_train_copy.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
        y = y_train.to_numpy()
        dataset_metadata_mfe, dataset_metadata_names, dataset_metadata_groups = get_mfe_dataset_metadata(X, y, group)
        metafeatures = dataset_metadata_mfe[1]
        new_row = pd.DataFrame([metafeatures], columns=dataset_metadata_names)
        matching_indices = result_matrix[result_matrix["feature - name"] == str(featurename)].index
        for idx in matching_indices:
            new_columns.loc[idx] = new_row.iloc[0]
    insert_position = result_matrix.shape[1] - 2
    result_matrix = pd.concat([result_matrix.iloc[:, :insert_position], new_columns, result_matrix.iloc[:, insert_position:]], axis=1)
    return result_matrix


def add_mfe_metadata_columns_groups(X_train, y_train, result_matrix, groups):
    columns_list = []
    unary_operators, binary_operators = get_operators()
    operators = unary_operators + binary_operators + ["without"]
    for group in groups:
        columns_list.extend(group)
    new_columns = pd.DataFrame(index=result_matrix.index, columns=columns_list)
    for row in result_matrix.iterrows():
        featurename = row[1][1]
        X_train_copy = X_train.copy()
        operator_count = get_operator_count(featurename, operators)
        if featurename.startswith("without"):
            feature_to_delete = featurename.split(" - ")[1]
            X_train_copy = X_train_copy.drop(feature_to_delete, axis=1)
        elif operator_count > 1:
            features = featurename.split("(")[1].replace(")", "").replace(" ", "")
            inner = featurename.split("(", 1)[1].rsplit(")", 1)[0]
            args = split_top_level_args(inner)
            featurename1 = args[0]
            featurename2 = args[1] if len(args) > 1 else None
            if "(" in featurename1:
                feature1 = X_train_copy[featurename1]
            else:
                featurename1 = features.split(",")[0]
                feature1 = X_train_copy[featurename1]
            if featurename2 is not None:
                if "(" in featurename2:
                    feature2 = X_train_copy[featurename2]
                else:
                    feature2 = X_train_copy[featurename2]
            else:
                feature2 = None
            new_feature = create_feature(feature1, feature2, featurename)
            X_train_copy = X_train_copy.reset_index(drop=True)
            new_feature_df = pd.DataFrame(new_feature, columns=[featurename])
            new_feature_df = new_feature_df.reset_index(drop=True)
            X_train_copy = pd.concat([X_train_copy, new_feature_df], axis=1, ignore_index=False)
        else:
            features = featurename.split("(")[1].replace(")", "").replace(" ", "")
            if "," in features:
                featurename1 = features.split(",")[0]
                feature1 = X_train_copy[featurename1]
                featurename2 = features.split(",")[1]
                feature2 = X_train_copy[featurename2]
            else:
                featurename1 = features
                feature1 = X_train_copy[featurename1]
                feature2 = None
            new_feature = create_feature(feature1, feature2, featurename)
            X_train_copy = X_train_copy.reset_index(drop=True)
            new_feature_df = pd.DataFrame(new_feature, columns=[featurename])
            new_feature_df = new_feature_df.reset_index(drop=True)
            X_train_copy = pd.concat([X_train_copy, new_feature_df], axis=1)
        X = X_train_copy.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
        y = y_train.to_numpy()
        metafeatures = []
        for group in groups:
            dataset_metafeatures_group_mfe, dataset_metadata_group_names, _ = get_mfe_dataset_metadata(X, y, group)
            metafeatures.extend(dataset_metafeatures_group_mfe[1])
        new_columns.columns = columns_list
        new_row = pd.DataFrame([metafeatures], columns=columns_list)
        matching_indices = result_matrix[result_matrix["feature - name"] == str(featurename)].index
        for idx in matching_indices:
            new_columns.loc[idx] = new_row.iloc[0]
    insert_position = result_matrix.shape[1] - 2
    result_matrix = pd.concat([result_matrix.iloc[:, :insert_position], new_columns, result_matrix.iloc[:, insert_position:]], axis=1)
    return result_matrix


def add_mfe_metadata_columns(X_train, y_train, result_matrix):
    columns = get_additional_mfe_columns()
    new_columns = pd.DataFrame(index=result_matrix.index, columns=columns)
    dataset_metadata_general_names = dataset_metadata_statistical_names = dataset_metadata_info_theory_names = dataset_metadata_landmarking_names = dataset_metadata_complexity_names = dataset_metadata_clustering_names = dataset_metadata_concept_names = dataset_metadata_itemset_names = None
    unary_operators, binary_operators = get_operators()
    operators = unary_operators + binary_operators + ["without"]
    for row in result_matrix.iterrows():
        featurename = row[1][1]
        X_train_copy = X_train.copy()
        operator_count = get_operator_count(featurename, operators)
        if featurename.startswith("without"):
            feature_to_delete = featurename.split(" - ")[1]
            X_train_copy = X_train_copy.drop(feature_to_delete, axis=1)
        elif operator_count > 1:
            features = featurename.split("(")[1].replace(")", "").replace(" ", "")
            inner = featurename.split("(", 1)[1].rsplit(")", 1)[0]
            args = split_top_level_args(inner)
            featurename1 = args[0]
            featurename2 = args[1] if len(args) > 1 else None
            if "(" in featurename1:
                feature1 = X_train_copy[featurename1]
            else:
                featurename1 = features.split(",")[0]
                feature1 = X_train_copy[featurename1]
            if featurename2 is not None:
                if "(" in featurename2:
                    feature2 = X_train_copy[featurename2]
                else:
                    feature2 = X_train_copy[featurename2]
            else:
                feature2 = None
            new_feature = create_feature(feature1, feature2, featurename)
            X_train_copy = X_train_copy.reset_index(drop=True)
            new_feature_df = pd.DataFrame(new_feature, columns=[featurename])
            new_feature_df = new_feature_df.reset_index(drop=True)
            X_train_copy = pd.concat([X_train_copy, new_feature_df], axis=1, ignore_index=False)
        else:
            features = featurename.split("(")[1].replace(")", "").replace(" ", "")
            if "," in features:
                featurename1 = features.split(",")[0]
                feature1 = X_train_copy[featurename1]
                featurename2 = features.split(",")[1]
                feature2 = X_train_copy[featurename2]
            else:
                featurename1 = features
                feature1 = X_train_copy[featurename1]
                feature2 = None
            new_feature = create_feature(feature1, feature2, featurename)
            X_train_copy = X_train_copy.reset_index(drop=True)
            new_feature_df = pd.DataFrame(new_feature, columns=[featurename])
            new_feature_df = new_feature_df.reset_index(drop=True)
            X_train_copy = pd.concat([X_train_copy, new_feature_df], axis=1)
        X = X_train_copy.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
        y = y_train.to_numpy()
        dataset_metadata_general_mfe, dataset_metadata_general_names, dataset_metadata_general_groups = get_mfe_dataset_metadata(X, y, "general")
        dataset_metadata_statistical_mfe, dataset_metadata_statistical_names, dataset_metadata_statistical_groups = get_mfe_dataset_metadata(X, y, "statistical")
        dataset_metadata_info_theory_mfe, dataset_metadata_info_theory_names, dataset_metadata_info_theory_groups = get_mfe_dataset_metadata(X, y, "info-theory")
        dataset_metadata_landmarking_mfe, dataset_metadata_landmarking_names, dataset_metadata_landmarking_groups = get_mfe_dataset_metadata(X, y, "landmarking")
        dataset_metadata_complexity_mfe, dataset_metadata_complexity_names, dataset_metadata_complexity_groups = get_mfe_dataset_metadata(X, y, "complexity")
        dataset_metadata_clustering_mfe, dataset_metadata_clustering_names, dataset_metadata_clustering_groups = get_mfe_dataset_metadata(X, y, "clustering")
        dataset_metadata_concept_mfe, dataset_metadata_concept_names, dataset_metadata_concept_groups = get_mfe_dataset_metadata(X, y, "concept")
        dataset_metadata_itemset_mfe, dataset_metadata_itemset_names, dataset_metadata_itemset_groups = get_mfe_dataset_metadata(X, y, "itemset")
        metafeatures = dataset_metadata_general_mfe[1] + dataset_metadata_statistical_mfe[1] + dataset_metadata_info_theory_mfe[1] + dataset_metadata_landmarking_mfe[1] + dataset_metadata_complexity_mfe[1] + dataset_metadata_clustering_mfe[1] + dataset_metadata_concept_mfe[1] + dataset_metadata_itemset_mfe[1]
        new_column_names = list(dataset_metadata_general_names + dataset_metadata_statistical_names + dataset_metadata_info_theory_names + dataset_metadata_landmarking_names + dataset_metadata_complexity_names + dataset_metadata_clustering_names + dataset_metadata_concept_names + dataset_metadata_itemset_names)
        new_columns.columns = new_column_names
        new_row = pd.DataFrame([metafeatures], columns=new_column_names)
        matching_indices = result_matrix[result_matrix["feature - name"] == str(featurename)].index
        for idx in matching_indices:
            new_columns.loc[idx] = new_row.iloc[0]
    insert_position = result_matrix.shape[1] - 2
    result_matrix = pd.concat([result_matrix.iloc[:, :insert_position], new_columns, result_matrix.iloc[:, insert_position:]], axis=1)
    return result_matrix, dataset_metadata_general_names, dataset_metadata_statistical_names, dataset_metadata_info_theory_names, dataset_metadata_landmarking_names, dataset_metadata_complexity_names, dataset_metadata_clustering_names, dataset_metadata_concept_names, dataset_metadata_itemset_names


def main():
    result_matrix = pd.read_parquet("src/Metadata/core/Core_Matrix_Complete.parquet")
    columns = get_additional_mfe_columns()
    result_matrix_columns = result_matrix.columns.values.tolist()
    columns = columns + result_matrix_columns
    result_matrix_pandas = pd.DataFrame(columns=columns)
    start = time.time()
    counter = 0
    for dataset, _ in result_matrix.groupby('dataset - id'):
        print("Dataset: " + str(dataset))
        try:
            pd.read_parquet("src/Metadata/mfe/MFE_all_Matrix_Complete" + str(dataset) + ".parquet")
        except FileNotFoundError:
            try:
                counter += 1
                start_dataset = time.time()
                X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(int(str(dataset)))
                result_matrix_dataset = result_matrix[result_matrix['dataset - id'] == dataset]
                result_matrix_dataset, dataset_metadata_general_names, dataset_metadata_statistical_names, dataset_metadata_info_theory_names, dataset_metadata_landmarking_names, dataset_metadata_complexity_names, dataset_metadata_clustering_names, dataset_metadata_concept_names, dataset_metadata_itemset_names = add_mfe_metadata_columns(X_train, y_train, result_matrix_dataset)
                result_matrix_pandas.columns = result_matrix_dataset.columns
                result_matrix_pandas = pd.concat([result_matrix_pandas, result_matrix_dataset], axis=0)
                result_matrix_pandas.to_parquet("src/Metadata/mfe/MFE_all_Matrix_Complete" + str(dataset) + ".parquet")
                end_dataset = time.time()
                print("Time for MFE on Dataset " + str(dataset) + ": " + str(end_dataset - start_dataset))
            except TypeError:
                continue
    result_matrix_pandas.to_parquet("src/Metadata/mfe/MFE_all_Matrix_Complete.parquet")
    end = time.time()
    print("Time for complete MFE MF: " + str(end - start) + " on " + str(counter) + " datasets.")


if __name__ == '__main__':
    main()
