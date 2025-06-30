import numpy as np
import pandas as pd

from src.utils.create_feature_and_featurename import create_feature
from src.utils.get_data import get_openml_dataset_split_and_metadata
from src.utils.get_matrix import get_additional_mfe_columns
from src.utils.get_metafeatures import get_mfe_feature_metadata, get_mfe_dataset_metadata


def add_mfe_metadata_columns(X_train, y_train, result_matrix):
    columns = get_additional_mfe_columns()
    new_columns = pd.DataFrame(index=result_matrix.index, columns=columns)
    dataset_metadata_general_names = dataset_metadata_statistical_names = dataset_metadata_info_theory_names = dataset_metadata_landmarking_names = dataset_metadata_complexity_names = dataset_metadata_clustering_names = dataset_metadata_concept_names = dataset_metadata_itemset_names = None
    for row in result_matrix.iterrows():
        featurename = row[1][1]
        X_train_copy = X_train.copy()
        # X_test_copy = X_test.copy()
        if featurename.startswith("without"):
            feature_to_delete = featurename.split(" - ")[1]
            feature = X_train_copy[feature_to_delete].to_numpy()
            X_train_copy = X_train_copy.drop(feature_to_delete, axis=1)
            # X_test_copy = X_test_copy.drop(feature_to_delete, axis=1)
        else:
            features = featurename.split("(")[1].replace(")", "").replace(" ", "")
            if "," in features:
                featurename1 = features.split(",")[0]
                feature1 = X_train_copy[featurename1]
                # feature1_test = X_test_copy[featurename1]
                featurename2 = features.split(",")[1]
                feature2 = X_train_copy[featurename2]
                # feature2_test = X_test_copy[featurename2]
            else:
                featurename1 = features
                feature1 = X_train_copy[featurename1]
                # feature1_test = X_test_copy[featurename1]
                feature2 = None
                feature2_test = None
            new_feature = create_feature(feature1, feature2, featurename)
            # new_feature_test = create_feature(feature1_test, feature2_test, featurename)
            new_feature_df = pd.DataFrame(new_feature, columns=[featurename])
            # new_feature_test_df = pd.DataFrame(new_feature_test, columns=[featurename])
            new_feature_df.index = X_train_copy.index
            X_train_copy = pd.concat([X_train_copy, new_feature_df], axis=1)
            # new_feature_test_df.index = X_test_copy.index
            # X_test_copy = pd.concat([X_test_copy, new_feature_test_df], axis=1)
            feature = pd.DataFrame(X_train_copy[featurename]).to_numpy()
        # X = (pd.concat([X_train_copy, X_test_copy]).replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy())
        # y = pd.concat([y_train, y_test]).to_numpy()
        X = X_train_copy.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
        y = y_train.to_numpy()
        # feature_metadata_mfe, feature_metadata_names, feature_metadata_groups = get_mfe_feature_metadata(feature)
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
    result_matrix = pd.read_parquet("src/Metadata/core/Core_Matrix_Example.parquet")
    columns = get_additional_mfe_columns()
    result_matrix_columns = result_matrix.columns.values.tolist()
    columns = columns + result_matrix_columns
    result_matrix_pandas = pd.DataFrame(columns=columns)
    for dataset, _ in result_matrix.groupby('dataset - id'):
        print("Dataset: " + str(dataset))
        X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(int(str(dataset)))
        result_matrix_dataset = result_matrix[result_matrix['dataset - id'] == dataset]
        result_matrix_dataset, dataset_metadata_general_names, dataset_metadata_statistical_names, dataset_metadata_info_theory_names, dataset_metadata_landmarking_names, dataset_metadata_complexity_names, dataset_metadata_clustering_names, dataset_metadata_concept_names, dataset_metadata_itemset_names = add_mfe_metadata_columns(X_train, y_train, result_matrix_dataset)
        new_column_names = list(dataset_metadata_general_names + dataset_metadata_statistical_names + dataset_metadata_info_theory_names + dataset_metadata_landmarking_names + dataset_metadata_complexity_names + dataset_metadata_clustering_names + dataset_metadata_concept_names + dataset_metadata_itemset_names)
        result_matrix_pandas.columns = result_matrix_columns[0] + result_matrix_columns[1] + result_matrix_columns[2] + new_column_names + result_matrix_columns[3] + result_matrix_columns[4]
        result_matrix_pandas = pd.concat([result_matrix_pandas, result_matrix_dataset], axis=0)
        result_matrix_pandas.to_parquet("src/Metadata/mfe/MFE_all_Matrix_Complete" + str(dataset) + ".parquet")
    result_matrix_pandas.to_parquet("src/Metadata/mfe/MFE_all_Matrix_Complete.parquet")


if __name__ == '__main__':
    main()
