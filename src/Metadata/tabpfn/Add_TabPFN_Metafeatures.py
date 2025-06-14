import os

import pandas as pd
import scipy
from tabpfn import TabPFNClassifier

from src.utils.create_feature_and_featurename import create_feature
from src.utils.get_data import get_openml_dataset_split_and_metadata
from src.utils.get_matrix import get_additional_tabpfn_columns


def get_tabpfn_embedding(X, y):
    os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    clf = TabPFNClassifier()
    clf.fit(X, y)
    embeddings = clf.get_embeddings(X)
    return embeddings


def add_tabpfn_metadata_columns(X_train, y_train, result_matrix):
    columns = get_additional_tabpfn_columns(str(result_matrix.columns[0]), result_matrix.columns[1])
    new_columns = pd.DataFrame(index=result_matrix.index, columns=columns)
    for row in result_matrix.iterrows():
        featurename = row[1][1]
        X_train_copy = X_train.copy()
        if featurename.startswith("without"):
            feature_to_delete = featurename.split(" - ")[1]
            X_train_copy = X_train_copy.drop(feature_to_delete, axis=1)
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
            new_feature_df = pd.DataFrame(new_feature, columns=[featurename])
            new_feature_df.index = X_train_copy.index
            X_train_copy = pd.concat([X_train_copy, new_feature_df], axis=1)
        embedding = get_tabpfn_embedding(X_train_copy, y_train)
        embedding_representation = scipy.linalg.norm(embedding)
        matching_indices = result_matrix[result_matrix["feature - name"] == str(featurename)].index
        for idx in matching_indices:
            new_columns.loc[idx] = embedding_representation
    insert_position = result_matrix.shape[1] - 2
    result_matrix = pd.concat(
        [result_matrix.iloc[:, :insert_position], new_columns, result_matrix.iloc[:, insert_position:]], axis=1)
    return result_matrix


def main():
    result_matrix = pd.read_parquet("../core/Core_Matrix_Example.parquet")
    for dataset, _ in result_matrix.groupby('dataset - id'):
        print("Dataset: " + str(dataset))
        X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset)
        result_matrix = add_tabpfn_metadata_columns(X_train, y_train, result_matrix)
        result_matrix.to_parquet("TabPFN_Matrix_1_" + str(dataset) + ".parquet")
    result_matrix.to_parquet("TabPFN_Matrix_1.parquet")


if __name__ == '__main__':
    main()
