import os
import time

import numpy as np
import pandas as pd
import scipy
from sklearn.decomposition import PCA
from tabpfn import TabPFNClassifier

from src.utils.create_feature_and_featurename import create_feature
from src.utils.get_data import get_openml_dataset_split_and_metadata
from src.utils.get_matrix import get_additional_tabpfn_columns


def get_tabpfn_embedding(X, y):
    # os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    clf = TabPFNClassifier()
    clf.fit(X, y)
    embeddings = clf.get_embeddings(X)
    return embeddings


def add_tabpfn_metadata_columns(X_train, y_train, result_matrix):
    columns = get_additional_tabpfn_columns()
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
            X_train_copy = pd.concat([X_train_copy, new_feature_df], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0)
        embedding = get_tabpfn_embedding(X_train_copy, y_train)
        # Norm
        embedding_norm_representation = np.linalg.norm(embedding)
        # PCA
        embedding_reshaped = embedding.reshape(-1, 192)
        pca = PCA(n_components=1)
        pca_1d = pca.fit_transform(embedding_reshaped)
        embedding_pca_representation = np.mean(pca_1d)
        # Mean
        embedding_mean_representation = np.mean(embedding)
        embeddings = [embedding_norm_representation, embedding_pca_representation, embedding_mean_representation]
        new_row = pd.DataFrame([embeddings], columns=columns)
        matching_indices = result_matrix[result_matrix["feature - name"] == str(featurename)].index
        for idx in matching_indices:
            new_columns.loc[idx] = new_row.iloc[0]
    insert_position = result_matrix.shape[1] - 2
    result_matrix = pd.concat([result_matrix.iloc[:, :insert_position], new_columns, result_matrix.iloc[:, insert_position:]], axis=1)
    return result_matrix


def main():
    result_matrix = pd.read_parquet("src/Metadata/core/Core_Matrix_Complete.parquet")
    columns = get_additional_tabpfn_columns()
    result_matrix_columns = result_matrix.columns.values.tolist()
    columns = columns + result_matrix_columns
    result_matrix_pandas = pd.DataFrame(columns=columns)
    start = time.time()
    counter = 0
    for dataset, _ in result_matrix.groupby('dataset - id'):
        print("Dataset: " + str(dataset))
        try:
            pd.read_parquet("src/Metadata/tabpfn/TabPFN_Matrix_Complete" + str(dataset) + ".parquet")
        except FileNotFoundError:
            try:
                counter += 1
                start_dataset = time.time()
                X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(int(str(dataset)))
                result_matrix_dataset = result_matrix[result_matrix['dataset - id'] == dataset]
                result_matrix_dataset = add_tabpfn_metadata_columns(X_train, y_train, result_matrix_dataset)
                result_matrix_pandas.columns = result_matrix_dataset.columns
                result_matrix_pandas = pd.concat([result_matrix_pandas, result_matrix_dataset], axis=0)
                result_matrix_pandas.to_parquet("src/Metadata/tabpfn/TabPFN_Matrix_Complete" + str(dataset) + ".parquet")
                end_dataset = time.time()
                print("Time for TabPFN on Dataset " + str(dataset) + ": "+ str(end_dataset - start_dataset))
            except TypeError:
                continue
    result_matrix_pandas.to_parquet("src/Metadata/tabpfn/TabPFN_Matrix_Complete.parquet")
    end = time.time()
    print("Time for complete TabPFN MF: " + str(end - start) + " on " + str(counter) + " datasets.")


if __name__ == '__main__':
    main()
