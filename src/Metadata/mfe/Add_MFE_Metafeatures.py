import pandas as pd

from src.utils.create_feature_and_featurename import create_feature
from src.utils.get_data import get_openml_dataset_split_and_metadata
from src.utils.get_matrix import get_additional_mfe_columns
from src.utils.get_metafeatures import get_mfe_metadata

def add_mfe_metadata_columns(X_train, y_train, X_test, y_test, result_matrix):
    columns = get_additional_mfe_columns(str(result_matrix.columns[0]), result_matrix.columns[1])
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
            X_train_copy = pd.concat([X_train_copy, new_feature_df])
        feature = pd.DataFrame(X_train_copy[featurename])
        X = pd.concat([X_train, X_test])
        y = pd.concat([y_train, y_test])
        feature_metadata_mfe = get_mfe_metadata(X, y, feature)
        new_row = pd.DataFrame(columns=columns)
        new_row.loc[len(result_matrix)] = [
            feature_metadata_mfe[1][4],  # "attr_to_inst"
            feature_metadata_mfe[1][35],  # "nr_inst"
            feature_metadata_mfe[1][46],  # "sparsity.mean"
        ]
        matching_indices = result_matrix[result_matrix["feature - name"] == str(featurename)].index
        for idx in matching_indices:
            new_columns.loc[idx] = new_row.iloc[0]
    insert_position = result_matrix.shape[1] - 2
    result_matrix = pd.concat([result_matrix.iloc[:, :insert_position], new_columns, result_matrix.iloc[:, insert_position:]], axis=1)
    return result_matrix


def main():
    result_matrix = pd.read_parquet("../core/Core_Matrix_Example.parquet")
    for dataset, _ in result_matrix.groupby('dataset - id'):
        print("Dataset: " + str(dataset))
        X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset)
        result_matrix = add_mfe_metadata_columns(dataset_metadata, X_train, y_train, X_test, y_test, result_matrix)
        result_matrix.to_parquet("MFE_Matrix_1_" + str(dataset) + ".parquet")
    result_matrix.to_parquet("MFE_Matrix_1.parquet")


if __name__ == '__main__':
    main()
