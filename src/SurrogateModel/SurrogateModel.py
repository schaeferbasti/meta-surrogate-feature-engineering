import pandas as pd
import numpy as np

from src.utils.create_feature_and_featurename import create_featurenames
from src.utils.get_dataset import get_openml_dataset_and_metadata
from src.utils.get_matrix import get_matrix_columns
from src.utils.get_metafeatures import get_numeric_pandas_metafeatures, get_categorical_pandas_metafeatures
from src.utils.preprocess_data import factorize_data
from src.utils.run_autogluon import multi_predict_operators_for_models, predict_operators_for_models


def add_metadata(X_predict, dataset_metadata, models):
    columns = get_matrix_columns()
    X_predict_new = pd.DataFrame(columns=columns)
    X_predict = factorize_data(X_predict)
    for model in models:
        for featurename in X_predict.columns:
            feature_df = pd.DataFrame(X_predict, columns=[featurename])
            feature_datatype = feature_df[featurename].dtype
            if feature_datatype == np.number:
                feature_metadata_numeric = get_numeric_pandas_metafeatures(feature_df, featurename)
                X_predict_new.loc[len(X_predict_new)] = [dataset_metadata["task_id"], dataset_metadata["task_type"], dataset_metadata["number_of_classes"], feature_metadata_numeric["feature - name"], str(feature_datatype), int(feature_metadata_numeric["feature - count"]), feature_metadata_numeric["feature - mean"], feature_metadata_numeric["feature - std"], feature_metadata_numeric["feature - min"], feature_metadata_numeric["feature - max"], feature_metadata_numeric["feature - lower percentile"], feature_metadata_numeric["feature - 50 percentile"], feature_metadata_numeric["feature - upper percentile"], None, None, None, model, None]
            else:
                feature_metadata_categorical = get_categorical_pandas_metafeatures(feature_df, featurename)
                X_predict_new.loc[len(X_predict_new)] = [dataset_metadata["task_id"], dataset_metadata["task_type"], dataset_metadata["number_of_classes"], feature_metadata_categorical["feature - name"], "operator", str(feature_datatype), int(feature_metadata_categorical["feature - count"]), None, None, None, None, None, None, None, feature_metadata_categorical["feature - unique"], feature_metadata_categorical["feature - top"], feature_metadata_categorical["feature - freq"], model, None]
    return X_predict_new


def add_new_featurenames(X_test):
    # Get new dataset with feature names and metafeatures and replicate each feature (=each row) x times, that we can repeat thus row with similar values, but instead of the feature name, we add a new name consisting of all available operators and respective features
    matrix_columns = get_matrix_columns()
    featurenames = create_featurenames(X_test["feature - name"].values)
    X_test_new = pd.DataFrame({
        matrix_columns[0]: np.repeat((data := X_test[matrix_columns[0]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[1]: np.repeat((data := X_test[matrix_columns[1]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[2]: np.repeat((data := X_test[matrix_columns[2]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[3]: np.repeat(np.array(featurenames), data.shape[0]).reshape(len(featurenames), data.shape[0]).T.flatten(),
        matrix_columns[4]: np.repeat((data := X_test[matrix_columns[4]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[5]: np.repeat((data := X_test[matrix_columns[5]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[6]: np.repeat((data := X_test[matrix_columns[6]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[7]: np.repeat((data := X_test[matrix_columns[7]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[8]: np.repeat((data := X_test[matrix_columns[8]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[9]: np.repeat((data := X_test[matrix_columns[9]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[10]: np.repeat((data := X_test[matrix_columns[10]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[11]: np.repeat((data := X_test[matrix_columns[11]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[12]: np.repeat((data := X_test[matrix_columns[12]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[13]: np.repeat((data := X_test[matrix_columns[13]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[14]: np.repeat((data := X_test[matrix_columns[14]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[15]: np.repeat((data := X_test[matrix_columns[15]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[16]: np.repeat((data := X_test[matrix_columns[16]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[17]: np.repeat((data := X_test[matrix_columns[17]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
    })
    return X_test_new


def main(dataset_id, models):
    # Select Models
    if models is None:
        models = {"GBM": {}, "RF": {}, "KNN": {}, "CAT": {}, "XGB": {}, "LR": {}, "FASTAI": {}, "AG_AUTOMM": {}, "NN_TORCH": {}}
    else:
        models = {"GBM": {}}
    # Read and Prepare Training Data
    result_matrix = pd.read_parquet("../Metadata/Operator_Model_Feature_Matrix_2_168911.parquet")

    # Read and Prepare Test Data
    X_predict, y_predict, dataset_metadata = get_openml_dataset_and_metadata(dataset_id)
    prediction_data = add_metadata(X_predict, dataset_metadata, models)
    prediction_data['improvement'] = 0

    # Single-predictor (improvement given all possible operations on features)
    prediction = predict_operators_for_models(result_matrix, prediction_data, models=models)
    prediction.to_parquet("Prediction.parquet")
    #  evaluation, prediction, best_operations = predict_operators_for_models(X, y, X_predict, y_predict, models=models, zeroshot=False)
    #  evaluation.to_parquet('Evaluation.parquet')
    prediction_result = pd.read_parquet("Prediction.parquet")
    best_operations = prediction_result.nlargest(n=20, columns="predicted_improvement", keep="all")
    best_operations.to_parquet("Best_Operations.parquet")

    # Multi-predictor (features + operator & improvement)
    multi_prediction = multi_predict_operators_for_models(result_matrix, prediction_data)
    multi_prediction.to_parquet('Multi_Prediction.parquet')
    #  multi_evaluation, multi_prediction = multi_predict_operators_for_models(X, y, X_predict, y_predict, models=models, zeroshot=False)
    #  multi_evaluation.to_parquet('Multi_Evaluation.parquet')
    multi_prediction_result = pd.read_parquet("Multi_Prediction.parquet")
    multi_best_operations = multi_prediction_result.nlargest(n=20, columns="predicted_improvement", keep="all")
    multi_best_operations.to_parquet("Multi_Best_Operations.parquet")


if __name__ == '__main__':
    dataset_id = 190411
    models = "GBM"
    main(dataset_id, models)
