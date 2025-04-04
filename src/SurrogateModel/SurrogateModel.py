import pandas as pd

from src.utils.get_data import get_openml_dataset_and_metadata
from src.utils.get_matrix import get_matrix_columns
from src.utils.get_metafeatures import get_numeric_pandas_metafeatures, get_categorical_pandas_metafeatures
from src.utils.preprocess_data import factorize_data
from src.utils.run_models import multi_predict_autogluon_lgbm, predict_autogluon_lgbm


def add_metadata(X_predict, dataset_metadata, models):
    columns = get_matrix_columns()
    X_predict_new = pd.DataFrame(columns=columns)
    X_predict = factorize_data(X_predict)
    for model in models:
        for featurename in X_predict.columns:
            feature_df = pd.DataFrame(X_predict, columns=[featurename])
            feature_datatype = feature_df[featurename].dtype
            if pd.api.types.is_numeric_dtype(feature_df[featurename]):
                feature_metadata_numeric = get_numeric_pandas_metafeatures(feature_df, featurename)
                X_predict_new.loc[len(X_predict_new)] = [dataset_metadata["task_id"], dataset_metadata["task_type"], dataset_metadata["number_of_classes"], feature_metadata_numeric["feature - name"], "operator", str(feature_datatype), int(feature_metadata_numeric["feature - count"]), feature_metadata_numeric["feature - mean"], feature_metadata_numeric["feature - std"], feature_metadata_numeric["feature - min"], feature_metadata_numeric["feature - max"], feature_metadata_numeric["feature - lower percentile"], feature_metadata_numeric["feature - 50 percentile"], feature_metadata_numeric["feature - upper percentile"], None, None, None, model, None]
            else:
                feature_metadata_categorical = get_categorical_pandas_metafeatures(feature_df, featurename)
                X_predict_new.loc[len(X_predict_new)] = [dataset_metadata["task_id"], dataset_metadata["task_type"], dataset_metadata["number_of_classes"], feature_metadata_categorical["feature - name"], "operator", str(feature_datatype), int(feature_metadata_categorical["feature - count"]), None, None, None, None, None, None, None, feature_metadata_categorical["feature - unique"], feature_metadata_categorical["feature - top"], feature_metadata_categorical["feature - freq"], model, None]
    return X_predict_new


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
    prediction = predict_autogluon_lgbm(result_matrix, prediction_data, models=models)
    prediction.to_parquet("Prediction.parquet")
    #  evaluation, prediction, best_operations = predict_operators_for_models(X, y, X_predict, y_predict, models=models, zeroshot=False)
    #  evaluation.to_parquet('Evaluation.parquet')
    prediction_result = pd.read_parquet("Prediction.parquet")
    best_operations = prediction_result.nlargest(n=20, columns="predicted_improvement", keep="first")
    best_operations.to_parquet("Best_Operations.parquet")

    # Multi-predictor (features + operator & improvement)
    multi_prediction = multi_predict_autogluon_lgbm(result_matrix, prediction_data)
    multi_prediction.to_parquet('Multi_Prediction.parquet')
    #  multi_evaluation, multi_prediction = multi_predict_operators_for_models(X, y, X_predict, y_predict, models=models, zeroshot=False)
    #  multi_evaluation.to_parquet('Multi_Evaluation.parquet')
    multi_prediction_result = pd.read_parquet("Multi_Prediction.parquet")
    multi_best_operations = multi_prediction_result.nlargest(n=20, columns="predicted_improvement", keep="first")
    multi_best_operations.to_parquet("Multi_Best_Operations.parquet")


if __name__ == '__main__':
    dataset_id = 190411
    models = "GBM"
    main(dataset_id, models)
