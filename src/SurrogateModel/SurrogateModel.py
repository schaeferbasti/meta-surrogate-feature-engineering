import pandas as pd
import numpy as np

from sklearn.metrics import root_mean_squared_error
from autogluon.tabular import TabularPredictor

from src.SurrogateModel.Autogluon_MultilabelPredictor import MultilabelPredictor
from src.utils.create_feature_and_featurename import create_featurenames
from src.utils.get_dataset import get_openml_dataset_and_metadata
from src.utils.get_matrix import get_matrix_columns
from src.utils.preprocess_data import factorize_data


def add_metadata(X_predict, dataset_metadata, models):
    columns = get_matrix_columns()
    X_predict_new = pd.DataFrame(columns=columns)
    X_predict = factorize_data(X_predict)
    for model in models:
        for featurename in X_predict.columns:
            feature_df = pd.DataFrame(X_predict, columns=[featurename])
            feature_datatype = feature_df[featurename].dtype
            if feature_datatype == np.number:
                feature_pandas_description = feature_df.describe(include=np.number)
                feature_metadata_numeric = {"feature - name": featurename,
                                            "feature - count": feature_pandas_description.iloc[0].values[0],
                                            "feature - mean": feature_pandas_description.iloc[1].values[0],
                                            "feature - std": feature_pandas_description.iloc[2].values[0],
                                            "feature - min": feature_pandas_description.iloc[3].values[0],
                                            "feature - max": feature_pandas_description.iloc[4].values[0],
                                            "feature - lower percentile": feature_pandas_description.iloc[5].values[0],
                                            "feature - 50 percentile": feature_pandas_description.iloc[6].values[0],
                                            "feature - upper percentile": feature_pandas_description.iloc[7].values[0]}
                X_predict_new.loc[len(X_predict_new)] = [dataset_metadata["task_id"], dataset_metadata["task_type"], dataset_metadata["number_of_classes"], feature_metadata_numeric["feature - name"], str(feature_datatype), int(feature_metadata_numeric["feature - count"]), feature_metadata_numeric["feature - mean"], feature_metadata_numeric["feature - std"], feature_metadata_numeric["feature - min"], feature_metadata_numeric["feature - max"], feature_metadata_numeric["feature - lower percentile"], feature_metadata_numeric["feature - 50 percentile"], feature_metadata_numeric["feature - upper percentile"], None, None, None, model, None]
            else:
                feature_pandas_description = feature_df.describe()
                feature_metadata_categorical = {"feature - name": featurename,
                                                "feature - count": feature_pandas_description.iloc[0].values[0],
                                                "feature - unique": feature_pandas_description.iloc[1].values[0],
                                                "feature - top": feature_pandas_description.iloc[2].values[0],
                                                "feature - freq": feature_pandas_description.iloc[3].values[0]}
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


def predict_operators_for_models(train_data, X_test, models):
    # Prepare Data
    X_test = add_new_featurenames(X_test)
    label = 'improvement'  # train_data["improvement"].name

    # Predictor
    predictor = TabularPredictor(
        label=label,
        eval_metric="root_mean_squared_error",  # roc_auc (binary), log_loss (multiclass)
        problem_type="regression",  # binary, multiclass
        verbosity=-1,
    )
    predictor.fit(
        time_limit=int(30),
        memory_limit=8 * 1024 * 1024,
        num_cpus=8,
        num_gpus=0,
        train_data=train_data,
        #presets="best_quality",
        dynamic_stacking=False,
        hyperparameters=models,
        # Validation Protocol
        num_bag_folds=8,
        num_bag_sets=1,
        num_stack_levels=0,
    )
    # Evaluation
    # evaluation = pd.DataFrame(predictor.evaluate(X_test, ))
    # Prediction
    prediction = predictor.predict(X_test)
    prediction.rename("predicted_improvement", inplace=True)
    prediction_result = pd.concat([X_test[["dataset - id", "feature - name", "model"]], prediction], axis=1)
    return prediction_result  # evaluation,


def multi_predict_operators_for_models(train_data, X_test):
    labels = ['feature - name', 'improvement']  # which columns to predict based on the others
    problem_types = ['multiclass', 'regression']  # type of each prediction problem
    save_path = 'agModels'  # specifies folder to store trained models
    time_limit = 5  # how many seconds to train the TabularPredictor for each label, set much larger in your applications!
    # Multi Predictor
    multi_predictor = MultilabelPredictor(labels=labels, problem_types=problem_types, path=save_path)
    multi_predictor.fit(train_data, labels, time_limit=time_limit)
    # Evaluation
    #  multi_evaluation = pd.DataFrame(multi_predictor.evaluate(X_test))
    # Prediction
    multi_prediction = pd.DataFrame(multi_predictor.predict(X_test))
    multi_prediction.rename(columns={"feature - name": "new - feature - name", "improvement": "predicted_improvement"}, inplace=True)
    multi_prediction_result = pd.concat([X_test[["dataset - id", "feature - name", "model"]], multi_prediction], axis=1)
    return multi_prediction_result  # multi_evaluation,


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
