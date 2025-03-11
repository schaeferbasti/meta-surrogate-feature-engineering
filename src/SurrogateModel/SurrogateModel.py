import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from autogluon.tabular.src.autogluon.tabular import TabularPredictor
from tabrepo_2024_custom import zeroshot2024


def run_autogluon_lgbm(X_train, y_train, X_test, y_test, zeroshot=False):
    label = y_train.columns[0]
    train_data = pd.concat([X_train, y_train], axis=1)
    allowed_models = [
        "GBM",
    ]
    for k in list(zeroshot2024.keys()):
        if k not in allowed_models:
            del zeroshot2024[k]
        else:
            if not zeroshot:
                zeroshot2024[k] = zeroshot2024[k][:1]
    # -- Run AutoGluon
    predictor = TabularPredictor(
        label=label,
        eval_metric="root_mean_squared_error",  # roc_auc (binary), log_loss (multiclass)
        problem_type="regression",  # binary, multiclass
        verbosity=-1,
    )
    predictor.fit(
        time_limit=int(60 * 60 * 4),
        memory_limit=48 * 1024 * 1024,
        num_cpus=8,
        num_gpus=0,
        train_data=train_data,
        presets="best_quality",
        dynamic_stacking=False,
        hyperparameters=zeroshot2024,
        # Validation Protocol
        num_bag_folds=8,
        num_bag_sets=1,
        num_stack_levels=0,
    )
    predictor.fit_summary(verbosity=-1)
    predictions = predictor.predict(X_test)
    predictions.rename("predicted_score", inplace=True)
    rmse = root_mean_squared_error(y_true=y_test, y_pred=predictions)
    print(rmse)
    predictions_df = pd.concat([X_test, y_test, predictions], axis=1)
    print(predictions_df)
    return predictions_df, rmse  # , lb


def main():
    result_matrix = pd.read_parquet("../Metadata/Operator_Model_Feature_Matrix_1.parquet")
    X = result_matrix.drop('score', axis=1)
    y = result_matrix["score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = pd.DataFrame(X_train)
    y_train = pd.DataFrame(y_train)
    X_test = pd.DataFrame(X_test)
    y_test = pd.DataFrame(y_test)
    prediction, rmse = run_autogluon_lgbm(X_train, y_train, X_test, y_test, zeroshot=False)
    prediction.to_parquet("Prediction.parquet")
    best_operations = prediction[prediction.predicted_score == prediction.predicted_score.max()]
    best_operations.to_parquet("Best Operations.parquet")
    print(best_operations)


if __name__ == '__main__':
    main()
