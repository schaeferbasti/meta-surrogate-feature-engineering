import pandas as pd

from autogluon.tabular import TabularPredictor

from Autogluon_MultilabelPredictor import MultilabelPredictor
from src.SurrogateModel.SurrogateModel import add_new_featurenames
from tabrepo_2024_custom import zeroshot2024


def run_autogluon_lgbm(X_train, y_train, zeroshot=False):
    label = "target"
    X_train["target"] = y_train

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
        eval_metric="log_loss",  # roc_auc (binary), log_loss (multiclass)
        problem_type="multiclass",  # binary, multiclass
        verbosity=-1,
    )

    predictor.fit(
        time_limit=int(60 * 60 * 4),
        memory_limit=48 * 1024 * 1024,
        num_cpus=8,
        num_gpus=0,
        train_data=X_train,
        presets="best_quality",
        dynamic_stacking=False,
        hyperparameters=zeroshot2024,
        # Validation Protocol
        num_bag_folds=8,
        num_bag_sets=1,
        num_stack_levels=0,
    )
    predictor.fit_summary(verbosity=-1)
    data = pd.concat([X_train, y_train], axis=1)
    lb = predictor.leaderboard(data)
    return lb


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