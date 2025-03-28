import pandas as pd
import numpy as np
import logging
import os
import ray
import tempfile

from sklearn.metrics import log_loss
from autogluon.tabular import TabularPredictor
import lightgbm as lgb

from src.utils.Autogluon_MultilabelPredictor import MultilabelPredictor
from src.SurrogateModel.SurrogateModel import add_new_featurenames
from src.utils.tabrepo_2024_custom import zeroshot2024


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


def run_lgbm(X_train, y_train, X_test, y_test):
    """
    # lgb_train = lgb.Dataset(X_train, y_train, params={'verbose': -1})
    # lgb_eval = lgb.Dataset(X_test, y_test, params={'verbose': -1}, reference=lgb_train)

    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": {"l2", "l1"},
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    # gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=[lgb_eval], callbacks=[lgb.early_stopping(stopping_rounds=5))
    # y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration, verbose_eval=False)
    # log_loss_test = log_loss(y_test, y_pred)
    """

    clf = lgb.LGBMClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(y_test)
    print(y_pred)
    df_train = pd.concat([X_train, pd.Series(y_train)], axis=1)
    df_test = pd.concat([X_test, pd.Series(y_test)], axis=1)
    df_original = pd.concat([df_train, df_test], axis=0)
    labels = df_original.iloc[:, -1].unique()
    log_loss_test = log_loss(y_test, y_pred, labels=labels)
    print(log_loss_test)
    return log_loss_test

def run_autogluon_lgbm_cluster(X_train, y_train, X_test, y_test, zeroshot=False):
    log = logging.getLogger(__name__)
    ray_mem_in_gb = 48
    log.info(f"Running on SLURM, initializing Ray with unique temp dir with {ray_mem_in_gb}GB.")
    ray_mem_in_b = int(ray_mem_in_gb * (1024.0 ** 3))
    tmp_dir_base_path = "/tmp"
    ray_dir = f"{tmp_dir_base_path}"
    print(f"Start local ray instances. Using {os.environ.get('RAY_MEM_IN_GB')} GB for Ray.")
    ray.shutdown()
    ray.init(
        address="local",
        _memory=ray_mem_in_b,
        object_store_memory=int(ray_mem_in_b * 0.3),
        _temp_dir=ray_dir,
        include_dashboard=False,
        logging_level=logging.INFO,
        log_to_driver=True,
        num_gpus=0,
        num_cpus=8,
        ignore_reinit_error=True
    )

    label = "target"
    X_train["target"] = y_train
    X_test["target"] = y_test

    allowed_models = [
        "GBM",
    ]

    for k in list(zeroshot2024.keys()):
        if k not in allowed_models:
            del zeroshot2024[k]
        else:
            if not zeroshot:
                zeroshot2024[k] = zeroshot2024[k][:1]
            else:
                zeroshot2024[k] = zeroshot2024[k][1:]

    # -- Run AutoGluon
    predictor = TabularPredictor(
        label=label,
        eval_metric="log_loss",  # roc_auc (binary), log_loss (multiclass)
        problem_type="multiclass",  # binary, multiclass
        verbosity=0,
        path=tempfile.mkdtemp() + os.sep,
    )

    predictor.fit(
        time_limit=int(60 * 60 * 4),
        memory_limit=48,
        num_cpus=8,
        num_gpus=0,
        train_data=X_train,
        presets="best_quality",
        dynamic_stacking=False,
        hyperparameters=zeroshot2024,
        num_bag_folds=8,
        num_bag_sets=1,
        num_stack_levels=0,
        fit_weighted_ensemble=False
    )
    predictor.fit_summary(verbosity=-1)
    lb = predictor.leaderboard(X_test, display=True)
    log_loss_test = np.abs(lb.score_test[0])
    ray.shutdown()

    return log_loss_test


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


def test_fe_for_model(train_data, X_test, target_label, model):
    # Prepare Data
    label = target_label
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
        hyperparameters=model,
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