"""AutoGluon fits used for the core-matrix labels and for the evaluation.

Setup (unchanged from the experiments so far): 8-fold bagging, best_quality, no stacking, no weighted ensemble,
log_loss (classification, always as multiclass) / RMSE (regression); every model with its default config. score_val is
the out-of-fold validation score on the training rows, score_test the score on the test rows (higher is better).
Models in GPU_MODELS use a GPU if the job has one (AutoGluon gives them one GPU per bagging fold), all others the CPUs.
"""
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

from metafe.config import NUM_CPUS, MEMORY_LIMIT_GB, FIT_TIME_LIMIT_S, TARGET

# The models of the experiments: AutoGluon model key -> model name in the leaderboard (without "_BAG_L1"), used in
# the core matrix column "model". TABICL is TabICLv2 (AutoGluon 1.6 default checkpoints).
MODEL_NAMES = {"LR": "LinearModel", "RF": "RandomForest", "GBM": "LightGBM", "TABICL": "TabICL", "REALMLP": "RealMLP"}
GPU_MODELS = {"TABICL", "REALMLP"}


def model_name(model_key):
    if model_key not in MODEL_NAMES:
        raise ValueError(f"Unknown model key {model_key!r}, the models are {sorted(MODEL_NAMES)}")
    return MODEL_NAMES[model_key] + "_BAG_L1"


def autogluon_version():
    from autogluon.tabular import __version__
    return __version__


def available_model_keys():
    try:
        from autogluon.tabular.registry import ag_model_registry  # AutoGluon >= 1.3
        return set(ag_model_registry.keys)
    except ImportError:
        from autogluon.tabular.trainer.model_presets.presets import ag_model_register  # AutoGluon 1.2
        return set(ag_model_register.keys)


def check_setup(experiment, model_keys=None):
    """Fail early if the installed AutoGluon is not the experiment's version or lacks a model."""
    version = autogluon_version()
    if not version.startswith(experiment.autogluon):
        raise RuntimeError(f"Experiment {experiment.name} needs AutoGluon {experiment.autogluon}.x, installed is "
                           f"{version} (wrong venv? see README.md)")
    unknown = set(model_keys or experiment.models) - available_model_keys()
    if unknown:
        raise ValueError(f"Model keys {sorted(unknown)} are not available in AutoGluon {version}")


def num_gpus(model_keys):
    """GPUs given to AutoGluon: all GPUs of the job if a model can use one, else 0."""
    if not set(model_keys) & GPU_MODELS:
        return 0
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        return 0


def hyperparameters(model_keys, seed=0):
    """Default config of every model with the given seed (AutoGluon's default seed is 0)."""
    if autogluon_version().startswith("1.2"):
        # AutoGluon 1.2: only LightGBM is seeded (as in the cached results of experiment lightgbm, bit-exact)
        return {key: [{"seed_value": seed} if key == "GBM" else {}] for key in model_keys}
    # AutoGluon >= 1.4: seed of every model (of every bagging fold model)
    return {key: [{"ag_args_ensemble": {"model_random_seed": seed}}] for key in model_keys}


def fit_leaderboard(X_train, y_train, X_test, y_test, classification, model_keys=("GBM",), seed=0):
    """Fit the models on the training rows; leaderboard with score_val (out-of-fold) and, if test rows are given,
    score_test per model. Models that fail are missing from the leaderboard.

    seed = random state of the bagging fold split and of the models. The predictor directory is deleted afterwards.
    """
    from autogluon.tabular import TabularPredictor

    train_data = X_train.copy()
    train_data[TARGET] = y_train
    predictor = TabularPredictor(
        label=TARGET,
        eval_metric="log_loss" if classification else "root_mean_squared_error",
        problem_type="multiclass" if classification else "regression",
        verbosity=0,
        path=tempfile.mkdtemp() + os.sep,
        learner_kwargs={"random_state": seed},
    )
    try:
        predictor.fit(
            train_data=train_data,
            time_limit=FIT_TIME_LIMIT_S,
            memory_limit=MEMORY_LIMIT_GB,
            num_cpus=NUM_CPUS,
            num_gpus=num_gpus(model_keys),
            presets="best_quality",
            dynamic_stacking=False,
            hyperparameters=hyperparameters(model_keys, seed),
            num_bag_folds=8,
            num_bag_sets=1,
            num_stack_levels=0,
            fit_weighted_ensemble=False,
        )
        if X_test is None:
            return predictor.leaderboard()
        test_data = X_test.copy()
        test_data[TARGET] = y_test
        return predictor.leaderboard(test_data)
    finally:
        shutil.rmtree(predictor.path, ignore_errors=True)


def validation_scores(leaderboard):
    """{model name: |score_val|} (log loss / RMSE, lower is better)."""
    return {model: abs(score) for model, score in zip(leaderboard["model"], leaderboard["score_val"])}


def relative_improvement(original_score, modified_score):
    """Relative improvement of an error (lower is better); symmetric definition of the first core matrix."""
    if np.isclose(original_score, modified_score):
        return 0.0
    if original_score > modified_score:
        return (original_score - modified_score) / original_score
    return (original_score - modified_score) / modified_score


# ------------------------------------------------------------------------------------- repeated LightGBM CV labels

def lightgbm_frame(X):
    """X for plain LightGBM: non-numeric/boolean columns as pandas categories (sorted), all others float; columns
    renamed to f0, f1, ... (LightGBM rejects JSON characters, which feature names contain)."""
    from metafe.operators import is_categorical

    X = X.copy()
    for c in X.columns:
        if is_categorical(X[c]):
            values = X[c].astype(object).where(X[c].notna(), None).map(lambda v: None if v is None else str(v))
            X[c] = pd.Categorical(values, categories=sorted(values.dropna().unique()))
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    X.columns = [f"f{i}" for i in range(X.shape[1])]
    return X


def cv_folds(y, classification, repeats, n_folds):
    """Fold splits (train, validation positions) of every repeat r (shuffle seed r), stratified if possible."""
    from sklearn.model_selection import KFold, StratifiedKFold

    folds = []
    for r in range(repeats):
        try:
            if not classification:
                raise ValueError
            folds.append(list(StratifiedKFold(n_folds, shuffle=True, random_state=r).split(np.zeros(len(y)), y)))
        except ValueError:  # regression, or a class with fewer rows than folds
            folds.append(list(KFold(n_folds, shuffle=True, random_state=r).split(np.zeros(len(y)))))
    return folds


def cv_losses(X, y, classification, n_classes, folds, n_estimators=300):
    """Validation loss (log loss / RMSE, lower is better) of plain LightGBM per repeat and fold: array (repeats,
    folds). y: class codes 0..K-1 (metafe.data.encode_target) or values; the model seed is the repeat number.
    min_sum_hessian_in_leaf=1: with the default (1e-3), leaves of nearly pure rare classes get huge outputs and the
    multiclass fit diverges at random (shuttle: fold log loss 0.003 to 8.5); no effect on regression (hessian 1 per
    row, at least min_child_samples rows per leaf)."""
    import lightgbm as lgb

    X = lightgbm_frame(X)
    losses = np.zeros((len(folds), len(folds[0])))
    for r, repeat in enumerate(folds):
        for f, (train, valid) in enumerate(repeat):
            params = dict(n_estimators=n_estimators, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                          min_sum_hessian_in_leaf=1.0, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          random_state=r, verbose=-1, n_jobs=1)
            if classification:
                params["objective"] = "multiclass" if n_classes > 2 else "binary"
                model = lgb.LGBMClassifier(**params).fit(X.iloc[train], y[train])
                proba = np.full((len(valid), n_classes), 1e-6)
                proba[:, model.classes_.astype(int)] = model.predict_proba(X.iloc[valid])
                proba /= proba.sum(1, keepdims=True)
                losses[r, f] = -np.mean(np.log(np.clip(proba[np.arange(len(valid)), y[valid]], 1e-15, 1)))
            else:
                model = lgb.LGBMRegressor(**params).fit(X.iloc[train], y[train])
                losses[r, f] = np.sqrt(np.mean((y[valid] - model.predict(X.iloc[valid])) ** 2))
    return losses
