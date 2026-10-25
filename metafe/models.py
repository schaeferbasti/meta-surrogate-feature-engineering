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
