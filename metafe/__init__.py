"""MetaFE: meta-learned, model-aware feature engineering with a surrogate model.

Pipeline (each step is a module with a command line interface, see README.md):
  1. metafe.core_matrix   labels: improvement of AutoGluon's validation score per candidate feature and dataset
  2. metafe.metafeatures  pandas metafeatures of every candidate (training data of the surrogate)
  3. metafe.surrogate     CatBoost surrogate that predicts the improvement of a candidate
  4. metafe.metafe        recursive feature engineering with the surrogate, for several time limits at once
  5. metafe.evaluation    baselines (OpenFE, TabPrep), evaluation with AutoGluon, checks and plots
"""
