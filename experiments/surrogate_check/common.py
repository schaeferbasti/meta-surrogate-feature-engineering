"""Shared helpers of the surrogate check (handoff 2026-10-26, steps 1-3).

Target-aware metafeatures ("v2") of a candidate feature, computed on the training rows only:
  - how much the feature tells about y (mutual information) and about the residual of a LightGBM fitted on the
    current columns (= information the model does not have yet), and the gain of both over the operands,
  - redundancy with the existing columns (max |Spearman|),
  - scale-free distribution statistics,
  - dataset-level metafeatures (size, columns, classes, categorical share, baseline difficulty).
"""
import os
import warnings
from pathlib import Path

import numpy as np
import openml
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(os.environ.get("SURROGATE_CHECK_DIR", REPO / "results" / "surrogate_check"))
# Knowledge base whose labels are checked; SURROGATE_CHECK_KB=lightgbm_cv for the repeated-CV labels (use another
# SURROGATE_CHECK_DIR then, the metafeatures and results depend on the knowledge base)
KB_MATRIX = REPO / "data" / os.environ.get("SURROGATE_CHECK_KB", "lightgbm_reuse") / "metafeature_matrix.parquet"
OPENML_CACHE = os.environ.get("OPENML_CACHE", str(REPO.parent / "openml_cache"))
openml.config.set_root_cache_directory(OPENML_CACHE)

from metafe.data import load_task, is_classification  # noqa: E402
from metafe.operators import _to_numeric, compute_feature, parse_featurename, DELETE, is_categorical  # noqa: E402

STAT_ROWS = 3000        # subsample (fixed seed) for the statistics of a candidate
N_BINS = 16             # quantile bins of the correlation ratio
OLD_MF = ["feature - count", "feature - unique", "feature - freq", "feature - mean", "feature - std", "feature - min",
          "feature - 25%", "feature - 50%", "feature - 75%", "feature - max"]
NEW_MF = ["n_rows_log", "n_cols", "n_classes", "frac_cat_cols", "baseline_difficulty",
          "nan_frac", "unique_ratio", "top_freq", "skew", "kurt",
          "mi_y", "mi_y_gain", "eta_y", "eta_y_gain", "eta_res", "eta_res_gain", "rho_res", "rho_res_gain",
          "max_corr_cols", "max_corr_operands", "operand_cat_any", "operand_unique_ratio_min", "depth"]


def lgb_params(classification, n_classes, seed=0, n_estimators=200):
    p = dict(n_estimators=n_estimators, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=1)
    if classification:
        p["objective"] = "multiclass" if n_classes > 2 else "binary"
    return p


def prepare_X(X, reference=None):
    """LightGBM-ready frame: non-numeric/bool columns -> pandas category (categories of the reference frame)."""
    X = X.copy()
    for c in X.columns:
        ref = X[c] if reference is None else reference[c]
        if is_categorical(ref):
            cats = pd.Index(pd.Series(ref.astype(object)).dropna().astype(str).unique()).sort_values()
            X[c] = pd.Categorical(X[c].astype(object).where(X[c].notna(), None).map(
                lambda v: None if v is None else str(v)), categories=cats)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X


def encode_y(y):
    """Classification target as codes 0..K-1 (sorted)."""
    cats = np.sort(y.astype(str).unique())
    return y.astype(str).map({c: i for i, c in enumerate(cats)}).to_numpy(), len(cats)


def oof_predictions(X, y, classification, n_classes, seed=0, n_splits=5, max_rows=20000, n_estimators=200):
    """Out-of-fold predictions of LightGBM on (a subsample of) the training rows: (rows, oof, score).
    score: log loss (classification) or RMSE (regression); oof: class probabilities (n x K) or values."""
    import lightgbm as lgb
    from sklearn.model_selection import KFold, StratifiedKFold

    rng = np.random.RandomState(seed)
    rows = np.arange(len(X))
    if len(rows) > max_rows:
        rows = np.sort(rng.choice(rows, max_rows, replace=False))
    Xs, ys = prepare_X(X.iloc[rows]), y[rows]
    Xs.columns = [f"f{i}" for i in range(Xs.shape[1])]  # LightGBM rejects JSON characters in names
    if classification:
        oof = np.zeros((len(rows), n_classes))
        try:
            folds = list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(Xs, ys))
        except ValueError:
            folds = list(KFold(n_splits, shuffle=True, random_state=seed).split(Xs))
    else:
        oof = np.zeros(len(rows))
        folds = list(KFold(n_splits, shuffle=True, random_state=seed).split(Xs))
    for tr, va in folds:
        model = (lgb.LGBMClassifier if classification else lgb.LGBMRegressor)(
            **lgb_params(classification, n_classes, seed, n_estimators))
        model.fit(Xs.iloc[tr], ys[tr])
        if classification:
            proba = model.predict_proba(Xs.iloc[va])
            full = np.full((len(va), n_classes), 1e-6)
            full[:, model.classes_.astype(int)] = proba
            oof[va] = full / full.sum(1, keepdims=True)
        else:
            oof[va] = model.predict(Xs.iloc[va])
    return rows, oof, fold_scores(ys, oof, folds, classification)


def fold_scores(y, oof, folds, classification):
    """Per-fold loss (lower is better)."""
    out = []
    for _, va in folds:
        if classification:
            p = np.clip(oof[va, :][np.arange(len(va)), y[va]], 1e-15, 1)
            out.append(-np.mean(np.log(p)))
        else:
            out.append(np.sqrt(np.mean((y[va] - oof[va]) ** 2)))
    return np.array(out)


def numeric(s):
    """Feature as float array (sorted category codes for non-numeric values)."""
    v, _ = _to_numeric(pd.Series(s).reset_index(drop=True))
    return np.array(v, dtype="float64")


def rank01(v):
    r = np.array(pd.Series(v).rank(method="average"), dtype="float64")
    r[np.isnan(v)] = np.nanmean(r) if np.isfinite(r).any() else 0.0
    return r


def corr_ratio(v, R, n_bins=N_BINS):
    """Chance-corrected correlation ratio eta^2 of the residual matrix R (n x K) over quantile bins of v
    (NaN = own bin), summed over the K columns and divided by the total variance."""
    v = np.asarray(v, dtype="float64")
    codes = np.full(len(v), -1)
    ok = np.isfinite(v)
    if ok.sum() > 0:
        uniq = np.unique(v[ok])
        if len(uniq) <= n_bins:
            codes[ok] = np.searchsorted(uniq, v[ok])
        else:
            edges = np.unique(np.quantile(v[ok], np.linspace(0, 1, n_bins + 1)[1:-1]))
            codes[ok] = np.searchsorted(edges, v[ok], side="right")
    codes = codes + 1
    n_groups = codes.max() + 1
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    R = R.reshape(len(v), -1)
    R = R - R.mean(0)
    total = (R ** 2).sum()
    if total <= 0:
        return 0.0
    between = 0.0
    for k in range(R.shape[1]):
        sums = np.bincount(codes, weights=R[:, k], minlength=n_groups)
        between += (sums[counts > 0] ** 2 / counts[counts > 0]).sum()
    used = (counts > 0).sum()
    return float(between / total - (used - 1) / max(len(v) - 1, 1))  # minus the expectation under independence


def max_abs_spearman(v, R):
    rv = rank01(v)
    if np.std(rv) == 0:
        return 0.0
    rv = (rv - rv.mean()) / rv.std()
    R = R.reshape(len(v), -1)
    out = 0.0
    for k in range(R.shape[1]):
        rk = rank01(R[:, k])
        if np.std(rk) > 0:
            out = max(out, abs(float(np.mean(rv * (rk - rk.mean()) / rk.std()))))
    return out


def mutual_info(v, y, classification):
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    finite = np.isfinite(v)
    v = np.where(finite, v, np.median(v[finite]) if finite.any() else 0.0).reshape(-1, 1)  # median of finite values only
    if np.std(v) == 0:
        return 0.0
    f = mutual_info_classif if classification else mutual_info_regression
    return float(f(v, y, n_neighbors=3, random_state=0)[0])


class DatasetContext:
    """Training rows of a dataset with everything needed to describe candidates: subsample, encoded target,
    residual of LightGBM on the current columns, ranks of the current columns."""

    def __init__(self, X_train, y_train, metadata, seed=0):
        self.X = X_train
        self.classification = is_classification(metadata)
        if self.classification:
            self.y, self.n_classes = encode_y(y_train)
        else:
            self.y, self.n_classes = pd.to_numeric(y_train).to_numpy(dtype="float64"), 0
        rows, oof, scores = oof_predictions(X_train, self.y, self.classification, self.n_classes, seed)
        rng = np.random.RandomState(seed + 1)
        sub = rng.choice(len(rows), min(STAT_ROWS, len(rows)), replace=False)
        self.stat_rows = rows[sub]  # positions in X_train
        ys = self.y[self.stat_rows]
        if self.classification:
            onehot = np.eye(self.n_classes)[ys]
            self.R = onehot - oof[sub]
            self.Y = onehot - onehot.mean(0)
            prior = np.bincount(self.y, minlength=self.n_classes) / len(self.y)
            entropy = -np.sum(prior[prior > 0] * np.log(prior[prior > 0]))
            self.baseline_difficulty = float(scores.mean() / max(entropy, 1e-9))
        else:
            sd = np.std(self.y) or 1.0
            self.R = ((ys - oof[sub]) / sd).reshape(-1, 1)
            self.Y = ((ys - ys.mean()) / sd).reshape(-1, 1)
            self.baseline_difficulty = float(scores.mean() / sd)
        self.ys = ys
        self.dataset_mf = {
            "n_rows_log": float(np.log10(len(X_train))), "n_cols": float(X_train.shape[1]),
            "n_classes": float(self.n_classes),
            "frac_cat_cols": float(np.mean([is_categorical(X_train[c]) for c in X_train.columns])),
            "baseline_difficulty": self.baseline_difficulty}
        self.col_ranks = {}
        self._operand_cache = {}
        self._derived = {}  # training values of nested operands that are not columns of X

    def values(self, c):
        """Training values of column or (nested) feature name c."""
        if c in self.X.columns:
            return self.X[c]
        if c not in self._derived:
            _, operands = parse_featurename(c)
            frame = pd.DataFrame({o: self.values(o).to_numpy() for o in operands}, index=self.X.index)
            self._derived[c], _ = compute_feature(c, frame)
        return self._derived[c]

    def column_rank(self, c):
        if c not in self.col_ranks:
            r = rank01(numeric(self.X[c].iloc[self.stat_rows]))
            self.col_ranks[c] = (r - r.mean()) / (r.std() or 1.0)
        return self.col_ranks[c]

    def target_stats(self, v):
        """(mi_y, eta_y, eta_res, rho_res) of a feature subsampled to stat_rows."""
        return (mutual_info(v, self.ys, self.classification), corr_ratio(v, self.Y), corr_ratio(v, self.R),
                max_abs_spearman(v, self.R))

    def operand_stats(self, c):
        if c not in self._operand_cache:
            v = numeric(self.values(c).iloc[self.stat_rows])
            self._operand_cache[c] = self.target_stats(v) + (len(np.unique(v[np.isfinite(v)])) / len(v),)
        return self._operand_cache[c]

    def describe(self, name, feature=None):
        """Dict of the NEW_MF metafeatures of candidate `name` (feature = its training values, computed if None)."""
        operator, operands = parse_featurename(name)
        if operator == DELETE:
            feature = self.X[operands[0]]
            operands = []  # deleting: describe the column itself, no operand gain
        elif feature is None:
            feature = self.values(name) if any(o not in self.X.columns for o in operands) else \
                compute_feature(name, self.X)[0]
        full = numeric(feature)
        v = full[self.stat_rows]
        finite = v[np.isfinite(v)]
        mf = dict(self.dataset_mf)
        mf["nan_frac"] = float(np.mean(~np.isfinite(full)))
        mf["unique_ratio"] = float(len(np.unique(finite)) / len(v)) if len(v) else 0.0
        if len(finite):
            _, counts = np.unique(finite, return_counts=True)
            mf["top_freq"] = float(counts.max() / len(v))
            s = pd.Series(finite)
            mf["skew"] = float(np.clip(s.skew(), -50, 50)) if len(finite) > 2 else 0.0
            mf["kurt"] = float(np.clip(s.kurt(), -50, 500)) if len(finite) > 3 else 0.0
        else:
            mf.update(top_freq=1.0, skew=0.0, kurt=0.0)
        mi, eta_y, eta_res, rho_res = self.target_stats(v)
        mf.update(mi_y=mi, eta_y=eta_y, eta_res=eta_res, rho_res=rho_res)
        ops = [self.operand_stats(c) for c in operands]
        best = np.max(np.array([o[:4] for o in ops]), axis=0) if ops else np.zeros(4)
        mf.update(mi_y_gain=mi - best[0], eta_y_gain=eta_y - best[1], eta_res_gain=eta_res - best[2],
                  rho_res_gain=rho_res - best[3])
        rv = rank01(v)
        rv = (rv - rv.mean()) / (rv.std() or 1.0)
        corrs = {c: abs(float(np.mean(rv * self.column_rank(c)))) for c in self.X.columns
                 if not (operator == DELETE and c == name[len("without - "):])}
        mf["max_corr_cols"] = max(corrs.values()) if corrs else 0.0
        mf["max_corr_operands"] = max([corrs.get(c, 0.0) for c in operands], default=0.0)
        mf["operand_cat_any"] = float(any(is_categorical(self.values(c)) for c in operands))
        mf["operand_unique_ratio_min"] = float(min([o[4] for o in ops], default=mf["unique_ratio"]))
        mf["depth"] = float(name.count("("))
        return mf
