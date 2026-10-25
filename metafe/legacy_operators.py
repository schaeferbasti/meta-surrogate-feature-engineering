"""Training features of the first (leaky, row-order dependent) operator implementation.

Only used by metafe.core_matrix to decide whether an old label can be reused: a label only depends on the training
feature, so it stays valid where the fixed operators (metafe.operators) give the same training feature.
min/max/freq were O(n^2) (one min()/count() over the column per row) and are replicated exactly in O(n).
"""
import warnings

import numpy as np
import pandas as pd

from metafe.operators import canonical_operator


def _float_list(feature):
    try:
        return [float(x) for x in feature]
    except (ValueError, TypeError):
        return [float(x) for x in pd.factorize(feature)[0]]  # codes in order of appearance


def _lookup(stat, keys):
    return keys.apply(lambda x: stat.get(x, np.nan) if x in stat.index else np.nan).to_list()


def legacy_train_feature(operator, feature1, feature2=None):
    with warnings.catch_warnings():  # pandas FutureWarnings of the old groupby calls (behaviour kept as it was)
        warnings.simplefilter("ignore")
        return _legacy_train_feature(canonical_operator(operator), feature1, feature2)


def _legacy_train_feature(operator, feature1, feature2):
    if feature2 is None:
        values = _float_list(feature1)
        if operator in ("min", "max"):
            return [min(values) if operator == "min" else max(values)] * len(values)
        if operator == "freq":
            values = pd.Series(values)
            return values.map(values.value_counts(dropna=True)).fillna(0).tolist()  # list.count(nan) == 0
        f = {"abs": lambda x: abs(x), "log": lambda x: np.log(np.abs(x)), "sqrt": lambda x: np.sqrt(np.abs(x)),
             "square": np.square, "sigmoid": lambda x: 1 / (1 + np.exp(-x)), "round": np.floor,
             "residual": lambda x: x - np.floor(x)}[operator]
        with np.errstate(all="ignore"):
            return [f(x) for x in values]

    if operator in ("add", "subtract", "multiply", "divide"):
        try:
            a, b = [float(x) for x in feature1], [float(x) for x in feature2]
        except (ValueError, TypeError):
            a, b = [float(x) for x in pd.factorize(feature1)[0]], [float(x) for x in pd.factorize(feature2)[0]]
        f = {"add": lambda x, y: x + y, "subtract": lambda x, y: x - y, "multiply": lambda x, y: x * y,
             "divide": lambda x, y: x / y if y != 0 else x}[operator]
        return [f(x, y) for x, y in zip(a, b)]
    if operator in ("GroupByThenMin", "GroupByThenMax"):
        grouped = feature1.groupby(feature2)
        return _lookup(grouped.min() if operator == "GroupByThenMin" else grouped.max(), feature2)
    if operator in ("GroupByThenMean", "GroupByThenMedian", "GroupByThenStd"):
        grouped = pd.to_numeric(feature1, errors="coerce").groupby(feature2)
        stat = {"GroupByThenMean": grouped.mean, "GroupByThenMedian": grouped.median,
                "GroupByThenStd": grouped.std}[operator]()
        return _lookup(stat, feature2)
    if operator == "GroupByThenRank":
        return pd.to_numeric(feature1, errors="coerce").groupby(feature2).rank(ascending=True, pct=True).to_list()
    if operator == "GroupByThenNUnique":
        return _lookup(feature1.groupby(feature2).nunique(), feature2)
    combined = feature1.astype(str) + "_" + feature2.astype(str)
    combined[feature1.isna() | feature2.isna()] = np.nan
    if operator == "Combine":
        return pd.Series(combined.factorize()[0], index=feature1.index).astype("float64").to_list()
    if operator == "CombineThenFreq":
        return _lookup(combined.value_counts(), combined)
    raise NotImplementedError(f"Unknown operator {operator}")
