"""Feature operators of MetaFE (the OpenFE operator set), fitted on the training rows and applied to train AND test rows.

A candidate feature is named like a function call, e.g. "add(a, b)", "freq(a)", "GroupByThenMean(a, b)", or
"without - a" for deleting column a. Names can be nested ("log(add(a, b))") once "add(a, b)" is a column.

All statistics (min/max, counts, group statistics, category codes) come from the training rows only, and all codes are
assigned in sorted order, so a feature never depends on the row order. (The first implementation numbered categories
by order of appearance; OpenML returns the training rows grouped by class, so those codes leaked the target.)
Test values never seen in the training rows (unseen category, combination or group) become NaN, counts become 0.
"""
import re

import numpy as np
import pandas as pd

UNARY_OPERATORS = ["min", "max", "freq", "abs", "log", "sqrt", "square", "sigmoid", "round", "residual"]
BINARY_OPERATORS = ["add", "subtract", "multiply", "divide", "GroupByThenMin", "GroupByThenMax", "GroupByThenMean",
                    "GroupByThenMedian", "GroupByThenStd", "GroupByThenRank", "Combine", "CombineThenFreq",
                    "GroupByThenNUnique"]
DELETE = "delete"
# Symbols of the arithmetic operators in old feature names
_ALIASES = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}


def canonical_operator(operator):
    return _ALIASES.get(operator, operator)


# ----------------------------------------------------------------------------------------------------------- names

def featurename(operator, feature1, feature2=None):
    operator = canonical_operator(operator)
    if operator == DELETE:
        return f"without - {feature1}"
    if feature2 is None:
        return f"{operator}({feature1})"
    return f"{operator}({feature1}, {feature2})"


def parse_featurename(name):
    """(operator, [operands]) of a feature name; nested operands are kept as names."""
    if name.startswith("without - "):
        return DELETE, [name[len("without - "):].strip()]
    match = re.match(r"([^\s(]+)\s*\((.*)\)$", name)
    if not match:
        raise ValueError(f"Not a feature name: {name!r}")
    operands, depth, current = [], 0, ""
    for char in match.group(2):
        if char == "," and depth == 0:
            operands.append(current.strip())
            current = ""
            continue
        depth += {"(": 1, ")": -1}.get(char, 0)
        current += char
    operands.append(current.strip())
    return canonical_operator(match.group(1)), operands


def candidate_featurenames(columns):
    """All candidate features for the given columns: delete each column, every binary and every unary operator."""
    names = [featurename(DELETE, c) for c in columns]
    names += [featurename(op, a, b) for a in columns for b in columns for op in BINARY_OPERATORS]
    names += [featurename(op, c) for c in columns for op in UNARY_OPERATORS]
    return names


# ------------------------------------------------------------------------------------------------------- operators

def _categorical_codes(train, test=None):
    # Codes of the sorted training categories; unseen test categories -> NaN
    categories = np.sort(train.dropna().astype(str).unique())
    mapping = {c: float(i) for i, c in enumerate(categories)}

    def encode(s):
        return s.astype(object).where(s.isna(), s.astype(str)).map(mapping).astype("float64")

    return encode(train), (None if test is None else encode(test))


def _to_numeric(train, test=None):
    # Numeric columns stay as they are, all other columns are replaced by sorted category codes
    try:
        train_num = pd.to_numeric(train.astype(object), errors="raise").astype("float64")
        test_num = None if test is None else pd.to_numeric(test.astype(object), errors="raise").astype("float64")
        return train_num, test_num
    except (ValueError, TypeError):
        return _categorical_codes(train, test)


def _keys(s):
    # Group keys as plain objects, so that categorical dtypes map like their values
    return s.astype(object)


def _map_group_statistic(stat, train_key, test_key):
    stat = stat.copy()
    stat.index = stat.index.astype(object)
    mapping = stat.to_dict()
    return _keys(train_key).map(mapping), (None if test_key is None else _keys(test_key).map(mapping))


def _rank_against_train(train_values, train_key, test_values, test_key):
    # Percentile of each test value within the training values of its group (mid-rank, like rank(pct=True))
    result = pd.Series(np.nan, index=test_values.index, dtype="float64")
    train_groups = {k: np.sort(v.dropna().to_numpy()) for k, v in train_values.groupby(_keys(train_key))}
    for key, idx in test_values.groupby(_keys(test_key)).groups.items():
        sorted_values = train_groups.get(key)
        if sorted_values is None or len(sorted_values) == 0:
            continue
        values = test_values.loc[idx].to_numpy(dtype="float64")
        n_less = np.searchsorted(sorted_values, values, side="left")
        n_equal = np.searchsorted(sorted_values, values, side="right") - n_less
        ranks = (n_less + (n_equal + 1) / 2) / len(sorted_values)
        ranks[np.isnan(values)] = np.nan
        result.loc[idx] = ranks
    return result


def _unary(train, test, operator):
    x_train, x_test = _to_numeric(train, test)
    if operator in ("min", "max"):
        value = np.nanmin(x_train) if operator == "min" else np.nanmax(x_train)
        f = lambda x: pd.Series(value, index=x.index, dtype="float64")
    elif operator == "freq":
        counts = x_train.value_counts()
        f = lambda x: x.map(counts).fillna(0)  # missing values get count 0 (like the first implementation)
    elif operator == "abs":
        f = lambda x: x.abs()
    elif operator == "log":
        f = lambda x: np.log(np.abs(x))
    elif operator == "sqrt":
        f = lambda x: np.sqrt(np.abs(x))
    elif operator == "square":
        f = lambda x: np.square(x)
    elif operator == "sigmoid":
        f = lambda x: 1 / (1 + np.exp(-x))
    elif operator == "round":
        f = lambda x: np.floor(x)
    elif operator == "residual":
        f = lambda x: x - np.floor(x)
    else:
        raise NotImplementedError(f"Unknown operator {operator}")
    with np.errstate(all="ignore"):
        return f(x_train), (None if test is None else f(x_test))


def _binary(train1, train2, test1, test2, operator):
    has_test = test1 is not None
    if operator in ("add", "subtract", "multiply", "divide"):
        a_train, a_test = _to_numeric(train1, test1)
        b_train, b_test = _to_numeric(train2, test2)
        f = {"add": lambda a, b: a + b,
             "subtract": lambda a, b: a - b,
             "multiply": lambda a, b: a * b,
             # division by 0 keeps the numerator (like the first implementation)
             "divide": lambda a, b: (a / b.where(b != 0)).where(b != 0, a)}[operator]
        with np.errstate(all="ignore"):
            return f(a_train, b_train), (f(a_test, b_test) if has_test else None)

    if operator in ("GroupByThenMin", "GroupByThenMax"):
        # On the raw values (also non-numeric ones); ordered categories are compared in category order
        if isinstance(train1.dtype, pd.CategoricalDtype) and train1.cat.ordered:
            codes = train1.cat.codes.where(train1.cat.codes >= 0).groupby(_keys(train2))
            codes = codes.min() if operator == "GroupByThenMin" else codes.max()
            categories = train1.cat.categories
            stat = codes.map(lambda c: np.nan if pd.isna(c) else categories[int(c)])
        else:
            values = train1.astype(object) if isinstance(train1.dtype, pd.CategoricalDtype) else train1
            grouped = values.groupby(_keys(train2))
            stat = grouped.min() if operator == "GroupByThenMin" else grouped.max()
        return _map_group_statistic(stat, train2, test2)
    if operator in ("GroupByThenMean", "GroupByThenMedian", "GroupByThenStd"):
        grouped = pd.to_numeric(train1, errors="coerce").groupby(_keys(train2))
        stat = {"GroupByThenMean": grouped.mean, "GroupByThenMedian": grouped.median,
                "GroupByThenStd": grouped.std}[operator]()
        return _map_group_statistic(stat, train2, test2)
    if operator == "GroupByThenNUnique":
        return _map_group_statistic(train1.groupby(_keys(train2)).nunique(), train2, test2)
    if operator == "GroupByThenRank":
        values_train = pd.to_numeric(train1, errors="coerce")
        f_train = values_train.groupby(_keys(train2)).rank(ascending=True, pct=True)
        f_test = None
        if has_test:
            f_test = _rank_against_train(values_train, train2, pd.to_numeric(test1, errors="coerce"), test2)
        return f_train, f_test

    if operator in ("Combine", "CombineThenFreq"):
        def combine(a, b):
            return (a.astype(str) + "_" + b.astype(str)).where(a.notna() & b.notna())

        c_train = combine(train1, train2)
        c_test = combine(test1, test2) if has_test else None
        if operator == "Combine":
            return _categorical_codes(c_train, c_test)
        counts = c_train.value_counts()
        f = lambda c: c.map(counts).fillna(0).where(c.notna())
        return f(c_train), (f(c_test) if has_test else None)

    raise NotImplementedError(f"Unknown operator {operator}")


def fit_apply_operator(operator, train1, train2=None, test1=None, test2=None):
    """Feature for the training rows (and the test rows, if given), fitted on the training rows only.

    Returns (train_feature, test_feature) with the index of train1 / test1; test_feature is None without test rows.
    """
    operator = canonical_operator(operator)
    if train2 is None:
        return _unary(train1, test1, operator)
    return _binary(train1, train2, test1, test2, operator)


def compute_feature(name, X_train, X_test=None):
    """(train_feature, test_feature) of a (non-delete) feature name whose operands are columns of X_train."""
    operator, operands = parse_featurename(name)
    train2 = X_train[operands[1]] if len(operands) == 2 else None
    test1 = None if X_test is None else X_test[operands[0]]
    test2 = None if X_test is None or len(operands) == 1 else X_test[operands[1]]
    return fit_apply_operator(operator, X_train[operands[0]], train2, test1, test2)


def apply_features(X_train, X_test, names):
    """Add (or, for "without - <col>", remove) the named features to copies of X_train and X_test, in order."""
    X_train, X_test = X_train.copy(), (None if X_test is None else X_test.copy())
    for name in names:
        operator, operands = parse_featurename(name)
        if operator == DELETE:
            X_train = X_train.drop(columns=operands[0])
            X_test = None if X_test is None else X_test.drop(columns=operands[0])
            continue
        f_train, f_test = compute_feature(name, X_train, X_test)
        X_train[name] = f_train.to_numpy()
        if X_test is not None:
            X_test[name] = f_test.to_numpy()
    return X_train, X_test


def same_feature(a, b, rtol=0.0):
    """True if two feature columns are identical (NaN == NaN); rtol > 0 allows floating point differences."""
    if a is None or b is None:
        return a is None and b is None
    a, b = pd.Series(list(a)), pd.Series(list(b))
    if len(a) != len(b):
        return False
    try:
        a_num = pd.to_numeric(a, errors="raise").to_numpy(dtype="float64")
        b_num = pd.to_numeric(b, errors="raise").to_numpy(dtype="float64")
    except (ValueError, TypeError):
        return bool(a.astype(str).where(a.notna(), "<NA>").equals(b.astype(str).where(b.notna(), "<NA>")))
    if rtol > 0:
        return bool(np.allclose(a_num, b_num, rtol=rtol, atol=0.0, equal_nan=True))
    return bool(np.array_equal(a_num, b_num, equal_nan=True))
