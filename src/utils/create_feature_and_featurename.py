import numpy as np
import pandas as pd

from src.utils.get_operators import get_operators


def create_feature_and_featurename(feature1, feature2, operator):
    if feature2 is None:
        feature, featurename = create_unary_feature_and_featurename(feature1, operator)
    else:
        feature, featurename = create_binary_feature_and_featurename(feature1, feature2, operator)
    return feature, featurename


def create_unary_feature_and_featurename(feature1, operator):
    feature1_int_list = [int(x) for x in feature1]
    if operator == "min":
        feature = feature1.apply(lambda x: min(feature1_int_list))
        featurename = "min(" + str(feature1.name) + ")"
    elif operator == "max":
        feature = feature1.apply(lambda x: max(feature1_int_list))
        featurename = "max(" + str(feature1.name) + ")"
    elif operator == "freq":
        feature = feature1.apply(lambda x: feature1_int_list.count(int(x)))
        featurename = "freq(" + str(feature1.name) + ")"
    elif operator == "abs":
        feature = feature1.apply(lambda x: abs(float(x)))
        featurename = "abs(" + str(feature1.name) + ")"
    elif operator == "log":
        feature = feature1.apply(lambda x: np.log(np.abs(float(x))))
        featurename = "log(" + str(feature1.name) + ")"
    elif operator == "sqrt":
        feature = feature1.apply(lambda x: np.sqrt(np.abs(float(x))))
        featurename = "sqrt(" + str(feature1.name) + ")"
    elif operator == "square":
        feature = feature1.apply(lambda x: np.square(float(x)))
        featurename = "square(" + str(feature1.name) + ")"
    elif operator == "sigmoid":
        feature = feature1.apply(lambda x: 1 / (1 + np.exp(-float(x))))
        featurename = "sigmoid(" + str(feature1.name) + ")"
    elif operator == "round":
        feature = feature1.apply(lambda x: np.floor(float(x)))
        featurename = "round(" + str(feature1.name) + ")"
    elif operator == "residual":
        feature = feature1.apply(lambda x: float(x) - np.floor(float(x)))
        featurename = "residual(" + str(feature1.name) + ")"
    else:
        raise NotImplementedError(f"Unrecognized operator {operator}.")
    return feature, featurename


def create_binary_feature_and_featurename(feature1, feature2, operator):
    feature1_int_list = [int(x) for x in feature1]
    feature2_int_list = [int(x) for x in feature2]
    if operator == "+":
        feature = [f1 + f2 for f1, f2 in zip(feature1_int_list, feature2_int_list)]
        featurename = "add(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "-":
        feature = [f1 - f2 for f1, f2 in zip(feature1_int_list, feature2_int_list)]
        featurename = "subtract(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "*":
        feature = [f1 * f2 for f1, f2 in zip(feature1_int_list, feature2_int_list)]
        featurename = "multiply(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "/":
        feature = [f1 / f2 if f2 != 0 else f1 for f1, f2 in zip(feature1_int_list, feature2_int_list)]
        featurename = "divide(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "GroupByThenMin":
        temp = feature1.groupby(feature2).min()
        temp.loc[np.nan] = np.nan
        feature = feature2.apply(lambda x: temp.loc[x])
        featurename = "GroupByThenMin(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "GroupByThenMax":
        temp = feature1.groupby(feature2).max()
        temp.loc[np.nan] = np.nan
        feature = feature2.apply(lambda x: temp.loc[x])
        featurename = "GroupByThenMax(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "GroupByThenMean":
        feature1 = pd.to_numeric(feature1, errors='coerce')
        temp = feature1.groupby(feature2)
        temp = temp.mean()
        temp.loc[np.nan] = np.nan
        feature = feature2.apply(lambda x: temp.loc[x])
        featurename = "GroupByThenMean(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "GroupByThenMedian":
        feature1 = pd.to_numeric(feature1, errors='coerce')
        temp = feature1.groupby(feature2).median()
        temp.loc[np.nan] = np.nan
        feature = feature2.apply(lambda x: temp.loc[x])
        featurename = "GroupByThenMedian(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "GroupByThenStd":
        feature1 = pd.to_numeric(feature1, errors='coerce')
        temp = feature1.groupby(feature2).std()
        temp.loc[np.nan] = np.nan
        feature = feature2.apply(lambda x: temp.loc[x])
        featurename = "GroupByThenStd(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == 'GroupByThenRank':
        feature1 = pd.to_numeric(feature1, errors='coerce')
        feature = feature1.groupby(feature2).rank(ascending=True, pct=True)
        featurename = "GroupByThenRank(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "GroupByThenFreq":
        def _f(x):
            value_counts = x.value_counts()
            value_counts.loc[np.nan] = np.nan
            return x.apply(lambda x: value_counts.loc[x])

        feature = feature1.groupby(feature2).apply(_f)
        featurename = "GroupByThenFreq(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "GroupByThenNUnique":
        nunique = feature1.groupby(feature2).nunique()
        nunique.loc[np.nan] = np.nan
        feature = feature2.apply(lambda x: nunique.loc[x])
        featurename = "GroupByThenNUnique(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "Combine":
        temp = feature1.astype(str) + '_' + feature2.astype(str)
        temp[feature1.isna() | feature2.isna()] = np.nan
        temp, _ = temp.factorize()
        feature = pd.Series(temp, index=feature1.index).astype("float64")
        featurename = "Combine(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "CombineThenFreq":
        temp = feature1.astype(str) + '_' + feature2.astype(str)
        temp[feature1.isna() | feature2.isna()] = np.nan
        value_counts = temp.value_counts()
        value_counts.loc[np.nan] = np.nan
        feature = temp.apply(lambda x: value_counts.loc[x])
        featurename = "CombineThenFreq(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    else:
        raise NotImplementedError(f"Unrecognized operator {operator}.")
    return feature, featurename


def create_featurenames(feature_list):
    unary_operators, binary_operators = get_operators()
    featurenames = []
    for feature in feature_list:
        for operator in unary_operators:
            featurenames.append(operator + "(" + str(feature) + ")")
    for feature1 in feature_list:
        for operator in binary_operators:
            for feature2 in feature_list:
                featurenames.append(operator + "(" + str(feature1) + ", " + str(feature2) + ")")
    return featurenames
