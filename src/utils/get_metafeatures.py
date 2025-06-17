import pandas as pd
from pymfe.mfe import MFE
import numpy as np


def get_pymfe_metafeatures(feature):
    pymfe = MFE()
    pymfe.fit(np.array(feature))
    metafeatures = pymfe.extract()
    return metafeatures


def get_pandas_metafeatures(feature_df, featurename):
    feature_pandas_description = feature_df.describe(include="all")
    feature_pandas_description = check_and_complete_pandas_description(feature_pandas_description)
    feature_metadata = {
        "feature - count": feature_pandas_description.loc["count"].values[0],
        "feature - unique": feature_pandas_description.loc["unique"].values[0],
        "feature - top": feature_pandas_description.loc["top"].values[0],
        "feature - freq": feature_pandas_description.loc["freq"].values[0],
        "feature - mean": feature_pandas_description.loc["mean"].values[0],
        "feature - std": feature_pandas_description.loc["std"].values[0],
        "feature - min": feature_pandas_description.loc["min"].values[0],
        "feature - 25": feature_pandas_description.loc["25%"].values[0],
        "feature - 50": feature_pandas_description.loc["50%"].values[0],
        "feature - 75": feature_pandas_description.loc["75%"].values[0],
        "feature - max": feature_pandas_description.loc["max"].values[0],
    }
    return feature_metadata


def check_and_complete_pandas_description(feature_pandas_description):
    if "count" not in feature_pandas_description.index:
        feature_pandas_description.loc["count"] = np.NaN
    if "unique" not in feature_pandas_description.index:
        feature_pandas_description.loc["unique"] = np.NaN
    if "top" not in feature_pandas_description.index:
        feature_pandas_description.loc["top"] = np.NaN
    if "freq" not in feature_pandas_description.index:
        feature_pandas_description.loc["freq"] = np.NaN
    if "mean" not in feature_pandas_description.index:
        feature_pandas_description.loc["mean"] = np.NaN
    if "std" not in feature_pandas_description.index:
        feature_pandas_description.loc["std"] = np.NaN
    if "min" not in feature_pandas_description.index:
        feature_pandas_description.loc["min"] = np.NaN
    if "25%" not in feature_pandas_description.index:
        feature_pandas_description.loc["25%"] = np.NaN
    if "50%" not in feature_pandas_description.index:
        feature_pandas_description.loc["50%"] = np.NaN
    if "75%" not in feature_pandas_description.index:
        feature_pandas_description.loc["75%"] = np.NaN
    if "max" not in feature_pandas_description.index:
        feature_pandas_description.loc["max"] = np.NaN
    return feature_pandas_description


def get_mfe_feature_metadata(feature):
    # mfe = MFE(groups=["general", "statistical", "info-theory", "model-based", "landmarking"])
    mfe = MFE(groups="all")
    mfe.fit(feature)
    metafeatures = mfe.extract()
    columns = mfe.extract_metafeature_names()
    groups = mfe.parse_by_group(["general", "statistical", "model-based", "info-theory", "landmarking", "complexity", "clustering"], metafeatures)
    return metafeatures, columns, groups


def get_mfe_dataset_metadata(X, y, group):
    # mfe = MFE(groups=["general", "statistical", "info-theory", "model-based", "landmarking"])
    mfe = MFE(groups=group)
    mfe.fit(X, y)
    metafeatures = mfe.extract()
    columns = mfe.extract_metafeature_names()
    group = mfe.parse_by_group(group, metafeatures)
    return metafeatures, columns, group



