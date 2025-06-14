import pandas as pd
from pymfe.mfe import MFE
import numpy as np


def get_pymfe_metafeatures(feature):
    pymfe = MFE()
    pymfe.fit(np.array(feature))
    metafeatures = pymfe.extract()
    return metafeatures


def get_numeric_pandas_metafeatures(feature_df, featurename):
    if feature_df.empty or not pd.api.types.is_numeric_dtype(feature_df[featurename]):
        return {
            "feature - name": featurename,
            "feature - count": 0,
            "feature - mean": np.nan,
            "feature - std": np.nan,
            "feature - min": np.nan,
            "feature - max": np.nan,
            "feature - lower percentile": np.nan,
            "feature - 50 percentile": np.nan,
            "feature - upper percentile": np.nan,
        }
    feature_pandas_description = feature_df.describe(include=np.number)
    feature_metadata_numeric = {
        "feature - name": featurename,
        "feature - count": feature_pandas_description.iloc[0].values[0],
        "feature - mean": feature_pandas_description.iloc[1].values[0],
        "feature - std": feature_pandas_description.iloc[2].values[0],
        "feature - min": feature_pandas_description.iloc[3].values[0],
        "feature - max": feature_pandas_description.iloc[4].values[0],
        "feature - lower percentile": feature_pandas_description.iloc[5].values[0],
        "feature - 50 percentile": feature_pandas_description.iloc[6].values[0],
        "feature - upper percentile": feature_pandas_description.iloc[7].values[0]
    }
    return feature_metadata_numeric


def get_categorical_pandas_metafeatures(feature_df, featurename):
    if feature_df.empty or not pd.api.types.is_categorical_dtype(feature_df[featurename]):
        return {
            "feature - name": featurename,
            "feature - count": 0,
            "feature - mean": np.nan,
            "feature - std": np.nan,
            "feature - min": np.nan,
            "feature - max": np.nan,
            "feature - lower percentile": np.nan,
            "feature - 50 percentile": np.nan,
            "feature - upper percentile": np.nan,
        }
    feature_pandas_description = feature_df.describe(exclude=np.number)
    feature_metadata_categorical = {
        "feature - name": featurename,
        "feature - count": feature_pandas_description.iloc[0].values[0],
        "feature - unique": feature_pandas_description.iloc[1].values[0],
        "feature - top": feature_pandas_description.iloc[2].values[0],
        "feature - freq": feature_pandas_description.iloc[3].values[0]
    }
    return feature_metadata_categorical


def get_mfe_metadata(X, y, feature):
    # mfe = MFE(groups=["general", "statistical", "info-theory", "model-based", "landmarking"])
    mfe = MFE(groups="all")
    # mfe.fit(np.array(feature))
    mfe.fit(X, y)
    metafeatures = mfe.extract()
    return metafeatures



