import numpy as np
from sklearn import preprocessing
import pandas as pd


def factorize_dataset(X_train, y_train, X_test, y_test):
    lbl = preprocessing.LabelEncoder()
    for column in X_train.columns:  # select_dtypes(include=['object', 'category'])
        X_train[column] = lbl.fit_transform(X_train[column].astype(int))
    for column in X_test.columns:
        X_test[column] = lbl.fit_transform(X_test[column].astype(int))
    y_train_array, _ = pd.Series.factorize(y_train, use_na_sentinel=False)
    y_train = y_train.replace(y_train_array)
    y_test_array, _ = pd.Series.factorize(y_test, use_na_sentinel=False)
    y_test = y_test.replace(y_test_array)
    return X_train, y_train, X_test, y_test


def factorize_transformed_dataset(X_train, y_train, X_test, y_test):
    lbl = preprocessing.LabelEncoder()
    X_train = X_train.replace([np.inf, -np.inf], np.nan)  # Convert inf values to NaN
    X_train = X_train.dropna(axis=1, how="any")
    for column in X_train.columns:  # select_dtypes(include=['object', 'category'])
        X_train[column] = lbl.fit_transform(X_train[column].astype(int))
    X_test = X_test.replace([np.inf, -np.inf], np.nan)  # Convert inf values to NaN
    X_test = X_test.dropna(axis=1, how="any")
    for column in X_test.columns:
        X_test[column] = lbl.fit_transform(X_test[column].astype(int))
    y_train_array, _ = pd.Series.factorize(y_train, use_na_sentinel=False)
    y_train = y_train.replace(y_train_array)
    y_test_array, _ = pd.Series.factorize(y_test, use_na_sentinel=False)
    y_test = y_test.replace(y_test_array)
    return X_train, y_train, X_test, y_test


def factorize_data(X_train):
    lbl = preprocessing.LabelEncoder()
    for column in X_train.columns:
        X_train[column] = lbl.fit_transform(X_train[column].astype(int))
    return X_train


def factorize_data_split(X_train, y_train, X_test, y_test):
    lbl = preprocessing.LabelEncoder()
    for column in X_train.columns:  #select_dtypes(include=['object', 'category'])
        # X_train[column], _ = pd.factorize(X_train[column])
        X_train[column] = lbl.fit_transform(X_train[column].astype(int))
    for column in X_test.columns:
        X_test[column] = lbl.fit_transform(X_test[column].astype(int))
    y_train_array, _ = pd.Series.factorize(y_train, use_na_sentinel=False)
    y_train = y_train.replace(y_train_array)
    y_test_array, _ = pd.Series.factorize(y_test, use_na_sentinel=False)
    y_test = y_test.replace(y_test_array)
    return X_train, y_train, X_test, y_test


def factorize_data_old(X_train, y_train, X_test, y_test):
    # Identify categorical columns
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns

    # Apply LabelEncoder only to categorical columns
    for column in categorical_columns:
        lbl = preprocessing.LabelEncoder()
        X_train[column] = lbl.fit_transform(X_train[column].astype(str))  # Convert to string before encoding
        X_test[column] = lbl.transform(X_test[column].astype(str))  # Apply the same mapping to test data

    # Factorize target labels for consistency
    y_train, label_mapping = pd.factorize(y_train, use_na_sentinel=False)
    y_test = pd.Series(y_test).map(dict(enumerate(label_mapping))).fillna(0).astype(
        int)  # .interpolate(method="pad").astype(int)  # Ensure mapping consistency

    return X_train, y_train, X_test, y_test
