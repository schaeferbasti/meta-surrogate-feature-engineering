import pandas as pd
import numpy as np

from src.utils.create_feature_and_featurename import create_featurenames


def get_matrix_columns():
    return ['dataset - id', 'dataset - task type', 'dataset - number of classes', 'feature - name', 'operator', 'feature - type', 'feature - count', 'feature - mean', 'feature - std', 'feature - min', 'feature - max', 'feature - lower percentile', 'feature - 50 percentile', 'feature - upper percentile', 'feature - unique', 'feature - top', 'feature - freq', 'model', 'improvement']


def add_new_featurenames(X_test):
    # Get new dataset with feature names and metafeatures and replicate each feature (=each row) x times, that we can repeat thus row with similar values, but instead of the feature name, we add a new name consisting of all available operators and respective features
    matrix_columns = get_matrix_columns()
    featurenames = create_featurenames(X_test["feature - name"].values)
    X_test_new = pd.DataFrame({
        matrix_columns[0]: np.repeat((data := X_test[matrix_columns[0]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[1]: np.repeat((data := X_test[matrix_columns[1]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[2]: np.repeat((data := X_test[matrix_columns[2]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[3]: np.repeat(np.array(featurenames), data.shape[0]).reshape(len(featurenames), data.shape[0]).T.flatten(),
        matrix_columns[4]: np.repeat((data := X_test[matrix_columns[4]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[5]: np.repeat((data := X_test[matrix_columns[5]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[6]: np.repeat((data := X_test[matrix_columns[6]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[7]: np.repeat((data := X_test[matrix_columns[7]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[8]: np.repeat((data := X_test[matrix_columns[8]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[9]: np.repeat((data := X_test[matrix_columns[9]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[10]: np.repeat((data := X_test[matrix_columns[10]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[11]: np.repeat((data := X_test[matrix_columns[11]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[12]: np.repeat((data := X_test[matrix_columns[12]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[13]: np.repeat((data := X_test[matrix_columns[13]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[14]: np.repeat((data := X_test[matrix_columns[14]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[15]: np.repeat((data := X_test[matrix_columns[15]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[16]: np.repeat((data := X_test[matrix_columns[16]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[17]: np.repeat((data := X_test[matrix_columns[17]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
    })
    return X_test_new
