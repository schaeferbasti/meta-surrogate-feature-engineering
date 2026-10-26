"""Step 3: surrogate model = CatBoost regressor that predicts the label (relative improvement) of a candidate feature
from the metafeature matrix. It is trained on all benchmark datasets except the one it is applied to."""
import os
import tempfile

import numpy as np
import pandas as pd

from metafe.metafeatures import add_metafeatures
from metafe.operators import candidate_featurenames, parse_featurename

LABEL = "improvement"


class Surrogate:
    def __init__(self, matrix, exclude_dataset=None):
        """Train on the metafeature matrix (without the rows of exclude_dataset: leave-one-dataset-out)."""
        from autogluon.tabular.models import CatBoostModel

        if exclude_dataset is not None:
            matrix = matrix[matrix["dataset - id"] != exclude_dataset]
        self.columns = [c for c in matrix.columns if c != LABEL]
        self.operators = set(matrix["operator"])  # operators without a label cannot be predicted
        self.model = CatBoostModel(path=tempfile.mkdtemp() + os.sep)  # default: ./AutogluonModels
        self.model.fit(X=matrix[self.columns].reset_index(drop=True), y=matrix[LABEL].reset_index(drop=True))
        print(f"Surrogate trained on {len(matrix)} rows of {matrix['dataset - id'].nunique()} datasets, "
              f"operators: {sorted(self.operators)}")

    def candidates(self, X_train, dataset_id, task_type, model, exclude=()):
        """Metafeature rows of all candidate features of X_train, except features that are already columns, the
        names in exclude (e.g. features added or deleted before) and operators the surrogate has no label of."""
        exclude = set(exclude) | set(X_train.columns)
        names = [n for n in candidate_featurenames(X_train)
                 if n not in exclude and parse_featurename(n)[0] in self.operators]
        if X_train.shape[1] <= 1:  # never delete the last column
            names = [n for n in names if parse_featurename(n)[0] != "delete"]
        rows = pd.DataFrame({"dataset - id": dataset_id, "feature - name": names,
                             "operator": [parse_featurename(n)[0] for n in names], "model": model})
        return add_metafeatures(rows, X_train, task_type)

    def predict(self, candidates):
        """Predicted improvement of every candidate row."""
        return np.asarray(self.model.predict(X=candidates.reindex(columns=self.columns)))
