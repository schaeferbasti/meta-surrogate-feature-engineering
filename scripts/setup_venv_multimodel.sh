#!/bin/bash
# Create the venv of the multimodel experiment (AutoGluon 1.6.3, CUDA torch) and cache what the jobs download:
# the TabICLv2 checkpoints (Hugging Face) and the OpenML tasks. Run it on the login node (needs internet).
#   scripts/setup_venv_multimodel.sh
# Options: VENV (default ~/venvs/metafe-ag16), PYTHON (default python3.11),
#          TORCH_INDEX (default https://download.pytorch.org/whl/cu128; .../whl/cpu for a CPU-only test venv).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
VENV=${VENV:-$HOME/venvs/metafe-ag16}
PYTHON=${PYTHON:-python3.11}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}

"$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install "torch>=2.10,<2.14" --index-url "$TORCH_INDEX"
"$VENV/bin/pip" install -r "$REPO/requirements-multimodel.txt"

cd "$REPO"
PYTHONPATH="$REPO" "$VENV/bin/python" - <<'PY'
import numpy as np
import torch
from autogluon.tabular import __version__
from autogluon.tabular.models.tabicl.tabicl_model import TabICLModel
from tabicl import TabICLClassifier, TabICLRegressor

print(f"AutoGluon {__version__}, torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
# The checkpoints AutoGluon uses by default (TabICLv2); fitting once downloads them into ~/.cache/huggingface
X = np.random.rand(40, 3)
TabICLClassifier(checkpoint_version=TabICLModel.default_classification_model, device="cpu").fit(X, X[:, 0] > 0.5)
TabICLRegressor(checkpoint_version=TabICLModel.default_regression_model, device="cpu").fit(X, X[:, 1])
print("TabICLv2 checkpoints cached")

from metafe.config import DATASETS
from metafe.data import load_task
for task_id in DATASETS:
    load_task(task_id)
print(f"{len(DATASETS)} OpenML tasks cached")
PY
echo "Done: $VENV"
