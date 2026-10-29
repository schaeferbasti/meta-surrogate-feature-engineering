#!/bin/bash
# Runs label_reliability.py <args> in the analysis venv; "lgbcv-all" runs the LightGBM-CV arm on every dataset
set -euo pipefail
cd /work/dlclarge1/matusd-schaefeb-metafe/meta-surrogate-feature-engineering
PY=/work/dlclarge1/matusd-schaefeb-metafe/venvs/analysis/bin/python
export PYTHONPATH=.
echo "$(date) $(hostname) $*"
if [[ "$1" == "lgbcv-all" ]]; then
  for d in $($PY -c "import sys; sys.path.insert(0, 'experiments/label_reliability'); from label_reliability import DATASETS; print(*DATASETS)"); do
    $PY experiments/label_reliability/label_reliability.py lgbcv --dataset "$d" --workers "${SLURM_CPUS_PER_TASK:-8}"
  done
else
  $PY experiments/label_reliability/label_reliability.py "$@"
fi
echo "$(date) done"
