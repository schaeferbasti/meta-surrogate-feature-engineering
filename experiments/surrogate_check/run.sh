#!/bin/bash
# SLURM wrapper of the surrogate check: run.sh <script.py> [args...]; SLURM_ARRAY_TASK_ID -> --dataset (DATASETS order)
set -euo pipefail
cd /work/dlclarge1/matusd-schaefeb-metafe/meta-surrogate-feature-engineering
export PYTHONPATH=. OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
PY=/work/dlclarge1/matusd-schaefeb-metafe/venvs/analysis/bin/python
SCRIPT=$1; shift
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  ID=$($PY -c "from metafe.config import DATASETS; print(list(DATASETS)[$SLURM_ARRAY_TASK_ID])")
  set -- "$@" --dataset "$ID"
fi
echo "$(date) $(hostname) $SCRIPT $*"
$PY experiments/surrogate_check/$SCRIPT "$@"
echo "$(date) done"
