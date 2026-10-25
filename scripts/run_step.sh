#!/bin/bash
# Run one MetaFE pipeline step: scripts/run_step.sh <python module> [args...]
# In the arguments, {dataset} and {fold} are replaced by the dataset / fold of this array task:
#   SLURM_ARRAY_TASK_ID = i -> dataset = DATASETS[i]                         (default)
#                              dataset = DATASETS[i / 3], fold = i % 3      (PER_FOLD=1, 3 folds)
# Environment: REPO (default: the submit directory), VENV (default ~/venvs/metafe; ~/venvs/metafe-ag16 for the
# multimodel experiment), DATASETS (space-separated task IDs, default: all of metafe.config.DATASETS), MODULES
# (optional "module load" list, e.g. a CUDA module).
# Usually submitted by scripts/submit_slurm.sh, but it also runs directly, e.g.
#   scripts/run_step.sh metafe.metafe --experiment lightgbm --dataset 359955
set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
VENV=${VENV:-$HOME/venvs/metafe}
cd "$REPO"
if [ -n "${MODULES:-}" ]; then
    [ -f /nfs/opt/env/env.sh ] && . /nfs/opt/env/env.sh  # Lmod on calculco
    module load $MODULES
fi
source "$VENV/bin/activate"
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

dataset=""
fold=""
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    read -r -a ids <<< "${DATASETS:-$(python -c 'from metafe.config import DATASETS; print(*sorted(DATASETS))')}"
    if [ "${PER_FOLD:-0}" = 1 ]; then
        dataset=${ids[$((SLURM_ARRAY_TASK_ID / 3))]}
        fold=$((SLURM_ARRAY_TASK_ID % 3))
    else
        dataset=${ids[$SLURM_ARRAY_TASK_ID]}
    fi
fi
args=()
for arg in "$@"; do
    arg=${arg//\{dataset\}/$dataset}
    args+=("${arg//\{fold\}/$fold}")
done

echo "$(date) $(hostname) ($(lscpu | sed -n 's/^Model name: *//p')), ${SLURM_CPUS_PER_TASK:-?} CPUs: python -m ${args[*]}"
start=$(date +%s)
status=0
python -m "${args[@]}" || status=$?
echo "$(date) exit $status after $(( $(date +%s) - start )) s"
exit $status
