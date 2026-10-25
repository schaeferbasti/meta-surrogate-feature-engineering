#!/bin/bash
# Timing pilot of the multimodel experiment: one job per model and pilot dataset (metafe.config.PILOT_DATASETS), each
# fitting the original data and N_CANDIDATES candidates like the core matrix. Afterwards (on the login node):
#   python -m metafe.pilot summary --experiment multimodel
# Usage: DRY_RUN=1 scripts/submit_pilot.sh; options: N_CANDIDATES (default 10), PILOT_TIME (default 12:00:00), MODELS,
# DATASETS, SBATCH_OPTS, CPU_OPTS, GPU_OPTS (default "--gres=gpu:1"), MODULES, VENV (default ~/venvs/metafe-ag16).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
EXPERIMENT=multimodel
VENV=${VENV:-$HOME/venvs/metafe-ag16}
py() { PYTHONPATH="$REPO" "$VENV/bin/python" -c "from metafe.config import *; from metafe.models import GPU_MODELS; $1"; }
DATASETS=${DATASETS:-$(py 'print(*PILOT_DATASETS)')}
MODELS=${MODELS:-$(py "print(*EXPERIMENTS['$EXPERIMENT'].models)")}
GPU_MODEL_KEYS=$(py 'print(*sorted(GPU_MODELS))')
N_CANDIDATES=${N_CANDIDATES:-10}
PILOT_TIME=${PILOT_TIME:-12:00:00}
CPU_OPTS=${CPU_OPTS:-}
GPU_OPTS=${GPU_OPTS:---gres=gpu:1}
n=$(wc -w <<< "$DATASETS")
export REPO VENV DATASETS MODULES="${MODULES:-}"
mkdir -p logs

for model in $MODELS; do
    if [[ " $GPU_MODEL_KEYS " == *" $model "* ]]; then device=$GPU_OPTS; else device=$CPU_OPTS; fi
    cmd=(sbatch --parsable ${SBATCH_OPTS:-} --export=ALL --output="logs/%x.%A_%a.out" -J "pilot_$model"
         --array=0-$((n - 1)) -c 8 --mem=64G -t "$PILOT_TIME" $device
         scripts/run_step.sh metafe.pilot run --experiment "$EXPERIMENT" --dataset {dataset} --model "$model"
         --n_candidates "$N_CANDIDATES")
    if [ -n "${DRY_RUN:-}" ]; then echo "${cmd[*]}"; else echo "pilot $model: $("${cmd[@]}")"; fi
done
