#!/bin/bash
# Run one command of the pipeline inside an OAR job (calculco): scripts/oar_step.sh <command> [args...]
# e.g. scripts/oar_step.sh python -m metafe.metafe --experiment lightgbm_reuse --dataset 359955
# Environment: VENV (default ~/venvs/metafe). Usually submitted by scripts/submit_oar.sh.
# No modules are needed (the venv brings everything); /nfs/opt/env/env.sh is not sourced (it fails with set -u).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
VENV=${VENV:-$HOME/venvs/metafe}
cd "$REPO"
source "$VENV/bin/activate"
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

echo "$(date) $(hostname) ($(lscpu | sed -n 's/^Model name: *//p')), job ${OAR_JOB_ID:-?}: $*"
start=$(date +%s)
status=0
"$@" || status=$?
echo "$(date) exit $status after $(( $(date +%s) - start )) s"
exit $status
