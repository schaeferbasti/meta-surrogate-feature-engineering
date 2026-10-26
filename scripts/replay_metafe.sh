#!/bin/bash
# Rebuild the MetaFE files of all datasets from their step logs (metafe.metafe --replay), JOBS in parallel; for
# changes that only affect the test rows (MetaFE's choices depend only on the training rows).
#   scripts/replay_metafe.sh <experiment>     (inside a job: scripts/oar_step.sh scripts/replay_metafe.sh ...)
set -uo pipefail
EXPERIMENT=${1:?usage: replay_metafe.sh <experiment>}
JOBS=${JOBS:-$(nproc)}
DATASETS=${DATASETS:-$(python -c 'from metafe.config import DATASETS; print(*sorted(DATASETS))')}
mkdir -p logs
printf '%s\n' $DATASETS | xargs -P "$JOBS" -I{} bash -c \
    'python -m metafe.metafe --experiment '"$EXPERIMENT"' --dataset {} --replay > logs/replay_{}.log 2>&1 && echo "dataset {} done" || echo "dataset {} FAILED (logs/replay_{}.log)"'
! grep -q FAILED <(grep -L "steps replayed" logs/replay_*.log | sed 's/^/FAILED /')
