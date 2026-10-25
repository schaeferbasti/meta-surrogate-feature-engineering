#!/bin/bash
# Build the knowledge base of an experiment without a scheduler array: labels and metafeatures of all datasets,
# JOBS datasets in parallel, then the merges. Meant for experiment lightgbm_reuse (no model fits, about an hour);
# for experiments that compute labels, use the per-dataset jobs of scripts/submit_slurm.sh instead.
#   scripts/build_knowledge_base.sh <experiment>         (inside a job: scripts/oar_step.sh scripts/build_knowledge_base.sh ...)
# Environment: JOBS (default: number of cores), DATASETS (default: all), python of the active venv.
# Per-dataset logs: logs/kb_<experiment>_<id>.log. A failed dataset is reported; the others are still merged.
set -uo pipefail

EXPERIMENT=${1:?usage: build_knowledge_base.sh <experiment>}
JOBS=${JOBS:-$(nproc)}
DATASETS=${DATASETS:-$(python -c 'from metafe.config import DATASETS; print(*sorted(DATASETS))')}
mkdir -p logs

build_one() {
    python -m metafe.core_matrix build --experiment "$1" --dataset "$2" &&
    python -m metafe.metafeatures build --experiment "$1" --dataset "$2"
}
export -f build_one
printf '%s\n' $DATASETS | xargs -P "$JOBS" -I{} bash -c \
    'build_one '"$EXPERIMENT"' {} > logs/kb_'"$EXPERIMENT"'_{}.log 2>&1 && echo "dataset {} done" || echo "dataset {} FAILED (logs/kb_'"$EXPERIMENT"'_{}.log)"'

grep -h "^Dataset .* done in\|^Left out" logs/kb_"$EXPERIMENT"_*.log
python -m metafe.core_matrix merge --experiment "$EXPERIMENT" &&
python -m metafe.metafeatures merge --experiment "$EXPERIMENT"
