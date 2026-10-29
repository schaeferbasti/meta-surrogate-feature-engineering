#!/bin/bash
# Experiment "lightgbm_cv": knowledge base with the repeated LightGBM CV label (metafe/config.py), all candidates.
# One label job per dataset with CORE_CPUS parallel candidates (about 450-600 core-hours in total).
#   DRY_RUN=1 scripts/submit_lightgbm_cv.sh                   # print the sbatch commands
#   UNTIL=mf_merge scripts/submit_lightgbm_cv.sh              # knowledge base only: labels + metafeatures
# All options of scripts/submit_slurm.sh apply (FROM, UNTIL, DATASETS, SBATCH_OPTS, CPU_OPTS, OPENML_CACHE, ...).
export EXPERIMENT=lightgbm_cv
export VENV=${VENV:-$HOME/venvs/metafe}
export CORE_CPUS=${CORE_CPUS:-32}
export CORE_MEM=${CORE_MEM:-96G}
export CORE_TIME=${CORE_TIME:-1-00:00:00}
exec "$(dirname "$0")/submit_slurm.sh"
