#!/bin/bash
# MetaFE with LightGBM only (experiment "lightgbm", AutoGluon 1.2, CPU): completes the experiments with OpenFE and
# TabPrep, whose LightGBM results are cached in results/lightgbm/evaluation, so only the MetaFE data is evaluated.
# Run this first, to see whether MetaFE works, before the multimodel experiment.
#   DRY_RUN=1 scripts/submit_lightgbm.sh     # print the sbatch commands
#   scripts/submit_lightgbm.sh               # labels -> metafeatures -> MetaFE -> check -> evaluation -> plots
# All options of scripts/submit_slurm.sh apply (FROM, UNTIL, DATASETS, SBATCH_OPTS, CPU_OPTS, ...).
export EXPERIMENT=lightgbm
export VENV=${VENV:-$HOME/venvs/metafe}
exec "$(dirname "$0")/submit_slurm.sh"
