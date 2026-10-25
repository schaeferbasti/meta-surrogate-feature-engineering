#!/bin/bash
# MetaFE with Linear model, random forest, LightGBM, TabICLv2 and RealMLP (experiment "multimodel", AutoGluon 1.6;
# TabICL and RealMLP on GPU). Run scripts/submit_pilot.sh first and set max_candidates / max_label_rows in
# metafe/config.py from `python -m metafe.pilot summary --experiment multimodel`.
#   DRY_RUN=1 scripts/submit_multimodel.sh
#   GPU_OPTS="--partition=gpu --gres=gpu:1" CPU_OPTS="--partition=cpu" scripts/submit_multimodel.sh
# All options of scripts/submit_slurm.sh apply (FROM, UNTIL, DATASETS, MODELS, SBATCH_OPTS, MODULES, ...).
export EXPERIMENT=multimodel
export VENV=${VENV:-$HOME/venvs/metafe-ag16}
exec "$(dirname "$0")/submit_slurm.sh"
