#!/bin/bash
# Label reliability test on SLURM: AutoGluon arm (datasets x seeds x chunks, 8 CPUs each), LightGBM-CV arm (one job),
# then the analysis. DRY_RUN=1 prints the commands.
set -euo pipefail
cd /work/dlclarge1/matusd-schaefeb-metafe/meta-surrogate-feature-engineering
PART=${PART:-mldlc2_cpu-epyc9655}
N_CHUNKS=${N_CHUNKS:-5}
LOG=logs/label_reliability; mkdir -p $LOG
N_DATASETS=$(/work/dlclarge1/matusd-schaefeb-metafe/venvs/analysis/bin/python -c \
  "import sys; sys.path.insert(0, 'experiments/label_reliability'); from label_reliability import DATASETS; print(len(DATASETS))")
N_TASKS=$((N_DATASETS * 3 * N_CHUNKS))
run() { if [[ -n "${DRY_RUN:-}" ]]; then echo "$@" >&2; echo 0; else "$@"; fi; }
AG=$(run sbatch --parsable -p $PART -c 8 --mem=48G -t 12:00:00 --array=0-$((N_TASKS - 1))%80 -J lr_ag \
  -o $LOG/ag_%A_%a.log --export=ALL,N_CHUNKS=$N_CHUNKS experiments/label_reliability/run_ag.sh)
LG=$(run sbatch --parsable -p $PART -c 32 --mem=120G -t 12:00:00 -J lr_lgbcv -o $LOG/lgbcv_%j.log \
  experiments/label_reliability/run_py.sh lgbcv-all)
AN=$(run sbatch --parsable -p $PART -c 2 --mem=16G -t 1:00:00 -J lr_analyze -o $LOG/analyze_%j.log \
  --dependency=afterany:$AG:$LG experiments/label_reliability/run_py.sh analyze)
echo "ag $AG ($N_TASKS tasks), lgbcv $LG, analyze $AN"
