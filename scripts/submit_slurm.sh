#!/bin/bash
# Submit the MetaFE pipeline of one experiment (metafe/config.py) to SLURM. Usually called through
# scripts/submit_lightgbm.sh or scripts/submit_multimodel.sh, which set EXPERIMENT and VENV.
# Each stage starts when the previous one finished successfully (afterok):
#   core        metafe.core_matrix build      array over datasets, one job array per model   labels (resumable)
#   core_merge  metafe.core_matrix merge
#   mf          metafe.metafeatures build     array over datasets
#   mf_merge    metafe.metafeatures merge
#   metafe      metafe.metafe                 array over datasets, one job array per target model
#   check       metafe.evaluation.check_fe    (only reports; the evaluation does not wait for it)
#   evaluate    metafe.evaluation.evaluate    array over datasets x folds 0-2, one job array per model
#   merge       metafe.evaluation.merge_folds
#   plots       metafe.evaluation.plots
# Models in GPU_MODELS (TabICL, RealMLP) get GPU_OPTS in the core and evaluate stages.
#
# Usage (from the repository root):
#   DRY_RUN=1 scripts/submit_lightgbm.sh                # only print the sbatch commands
#   FROM=metafe scripts/submit_lightgbm.sh              # start at a stage (earlier stages are done)
#   UNTIL=mf_merge scripts/submit_lightgbm.sh           # stop after a stage
#   DATASETS="359955 168350" scripts/submit_lightgbm.sh # only some datasets (array stages)
#   MODELS="LR RF" scripts/submit_multimodel.sh         # only some models of the experiment
# Cluster options: SBATCH_OPTS (every job, e.g. "--account=xyz"), CPU_OPTS (CPU jobs, e.g. "--partition=cpu"),
# GPU_OPTS (GPU jobs, default "--gres=gpu:1"), MODULES (modules loaded in every job, e.g. a CUDA module).
# Time limits per stage: CORE_TIME, METAFE_TIME, EVAL_TIME (sbatch format); core stage resources: CORE_CPUS, CORE_MEM.
# OPENML_CACHE: OpenML cache directory of the jobs (metafe.data, METAFE_OPENML_CACHE).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
: "${EXPERIMENT:?set EXPERIMENT (lightgbm or multimodel), or use scripts/submit_lightgbm.sh / submit_multimodel.sh}"
VENV=${VENV:-$HOME/venvs/metafe}
py() { PYTHONPATH="$REPO" "$VENV/bin/python" -c "from metafe.config import *; from metafe.models import GPU_MODELS; $1"; }
DATASETS=${DATASETS:-$(py 'print(*sorted(DATASETS))')}
MODELS=${MODELS:-$(py "print(*EXPERIMENTS['$EXPERIMENT'].models)")}
GPU_MODEL_KEYS=$(py 'print(*sorted(GPU_MODELS))')
CORE_TIME=${CORE_TIME:-3-00:00:00}
CORE_CPUS=${CORE_CPUS:-8}
CORE_MEM=${CORE_MEM:-64G}
METAFE_TIME=${METAFE_TIME:-04:00:00}
EVAL_TIME=${EVAL_TIME:-12:00:00}
CPU_OPTS=${CPU_OPTS:-}
GPU_OPTS=${GPU_OPTS:---gres=gpu:1}
STAGES=(core core_merge mf mf_merge metafe check evaluate merge plots)
FROM=${FROM:-core}
UNTIL=${UNTIL:-plots}
n=$(wc -w <<< "$DATASETS")
export REPO VENV DATASETS MODULES="${MODULES:-}"  # passed to the jobs (sbatch --export=ALL)
[ -n "${OPENML_CACHE:-}" ] && export METAFE_OPENML_CACHE=$OPENML_CACHE
mkdir -p logs
echo "Experiment $EXPERIMENT, models: $MODELS, $n datasets, venv $VENV"

stage_index() { local i; for i in "${!STAGES[@]}"; do [ "${STAGES[$i]}" = "$1" ] && echo "$i" && return; done; echo "Unknown stage $1" >&2; exit 1; }
run_stage() { local i; i=$(stage_index "$1"); [ "$i" -ge "$(stage_index "$FROM")" ] && [ "$i" -le "$(stage_index "$UNTIL")" ]; }
device_opts() { if [[ " $GPU_MODEL_KEYS " == *" $1 "* ]]; then echo "$GPU_OPTS"; else echo "$CPU_OPTS"; fi; }

# submit <dependency job IDs "a:b:c" or ""> <sbatch options...> -- <module> [args...]; prints the job ID
submit() {
    local dep=$1; shift
    local opts=()
    while [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift
    [ -n "$dep" ] && opts+=("--dependency=afterok:$dep")
    local export=ALL
    [ -n "${PER_FOLD:-}" ] && export=ALL,PER_FOLD=1
    local cmd=(sbatch --parsable ${SBATCH_OPTS:-} --export=$export --output="logs/%x.%A_%a.out" "${opts[@]}"
               scripts/run_step.sh "$@" --experiment "$EXPERIMENT")
    if [ -n "${DRY_RUN:-}" ]; then
        echo "${cmd[*]}" >&2
        echo "DRYRUN_$(( RANDOM ))"
    else
        "${cmd[@]}"
    fi
}
join() { local IFS=:; echo "$*"; }

last=""
if run_stage core; then
    ids=()
    for model in $MODELS; do
        ids+=("$(submit "$last" -J "core_$model" --array=0-$((n - 1)) -c "$CORE_CPUS" --mem="$CORE_MEM" -t "$CORE_TIME" $(device_opts "$model") -- \
            metafe.core_matrix build --dataset {dataset} --models "$model")")
    done
    last=$(join "${ids[@]}")
    echo "core matrix: $last"
fi
if run_stage core_merge; then
    last=$(submit "$last" -J core_merge -c 1 --mem=16G -t 01:00:00 $CPU_OPTS -- metafe.core_matrix merge)
    echo "core merge: $last"
fi
if run_stage mf; then
    last=$(submit "$last" -J metafeatures --array=0-$((n - 1)) -c 2 --mem=32G -t 06:00:00 $CPU_OPTS -- \
        metafe.metafeatures build --dataset {dataset})
    echo "metafeatures: $last"
fi
if run_stage mf_merge; then
    last=$(submit "$last" -J metafeature_merge -c 1 --mem=32G -t 01:00:00 $CPU_OPTS -- metafe.metafeatures merge)
    echo "metafeature merge: $last"
fi
if run_stage metafe; then
    # MetaFE only uses the surrogate (CPU); 64 GB memory limit inside MetaFE, the rest for the monitor process
    ids=()
    for model in $MODELS; do
        ids+=("$(submit "$last" -J "metafe_$model" --array=0-$((n - 1)) -c 8 --mem=72G -t "$METAFE_TIME" $CPU_OPTS -- \
            metafe.metafe --dataset {dataset} --model "$model")")
    done
    last=$(join "${ids[@]}")
    echo "MetaFE: $last"
fi
if run_stage check; then
    check=$(submit "$last" -J check_fe -c 1 --mem=16G -t 02:00:00 $CPU_OPTS -- metafe.evaluation.check_fe)
    echo "FE check: $check (see logs/check_fe.${check}_*.out)"
fi
if run_stage evaluate; then
    ids=()
    for model in $MODELS; do
        ids+=("$(PER_FOLD=1 submit "$last" -J "evaluate_$model" --array=0-$((3 * n - 1)) -c 8 --mem=64G -t "$EVAL_TIME" \
            $(device_opts "$model") -- metafe.evaluation.evaluate --dataset {dataset} --fold {fold} --model "$model")")
    done
    last=$(join "${ids[@]}")
    echo "evaluation: $last"
fi
if run_stage merge; then
    last=$(submit "$last" -J merge_folds -c 1 --mem=8G -t 00:30:00 $CPU_OPTS -- metafe.evaluation.merge_folds)
    echo "merge folds: $last"
fi
if run_stage plots; then
    last=$(submit "$last" -J plots -c 1 --mem=8G -t 00:30:00 $CPU_OPTS -- metafe.evaluation.plots)
    echo "plots: $last"
fi
