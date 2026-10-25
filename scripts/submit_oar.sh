#!/bin/bash
# Submit experiment lightgbm_reuse on calculco (OAR; submit from the login node ritchie), CPU only:
#   kb        1 job: knowledge base (reused labels + metafeatures of all datasets, no model fits) + merges
#   metafe    1 job per dataset (after kb): MetaFE, writes every time limit up to 1 h
#   evaluate  1 job per dataset and fold (after the dataset's MetaFE job): LightGBM on the MetaFE data
#             (Original/OpenFE/TabPrep are copied from the cached results of experiment lightgbm)
# Afterwards, on the login node (seconds):
#   python -m metafe.evaluation.merge_folds --experiment lightgbm_reuse
#   python -m metafe.evaluation.plots --experiment lightgbm_reuse
# Usage: DRY_RUN=1 scripts/submit_oar.sh; options: FROM=metafe|evaluate (earlier stages done), DATASETS,
# EXPERIMENT (default lightgbm_reuse), VENV (default ~/venvs/metafe), PROPERTIES (node filter).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
EXPERIMENT=${EXPERIMENT:-lightgbm_reuse}
export VENV=${VENV:-$HOME/venvs/metafe}
# x86 nodes without GPUs (GraceA02 = ARM, the venvs are x86)
PROPERTIES=${PROPERTIES:-"(gpu='' OR gpu IS NULL) AND type='default' AND cputype<>'GraceA02'"}
DATASETS=${DATASETS:-$(PYTHONPATH=. "$VENV/bin/python" -c 'from metafe.config import DATASETS; print(*sorted(DATASETS))')}
METAFE_WALLTIME=$(PYTHONPATH=. "$VENV/bin/python" -c "from metafe.config import EXPERIMENTS
t = max(EXPERIMENTS['$EXPERIMENT'].time_limits) + 3600  # + surrogate training and margin
print(f'{t // 3600}:{t % 3600 // 60:02d}:00')")
FROM=${FROM:-kb}
mkdir -p logs

# submit <name> <resources> <anterior job ID or ""> <command...>; prints the job ID
submit() {
    local name=$1 resources=$2 dep=$3; shift 3
    local cmd=(oarsub -p "$PROPERTIES" -l "$resources" -n "$name" -O "logs/$name.%jobid%.out"
               -E "logs/$name.%jobid%.err" ${dep:+-a "$dep"} "$REPO/scripts/oar_step.sh $*")
    if [ -n "${DRY_RUN:-}" ]; then
        printf '%q ' "${cmd[@]}" >&2; echo >&2
        echo "DRYRUN"
    else
        local id
        id=$("${cmd[@]}" | sed -n 's/^OAR_JOB_ID=//p')
        [[ "$id" =~ ^[0-9]+$ ]] || { echo "oarsub for $name failed, stopping" >&2; kill -TERM $$; }
        echo "$id"
    fi
}

kb=""
if [ "$FROM" = kb ]; then
    kb=$(submit "kb_$EXPERIMENT" core=16,walltime=3:00:00 "" scripts/build_knowledge_base.sh "$EXPERIMENT")
    echo "knowledge base: $kb"
fi
for d in $DATASETS; do
    metafe=""
    if [ "$FROM" != evaluate ]; then
        metafe=$(submit "metafe_$d" "core=8,walltime=$METAFE_WALLTIME" "$kb" \
            python -m metafe.metafe --experiment "$EXPERIMENT" --dataset "$d")
    fi
    eval_ids=()
    for fold in 0 1 2; do
        eval_ids+=("$(submit "evaluate_${d}_$fold" core=8,walltime=6:00:00 "$metafe" \
            python -m metafe.evaluation.evaluate --experiment "$EXPERIMENT" --dataset "$d" --fold "$fold")")
    done
    echo "dataset $d: MetaFE ${metafe:-(done)}, evaluation ${eval_ids[*]}"
done
