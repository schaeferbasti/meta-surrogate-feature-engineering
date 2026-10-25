#!/bin/bash
# Submit the same-split rerun: missing FE data first (OpenFE per dataset, missing TabPrep), then the evaluation of
# every dataset in src/Apply_and_Test/test_data/ for folds 0-2. An evaluation job waits (oarsub -a) for the FE job of
# its dataset. CPU only: all jobs are restricted to nodes without GPUs (any CPU type; override with PROPERTIES=...).
#   ./submit_same_split.sh            # submit
#   DRY_RUN=1 ./submit_same_split.sh  # only print the oarsub commands

REPO=/nfs/home/lisic/bschafer/2_metafe/meta-surrogate-feature-engineering
DATA=$REPO/src/Apply_and_Test/test_data
PROPERTIES=${PROPERTIES:-"gpu=0 AND type='default'"}
FOLDS="0 1 2"
# Same list as NO_OPENFE_DATASETS in Test_FE.py (OpenFE did not finish on these)
NO_OPENFE="2073 359930 359931 359935 359938 359959 359962 359983"

cd $REPO
mkdir -p /nfs/home/lisic/bschafer/2_metafe/logs

submit() {
    # Prints the job ID of the submitted job
    if [ -n "$DRY_RUN" ]; then
        echo "oarsub -p \"$PROPERTIES\" $*" >&2
        echo "DRYRUN"
    else
        local id
        id=$(oarsub -p "$PROPERTIES" "$@" | sed -n 's/^OAR_JOB_ID=//p')
        # Stop at the first rejected job, otherwise evaluation jobs would be submitted without their FE dependency
        [[ "$id" =~ ^[0-9]+$ ]] || { echo "oarsub $* failed, stopping" >&2; kill -TERM $$; }
        echo "$id"
    fi
}

datasets=$(ls $DATA | sed -n 's/.*FE_\([0-9]*\)_.*/\1/p' | sort -un)

declare -A fe_job
missing_tabprep=""
for d in $datasets; do
    [ -f $DATA/FE_${d}_TabPrep.parquet ] || missing_tabprep="$missing_tabprep $d"
done
if [ -n "$missing_tabprep" ]; then
    id=$(submit -S "./run_fe_tabprep.oar$missing_tabprep")
    echo "TabPrep FE for$missing_tabprep: job $id"
    for d in $missing_tabprep; do fe_job[$d]=$id; done
fi

for d in $datasets; do
    if [[ " $NO_OPENFE " != *" $d "* ]] && [ ! -f $DATA/FE_${d}_OpenFE.parquet ]; then
        [ -n "${fe_job[$d]}" ] && { echo "Dataset $d needs both OpenFE and TabPrep FE, not supported"; exit 1; }
        fe_job[$d]=$(submit -S "./run_fe_openfe.oar $d")
        echo "OpenFE FE for $d: job ${fe_job[$d]}"
    fi
done

for d in $datasets; do
    dep=${fe_job[$d]:+-a ${fe_job[$d]}}
    for f in $FOLDS; do
        id=$(submit $dep -S "./run_test_fe.oar $d $f")
        echo "Evaluation of $d fold $f: job $id ${dep:+(after job ${fe_job[$d]})}"
    done
done
