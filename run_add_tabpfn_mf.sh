#!/bin/bash

# Define the partition on which the job shall run.
#SBATCH --partition mlhiwidlc_gpu-rtx2080  # bosch_cpu-cascadelake #

# Define a name for your job
#SBATCH --job-name TabPFN_MF

# Define the files to write the outputs of the job to.
#SBATCH --output logs/%x-%A.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A.out

# Define the amount of memory required per node
#SBATCH --mem=96GB
#SBATCH -c 8
#SBATCH --gres=localtmp:100  # gpu:8 ###

#Time Format = days-hours:minutes:seconds
#SBATCH --time=4-00:00:00

#SBATCH --propagate=NONE



echo "Workingdir: $PWD";
echo "Started at $(date)";

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Print the allocated memory per node
echo "Allocated memory per node: $SLURM_MEM_PER_NODE MB"

# Activate your environment
# shellcheck disable=SC1090
# source ~/miniconda3/bin/activate # Adjust to your path of Miniconda installation
# if conda info --envs | grep -q amltk_env; then echo "amltk_env already exists"; else conda create -y -n amltk_env; fi
# conda activate amltk_env
# echo "conda amltk_env activated"
# virtualenv .venv
pip install uv
uv venv --seed --python 3.11 ~/.venvs/tabarena
source ~/.venvs/tabarena
# source .venv/local/bin/activate
echo "Virtual Environment Activated"

git clone --branch main https://github.com/autogluon/tabrepo.git

pip install -r requirements.txt

# Set the PYTHONPATH to include the src directory
export PYTHONPATH=$PWD/src:$PYTHONPATH
echo "PYTHONPATH set to $PYTHONPATH"

# Running the job
# shellcheck disable=SC2006
start=`date +%s`

# shellcheck disable=SC2048
python3 src/Metadata/tabpfn/Add_TabPFN_Metafeatures.py  "$SLURM_ARRAY_TASK_ID" "$*"

# Print the allocated memory per node
echo "Allocated memory per node: $SLURM_MEM_PER_NODE MB"

deactivate

# shellcheck disable=SC2006
end=`date +%s`
runtime=$((end-start))

echo "Job execution complete."
echo "Runtime: $runtime"
