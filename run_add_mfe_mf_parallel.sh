#!/bin/bash

# Define the partition on which the job shall run.
#SBATCH --partition bosch_cpu-cascadelake  # mlhiwidlc_gpu-rtx2080

# Define a name for your job
#SBATCH --job-name MFE_Group_MF

# Define the files to write the outputs of the job to.
#SBATCH --output logs/%x-%A_%a.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A.out

# Define the amount of memory required per node
#SBATCH --mem=150GB
#SBATCH -c 8
#SBATCH --gres=localtmp:100

#Time Format = days-hours:minutes:seconds
#SBATCH --time=4-00:00:00

#SBATCH --propagate=NONE

#SBATCH --array=0-6  # Adjust based on the number of methods


echo "Workingdir: $PWD";
echo "Started at $(date)";

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Print the allocated memory per node
echo "Allocated memory per node: $SLURM_MEM_PER_NODE MB"

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

groups=("general" "statistical" "info-theory" "landmarking" "complexity" "clustering" "concept" "itemset")
# dataset_regression=${datasets_regression[$SLURM_ARRAY_TASK_ID]}
group=${groups[$SLURM_ARRAY_TASK_ID]}


# Running the job
# shellcheck disable=SC2006
start=`date +%s`

# shellcheck disable=SC2048
python3 src/Metadata/mfe/Add_MFE_Metafeatures_Parallel.py  --group "$group"

# Print the allocated memory per node
echo "Allocated memory per node: $SLURM_MEM_PER_NODE MB"

deactivate

# shellcheck disable=SC2006
end=`date +%s`
runtime=$((end-start))

echo "Job execution complete."
echo "Runtime: $runtime"
